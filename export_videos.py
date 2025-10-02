import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch


from isaaclab.utils.datasets import EpisodeData, HDF5DatasetFileHandler

try:
    import cv2  # OpenCV is widely available; used for robust MP4 writing
except ImportError as e:
    raise ImportError("This exporter requires OpenCV (cv2). Please `pip install opencv-python`.") from e


def _infer_fps_from_env_args(env_args: Dict, default_fps: int = 30) -> int:
    """
    Try to infer FPS from env_args saved in the HDF5 root group.
    Common keys we try: 'fps', 'camera_fps', 'frame_hz', 'step_hz'.
    """
    if not isinstance(env_args, dict):
        return default_fps
    for k in ["fps", "camera_fps", "frame_hz", "step_hz"]:
        if k in env_args:
            try:
                v = float(env_args[k])
                if v > 0:
                    return int(round(v))
            except Exception:
                pass
    return default_fps


def _to_uint8_rgb(frames: np.ndarray) -> np.ndarray:
    """
    Normalize/convert frames to uint8 RGB [T, H, W, 3].
    Accepts float in [0,1] or [0,255], or uint8. Handles gray channels.
    """
    if frames.dtype != np.uint8:
        # assume float-ish
        fmax = frames.max()
        # Heuristic: if <=1.0, scale by 255; else clip to [0,255]
        if fmax <= 1.0001:
            frames = (frames * 255.0).round()
        frames = np.clip(frames, 0, 255).astype(np.uint8)

    # Gray -> 3-channel
    if frames.ndim == 4 and frames.shape[-1] == 1:
        frames = np.repeat(frames, 3, axis=-1)
    return frames


def _ensure_T_H_W_C(t: torch.Tensor) -> Optional[np.ndarray]:
    """
    Accept torch tensor shaped either [T,H,W,C] or [T,C,H,W] (C in {1,3}),
    return np.ndarray [T,H,W,C]. Return None if shape not image-like.
    """
    if not isinstance(t, torch.Tensor):
        return None
    if t.ndim != 4:
        return None

    T, A, B, C = t.shape
    # Case 1: [T, H, W, C]
    if C in (1, 3):
        frames = t.detach().cpu().numpy()
        return frames
    # Case 2: [T, C, H, W]
    if A in (1, 3):
        frames = t.detach().cpu().numpy().transpose(0, 2, 3, 1)
        return frames

    return None


def _pad_to_even_hw(frames: np.ndarray) -> Tuple[np.ndarray, Tuple[int, int]]:
    """
    Pad frames on the bottom/right to even height/width if needed.
    Returns (padded_frames, (new_h, new_w)).
    """
    T, H, W, C = frames.shape
    new_h = H + (H % 2)
    new_w = W + (W % 2)
    if new_h == H and new_w == W:
        return frames, (H, W)
    pad_h = new_h - H
    pad_w = new_w - W
    # Pad with zeros (black)
    padded = np.pad(frames, ((0, 0), (0, pad_h), (0, pad_w), (0, 0)), mode="edge")
    return padded, (new_h, new_w)


def _key_path_join(prefix: str, key: str) -> str:
    return f"{prefix}/{key}" if prefix else key


def _scan_image_tensors(nested: Dict, prefix: str = "") -> List[Tuple[str, torch.Tensor]]:
    """
    Recursively scan a nested dict for image-like tensors.
    Return list of (key_path, tensor) where tensor is 4D and looks like images.
    """
    found: List[Tuple[str, torch.Tensor]] = []
    for k, v in nested.items():
        kp = _key_path_join(prefix, k)
        if isinstance(v, dict):
            found.extend(_scan_image_tensors(v, kp))
        elif isinstance(v, torch.Tensor) and v.ndim == 4:
            # light filter by name to prefer images
            name_hint = k.lower()
            if any(h in name_hint for h in ["image", "rgb", "camera", "frame", "color", "left", "right"]):
                found.append((kp, v))
            else:
                # still accept; shape check will happen later
                found.append((kp, v))
    return found


def export_episode_videos(
    hdf5_path: str,
    out_dir: str,
    *,
    handler_cls,
    device: str = "cpu",
    fps: Optional[int] = None,
    codec: str = "mp4v",
    cameras: Optional[List[str]] = None,
    image_keys: Optional[List[str]] = None,
    overwrite: bool = True,
    pad_to_even: bool = True,
) -> str:
    """
    Export all image-type observations into per-episode MP4 videos.

    Args:
        hdf5_path: Path to the dataset .hdf5 file.
        out_dir: Directory to write videos (created if needed).
        handler_cls: The dataset file handler class (e.g., HDF5DatasetFileHandler).
        device: 'cpu' or 'cuda' for loading tensors.
        fps: If provided, overrides inferred FPS. Otherwise inferred from env_args or defaults to 30.
        codec: FourCC string for OpenCV (e.g., 'mp4v', 'avc1', 'H264'). 'mp4v' is widely supported.
        cameras: If provided, only keys containing any of these substrings (case-insensitive) are exported.
        image_keys: If provided, only exact key paths in this list are exported (takes precedence over `cameras`).
        overwrite: If False, skip writing if the target .mp4 already exists.
        pad_to_even: If True, pad width/height to even numbers (safer for some encoders).

    Returns:
        Path to a manifest.json summarizing all exported videos.
    """
    os.makedirs(out_dir, exist_ok=True)

    handler = handler_cls()
    handler.open(hdf5_path, mode="r")

    # Read env_args (stored as JSON string in root group attrs)
    root_group = handler._hdf5_data_group  # safe here; open() guarantees presence
    env_args = {}
    if "env_args" in root_group.attrs:
        try:
            env_args = json.loads(root_group.attrs["env_args"])
        except Exception:
            env_args = {}

    inferred_fps = fps if fps is not None else _infer_fps_from_env_args(env_args, default_fps=30)

    # Iterate episodes
    manifest = {
        "dataset": os.path.abspath(hdf5_path),
        "output_dir": os.path.abspath(out_dir),
        "env_args": env_args,
        "default_fps_used": inferred_fps,
        "videos": [],  # each item: {episode, key_path, file, fps, shape}
    }

    for ep_name in handler.get_episode_names():
        ep = handler.load_episode(ep_name, device=device)
        if ep is None or ep.is_empty():
            continue

        # Discover all 4D image-like tensors
        candidates = _scan_image_tensors(ep.data)

        # Optional filtering by explicit image_keys
        if image_keys:
            key_set = set(image_keys)
            candidates = [(k, v) for (k, v) in candidates if k in key_set]

        # Optional filtering by camera name hints
        if cameras and not image_keys:
            hints = [c.lower() for c in cameras]
            def match_any(hay: str) -> bool:
                h = hay.lower()
                return any(hint in h for hint in hints)
            candidates = [(k, v) for (k, v) in candidates if match_any(k)]

        # Export each candidate
        for key_path, tensor in candidates:
            arr = _ensure_T_H_W_C(tensor)
            if arr is None:
                continue  # not an image-like 4D tensor we recognize

            # Convert to uint8 RGB [T,H,W,3]
            frames = _to_uint8_rgb(arr)

            # Ensure even dims if requested
            if pad_to_even:
                frames, (H, W) = _pad_to_even_hw(frames)
            else:
                _, H, W, _ = frames.shape

            # OpenCV expects BGR; convert per-frame (cheap, vectorized split/merge)
            # We'll do it on the fly to save memory
            # Prepare writer
            safe_key = key_path.replace("/", "-")
            video_name = f"{ep_name}__{safe_key}.mp4"
            video_path = os.path.join(out_dir, video_name)
            if (not overwrite) and os.path.exists(video_path):
                manifest["videos"].append({
                    "episode": ep_name,
                    "key_path": key_path,
                    "file": os.path.abspath(video_path),
                    "fps": inferred_fps,
                    "shape": [int(s) for s in frames.shape],
                    "skipped": True,
                    "reason": "exists",
                })
                continue

            fourcc = cv2.VideoWriter_fourcc(*codec)
            writer = cv2.VideoWriter(video_path, fourcc, float(inferred_fps), (W, H))
            if not writer.isOpened():
                raise RuntimeError(f"Failed to open VideoWriter for {video_path} with codec {codec}")

            try:
                # Write frames
                # RGB -> BGR
                for f in frames:
                    bgr = cv2.cvtColor(f, cv2.COLOR_RGB2BGR)
                    writer.write(bgr)
            finally:
                writer.release()

            manifest["videos"].append({
                "episode": ep_name,
                "key_path": key_path,
                "file": os.path.abspath(video_path),
                "fps": inferred_fps,
                "shape": [int(s) for s in frames.shape],
                "skipped": False,
            })

    # Save manifest
    manifest_path = os.path.join(out_dir, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    handler.close()
    return manifest_path



if __name__ == '__main__':
    # 假设你的 handler 类名就是 HDF5DatasetFileHandler，并与此函数在同一模块
    manifest_path = export_episode_videos(
        hdf5_path="/home/wgk/code/IsaacLab2/IsaacLab/datasets/dataset.hdf5",
        out_dir="/home/wgk/code/IsaacLab2/IsaacLab/datasets/videos",
        handler_cls=HDF5DatasetFileHandler,
        device="cpu",           # 或 "cuda"
        fps=None,               # None 表示自动从 env_args 推断（优先找 fps/camera_fps/frame_hz/step_hz），找不到则 30
        codec="mp4v",           # 通用、默认
        cameras=None,           # 例如 ["wrist", "left", "right", "top"]
        image_keys=None,        # 指定精确的 key 路径时可用，如 ["observations/images/wrist", ...]
        overwrite=True,         # 已存在视频是否覆盖
        pad_to_even=True,       # 遇到奇数分辨率时自动填充到偶数
    )
    print("Wrote manifest to:", manifest_path)
