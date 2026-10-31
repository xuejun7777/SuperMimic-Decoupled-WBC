# Copyright (c) 2024 Intel Corporation.
# SPDX-License-Identifier: BSD-3-Clause

"""Retargeted AMASS loader for the Unitree G1 teacher policy."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch


class AMassDatasetLoader:
    """Loader for the retargeted AMASS (G1) dataset.

    The dataset released in "Retargeted AMASS for Robotics" stores every motion as a
    ``.npy`` array with shape ``(T, 36)`` and the following layout::

        0:3    → root world position (m)
        3:7    → root quaternion (x, y, z, w)
        7:36   → 29 joint positions following the Unitree G1 XML order

    Columns ``7:36`` already match the joint naming used inside the simulator, so we
    simply slice indices ``15:29`` within this block to obtain the 14 arm joints that
    the teacher policy tracks.
    """

    _ARM_SLICE = slice(15, 29)

    def __init__(self, dataset_path: Optional[Path | str] = None, device: str = "cuda") -> None:
        self.dataset_path = Path(dataset_path).expanduser() if dataset_path else None
        self.device = torch.device(device)

        self.sequences: list[dict[str, Any]] = []
        self.total_frames: int = 0
        self._rng = np.random.default_rng()
        self.is_loaded: bool = False

        if self.dataset_path is not None:
            self._index_dataset()

    def _index_dataset(self) -> None:
        """Discover all retargeted motion files under the dataset root."""

        if not self.dataset_path.exists():
            print(f"[AMassLoader] Dataset path not found: {self.dataset_path}")
            raise ValueError("Invalid AMASS dataset path")

        candidate_roots = [
            self.dataset_path / "dataset" / "g1" / "CMU",
            self.dataset_path / "g1" / "CMU",
        ]
        search_roots = [root for root in candidate_roots if root.exists()]

        for root in search_roots:
            for npy_file in sorted(root.rglob("*_jpos.npy")):
                try:
                    arr = np.load(npy_file, mmap_mode="r")
                    if arr.ndim != 2 or arr.shape[1] < 36:
                        print(f"[AMassLoader] Skipping malformed file: {npy_file}")
                        continue

                   
                    fps = 120  
                    filename = npy_file.stem  
                    if "_60_jpos" in filename:
                        fps = 60
                    elif "_120_jpos" in filename:
                        fps = 120

                    self.sequences.append({
                        "path": npy_file,
                        "length": int(arr.shape[0]),
                        "cached_length": int(arr.shape[0]),
                        "fps": fps,  
                    })
                    self.total_frames += int(arr.shape[0])
                except Exception as exc:  # pragma: no cover - diagnostic path
                    print(f"[AMassLoader] Failed to index {npy_file}: {exc}")
                finally:
                    arr = None

        self.is_loaded = len(self.sequences) > 0
        if self.is_loaded:
            
            lengths = [seq["length"] for seq in self.sequences]
            fps_values = [seq["fps"] for seq in self.sequences]
            
            len_60 = sum(1 for l in lengths if l <= 60)
            len_120 = sum(1 for l in lengths if 60 < l <= 120)
            len_longer = sum(1 for l in lengths if l > 120)
            
            fps_60_count = sum(1 for f in fps_values if f == 60)
            fps_120_count = sum(1 for f in fps_values if f == 120)
            
            print(
                f"[AMassLoader] Indexed {len(self.sequences)} sequences "
                f"({self.total_frames} frames) from {self.dataset_path}"
            )
            print(
                f"[AMassLoader] Length distribution: "
                f"≤60 frames: {len_60}, 61-120 frames: {len_120}, >120 frames: {len_longer}"
            )
            print(
                f"[AMassLoader] FPS distribution: "
                f"60 FPS: {fps_60_count}, 120 FPS: {fps_120_count}"
            )
            print(
                f"[AMassLoader] Mean length: {np.mean(lengths):.1f} frames"
            )
        else:
            print(f"[AMassLoader] No sequences found under {self.dataset_path}")

    def sample_arm_sequence(self, num_envs: int, sequence_length: int = 1, interpolation_factor: int = 2) -> torch.Tensor:
        """Sample G1 arm targets of shape ``(num_envs, sequence_length, 14)``.
        
        Args:
            num_envs: Number of environments
            sequence_length: Target output sequence length after interpolation
            interpolation_factor: Number of interpolated frames between two original frames
                                 (e.g., 2 means insert 0 frames between each pair, total 2 frames per segment)
        
        Returns:
            torch.Tensor: Shape (num_envs, sequence_length, 14)
        """

        if sequence_length <= 0:
            raise ValueError("sequence_length must be >= 1")

        
        original_frames_needed = (sequence_length + interpolation_factor - 1) // interpolation_factor + 1
        
        batch = np.zeros((num_envs, sequence_length, 14), dtype=np.float32)

        # Choose random sequences for each environment.
        seq_indices = self._rng.integers(0, len(self.sequences), size=num_envs)

        # Group env indices by source file to minimise disk reads.
        file_to_envs: Dict[Path, list[tuple[int, int]]] = {}
        for env_idx, seq_idx in enumerate(seq_indices):
            meta = self.sequences[seq_idx]
            length = meta["length"]
            if length <= 0:
                continue

            max_start = max(length - original_frames_needed, 0)
            start_idx = int(self._rng.integers(0, max_start + 1)) if length > original_frames_needed else 0
            file_to_envs.setdefault(meta["path"], []).append((env_idx, start_idx))

        for path, env_entries in file_to_envs.items():
            arr = np.load(path, mmap_mode="r")
            arm_data = np.asarray(arr[:, 7:][:, self._ARM_SLICE], dtype=np.float32)

            for env_idx, start_idx in env_entries:
                end_idx = start_idx + original_frames_needed
                original_seq = arm_data[start_idx:end_idx]

                
                if original_seq.shape[0] < original_frames_needed:
                    if original_seq.shape[0] == 0:
                        original_seq = np.zeros((original_frames_needed, 14), dtype=np.float32)
                    else:
                        padding = np.repeat(original_seq[-1:], original_frames_needed - original_seq.shape[0], axis=0)
                        original_seq = np.concatenate([original_seq, padding], axis=0)

                
                interpolated_seq = self._interpolate_sequence(original_seq, interpolation_factor)
                
                
                interpolated_seq = interpolated_seq[:sequence_length]
                
                
                if interpolated_seq.shape[0] < sequence_length:
                    padding = np.repeat(interpolated_seq[-1:], sequence_length - interpolated_seq.shape[0], axis=0)
                    interpolated_seq = np.concatenate([interpolated_seq, padding], axis=0)
                
                batch[env_idx] = interpolated_seq

            del arr

        return torch.from_numpy(batch).to(self.device)
    
    def _interpolate_sequence(self, sequence: np.ndarray, interpolation_factor: int) -> np.ndarray:
        
        if len(sequence) <= 1:
            return sequence
        
        N = len(sequence)
        
        if interpolation_factor > 1:
            alphas = np.linspace(0, 1, interpolation_factor, dtype=np.float32)[:-1]  # excludes 1.0
        else:
            alphas = np.array([0.0], dtype=np.float32)
        
        
        frame_a = sequence[:-1]  # shape: (N-1, 14)
        frame_b = sequence[1:]   # shape: (N-1, 14)
        
        
        interpolated = (1 - alphas[:, None, None]) * frame_a[None, :, :] + alphas[:, None, None] * frame_b[None, :, :]
        
        
        interpolated = interpolated.transpose(1, 0, 2).reshape(-1, 14)
        
        
        result = np.concatenate([interpolated, sequence[-1:]], axis=0)
        
        return result

    def sample_arm_clips(
        self, 
        num_envs: int, 
        clip_duration_frames: int,
        control_dt: float = 0.02,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample full AMASS clips and resample to control frequency.
        
        
        
        Args:
            num_envs: Number of environments
            clip_duration_frames: Maximum clip length in control timesteps
            control_dt: Control timestep (e.g., 0.02 = 50Hz)
            
        Returns:
            trajectories: (num_envs, clip_duration_frames, 14) arm joint positions
            clip_lengths: (num_envs,) actual clip length for each env
            start_offsets: (num_envs,) random starting frame offset
        """
        import scipy.interpolate
        
        # Choose random sequences for each environment
        seq_indices = self._rng.integers(0, len(self.sequences), size=num_envs)
        
        trajectories = torch.zeros((num_envs, clip_duration_frames, 14), device=self.device, dtype=torch.float32)
        clip_lengths = torch.zeros(num_envs, device=self.device, dtype=torch.long)
        start_offsets = torch.zeros(num_envs, device=self.device, dtype=torch.long)
        
        for env_idx, seq_idx in enumerate(seq_indices):
            meta = self.sequences[seq_idx]
            path = meta["path"]
            original_length = meta["length"]
            amass_fps = meta["fps"]  
            
            if original_length <= 0:
                continue
                
            # Load full sequence
            arr = np.load(path, mmap_mode="r")
            arm_data = np.asarray(arr[:, 7:][:, self._ARM_SLICE], dtype=np.float32)
            
           
            sequence_duration = original_length / amass_fps
            
            
            if sequence_duration >= 2.0:
               
                min_duration = 1.0
                max_offset_duration = sequence_duration - min_duration
                max_offset = int(max_offset_duration * amass_fps)
            elif sequence_duration >= 1.0:
               
                min_duration = 0.5
                max_offset_duration = sequence_duration - min_duration
                max_offset = int(max_offset_duration * amass_fps)
            else:
                
                max_offset = 0
            
            t0 = int(self._rng.integers(0, max_offset + 1)) if max_offset > 0 else 0
            
            # Extract clip from t0
            clip_data = arm_data[t0:]
            
            # Resample to control frequency
            # Original time points
            original_time = np.arange(len(clip_data)) / amass_fps
            # Target time points at control frequency
            target_frames = int(original_time[-1] / control_dt) if len(original_time) > 1 else 1
            target_frames = min(target_frames, clip_duration_frames)
            target_time = np.arange(target_frames) * control_dt
            
            # Interpolate all 14 joints at once (vectorized)
            interp_func = scipy.interpolate.interp1d(
                original_time, 
                clip_data,  # shape: (T, 14) - interpolate all joints together
                kind='linear',
                axis=0,  # interpolate along time axis
                fill_value='extrapolate'
            )
            resampled = interp_func(target_time).astype(np.float32)
            
            # Store results
            trajectories[env_idx, :target_frames] = torch.from_numpy(resampled)
            clip_lengths[env_idx] = target_frames
            start_offsets[env_idx] = t0
            
            del arr
        
        return trajectories, clip_lengths, start_offsets


   

    def get_dataset_info(self) -> Dict[str, Any]:
        return {
            "is_loaded": self.is_loaded,
            "dataset_path": str(self.dataset_path) if self.dataset_path else None,
            "num_sequences": len(self.sequences),
            "total_frames": self.total_frames,
            "device": str(self.device),
            "placeholder_mode": not self.is_loaded,
        }


_global_amass_loader: Optional[AMassDatasetLoader] = None
_global_loader_key: Dict[str, Any] = {"dataset_path": None, "device": None}


def get_amass_loader(dataset_path: Optional[Path | str] = None, device: str = "cuda") -> AMassDatasetLoader:
    """Return a cached loader, reinitialising when the dataset path changes."""

    global _global_amass_loader, _global_loader_key

    resolved_path = Path(dataset_path).expanduser() if dataset_path else _global_loader_key["dataset_path"]
    device_str = str(device)

    path_changed = resolved_path is not None and resolved_path != _global_loader_key["dataset_path"]
    device_changed = device_str != _global_loader_key["device"]

    if _global_amass_loader is None or path_changed:
        _global_amass_loader = AMassDatasetLoader(resolved_path, device_str)
        _global_loader_key = {"dataset_path": resolved_path, "device": device_str}
    else:
        if device_changed and _global_amass_loader is not None:
            _global_amass_loader.device = torch.device(device_str)
            _global_loader_key["device"] = device_str

    return _global_amass_loader


def sample_amass_arm_poses(
    num_envs: int,
    sequence_length: int=1,
    device: str = "cuda",
    dataset_path: Optional[Path | str] = None,
    interpolation_factor: int = 4,
) -> torch.Tensor:
    
    loader = get_amass_loader(dataset_path=dataset_path, device=device)
    result = loader.sample_arm_sequence(num_envs, sequence_length=sequence_length, interpolation_factor=interpolation_factor)
    
    
    if sequence_length == 1:
        return result.squeeze(1)
    return result

if __name__ == "__main__":
    # Example usage
    dataset_path = "g1"

    poses = sample_amass_arm_poses(num_envs=4, device="cuda", dataset_path=dataset_path)
    print(poses.shape)  # (4, 14)
    print(poses)   