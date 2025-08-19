"""dataloader.py"""

import os
import copy
import math
import numpy as np
from glob import glob
from typing import Optional, Sequence, Tuple, Dict, List

import torch
import torchio as tio
from torchio.utils import to_tuple
from torchio.data.subject import Subject
from torchio.transforms.intensity_transform import IntensityTransform

class RandomPatchZeroOut(IntensityTransform):

    """
    Randomly zero-out non-overlapping patches until ~mask_ratio of voxels are masked.
    Args:
        patch_size: int or (w,h,d) in voxels.
        mask_ratio: float in (0,1], target fraction of voxels to zero (approximate).
        max_trials: cap on sampling attempts for non-overlapping placement.
        zero_value: value written inside patches (default 0.0).
        **kwargs: TorchIO Transform kwargs (e.g., include=['mri'])
    """

    def __init__(
        self,
        patch_size: int | Tuple[int, int, int] = 16,
        mask_ratio: float = 0.2,
        max_trials: int = 10000,
        zero_value: float = 0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        ps = np.array(to_tuple(patch_size))
        if ps.size == 1:
            ps = np.array([ps.item(), ps.item(), ps.item()], dtype=np.int64)
        self.patch_size = ps.astype(np.int64)
        if not (0 < mask_ratio <= 1):
            raise ValueError("mask_ratio must be in (0,1].")
        self.mask_ratio = float(mask_ratio)
        self.max_trials = int(max_trials)
        self.zero_value = float(zero_value)

    @staticmethod
    def _random_patch_start(spatial_shape: Sequence[int], patch_size: np.ndarray) -> np.ndarray:
        si, sj, sk = spatial_shape
        pi, pj, pk = patch_size
        if pi > si or pj > sj or pk > sk:
            raise ValueError(f"Patch size {patch_size.tolist()} exceeds image shape {spatial_shape}.")
        i0 = 0 if si - pi == 0 else int(torch.randint(si - pi + 1, size=(1,)).item())
        j0 = 0 if sj - pj == 0 else int(torch.randint(sj - pj + 1, size=(1,)).item())
        k0 = 0 if sk - pk == 0 else int(torch.randint(sk - pk + 1, size=(1,)).item())
        return np.array([i0, j0, k0], dtype=np.int64)

    @staticmethod
    def _overlaps(a0: np.ndarray, a1: np.ndarray, b0: np.ndarray, b1: np.ndarray) -> bool:
        return not (
            (a1[0] <= b0[0] or b1[0] <= a0[0]) or
            (a1[1] <= b0[1] or b1[1] <= a0[1]) or
            (a1[2] <= b0[2] or b1[2] <= a0[2])
        )

    @staticmethod
    def _insert_zero(tensor: torch.Tensor, start: np.ndarray, size: np.ndarray, zero_value: float):
        i0, j0, k0 = start.tolist()
        di, dj, dk = size.tolist()
        tensor[:, i0:i0+di, j0:j0+dj, k0:k0+dk] = zero_value

    def _plan_locations(
        self,
        spatial_shape: Tuple[int, int, int],
        patch_size: np.ndarray,
        mask_ratio: float,
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        si, sj, sk = spatial_shape
        total_vox = si * sj * sk
        patch_vox = int(np.prod(patch_size))
        target_patches = max(1, math.ceil(mask_ratio * total_vox / patch_vox))

        locations: List[Tuple[np.ndarray, np.ndarray]] = []
        trials = 0
        while len(locations) < target_patches and trials < self.max_trials:
            trials += 1
            start = self._random_patch_start(spatial_shape, patch_size)
            end = start + patch_size
            if all(not self._overlaps(start, end, s0, e0) for (s0, e0) in locations):
                locations.append((start, end))
        return locations 

    def apply_transform(self, subject: Subject) -> Subject:
        images_dict: Dict[str, tio.Image] = self.get_images_dict(subject)
        if not images_dict:
            return subject

        for _, image in images_dict.items():
            data = image.data  # (C, I, J, K)
            si, sj, sk = data.shape[-3:]
            locations = self._plan_locations((si, sj, sk), self.patch_size, self.mask_ratio)
            new_data = data.clone()
            for (start, end) in locations:
                size = end - start
                self._insert_zero(new_data, start, size, self.zero_value)
            image.set_data(new_data)
        return subject

class SSLPretextDataset(tio.data.SubjectsDataset):

    """
    Returns (clean, masked) tensors for SSL.
    - clean: base_transform(subj)
    - masked: base_transform(subj) -> mask_transform(...)
    """

    def __init__(
        self,
        root_dir: str,
        base_transform: tio.Transform | None = None,
        image_glob_pattern: str = "**t2f.nii.gz",
        mask_transform: tio.Transform | None = None,
        build_subjects_fn=None,
    ):
        self.root_dir = root_dir
        self.base_transform = base_transform
        self.mask_transform = mask_transform

        if build_subjects_fn is not None:
            subjects = build_subjects_fn(root_dir)
        else:
            img_paths = sorted(glob(os.path.join(root_dir, image_glob_pattern), recursive=True))
            subjects = [tio.Subject(mri=tio.ScalarImage(p)) for p in img_paths]

        self._source_subjects = subjects
        super().__init__(subjects, transform=None)

    def __len__(self) -> int:
        return len(self._source_subjects)

    def __getitem__(self, index: int):
        # A single subject instance is loaded from the list.
        # This is a lightweight object that just holds metadata (e.g., file paths).
        subject = self._source_subjects[index]

        # The base transform (which includes loading from disk) is applied once.
        if self.base_transform is not None:
            subj_clean = self.base_transform(subject)
        else:
            # If no transform, ensure data is loaded into the subject.
            subj_clean = subject.load()

        # The transformed subject (with data now in memory) is copied for masking.
        subj_masked = copy.deepcopy(subj_clean)
        if self.mask_transform is not None:
            subj_masked = self.mask_transform(subj_masked)

        clean = subj_clean["mri"][tio.DATA]   # (C, I, J, K)
        masked = subj_masked["mri"][tio.DATA] # (C, I, J, K)
        return {"clean": clean, "masked": masked}