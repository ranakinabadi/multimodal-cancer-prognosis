"""
DICOM loading, resampling, HU clipping, and tensor saving.
"""

import os
import numpy as np
import torch
import SimpleITK as sitk
from pathlib import Path
from typing import Optional


HU_MIN      = -1000
HU_MAX      = 400
TARGET_SIZE = (64, 64, 64)


def load_dicom_series(dicom_dir: str | Path) -> Optional[sitk.Image]:
    """load a DICOM series from a directory. Returns None if no DICOMs found."""
    dicom_dir  = str(dicom_dir)
    reader     = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(dicom_dir)
    if len(dicom_names) == 0:
        return None
    reader.SetFileNames(dicom_names)
    return reader.Execute()


def resample_to_isotropic(image: sitk.Image,
                           new_spacing: list[float] = [1.0, 1.0, 1.0]) -> sitk.Image:
    """resample image to isotropic 1mm spacing."""
    original_spacing = image.GetSpacing()
    original_size    = image.GetSize()

    new_size = [
        int(round(original_size[i] * original_spacing[i] / new_spacing[i]))
        for i in range(3)
    ]

    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(new_spacing)
    resampler.SetSize(new_size)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetOutputDirection(image.GetDirection())
    resampler.SetOutputOrigin(image.GetOrigin())
    return resampler.Execute(image)


def resize_to_target(image: sitk.Image,
                      target_size: tuple[int, int, int] = TARGET_SIZE) -> sitk.Image:
    """resize image to fixed target voxel dimensions."""
    current_size    = image.GetSize()
    current_spacing = image.GetSpacing()

    new_spacing = [
        current_spacing[i] * current_size[i] / target_size[i]
        for i in range(3)
    ]

    resampler = sitk.ResampleImageFilter()
    resampler.SetSize(list(target_size))
    resampler.SetOutputSpacing(new_spacing)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetOutputDirection(image.GetDirection())
    resampler.SetOutputOrigin(image.GetOrigin())
    return resampler.Execute(image)


def preprocess_volume(dicom_dir: str | Path,
                       target_size: tuple[int, int, int] = TARGET_SIZE,
                       hu_min: int = HU_MIN,
                       hu_max: int = HU_MAX) -> Optional[np.ndarray]:
    """
    Full preprocessing pipeline for one patient.
    Returns float32 numpy array in [0, 1], shape (D, H, W), or None on failure.
    """
    image = load_dicom_series(dicom_dir)
    if image is None:
        return None

    image = resample_to_isotropic(image)
    image = resize_to_target(image, target_size)

    arr = sitk.GetArrayFromImage(image).astype(np.float32)
    arr = np.clip(arr, hu_min, hu_max)
    arr = (arr - hu_min) / (hu_max - hu_min)   # [0, 1]
    return arr


def find_dicom_series(root_dir: str | Path, min_slices: int = 10) -> list[str]:
    """walk a directory tree and return paths containing DICOM files."""
    series_dirs = []
    for dirpath, _, files in os.walk(str(root_dir)):
        dcm_files = [f for f in files if f.lower().endswith(".dcm")]
        if len(dcm_files) >= min_slices:
            series_dirs.append(dirpath)
    return sorted(series_dirs)


def preprocess_all(raw_dir: str | Path,
                   tensor_dir: str | Path,
                   target_size: tuple[int, int, int] = TARGET_SIZE,
                   overwrite: bool = False) -> dict:
    """
    Preprocess all DICOM series in raw_dir and save as .pt tensors.

    Returns a summary dict: {processed: int, skipped: int, failed: list}
    """
    raw_dir    = Path(raw_dir)
    tensor_dir = Path(tensor_dir)
    tensor_dir.mkdir(parents=True, exist_ok=True)

    series_dirs = find_dicom_series(raw_dir)
    print(f"Found {len(series_dirs)} DICOM series in {raw_dir}")

    processed, skipped, failed = 0, 0, []

    for i, series_path in enumerate(series_dirs):
        patient_id = f"patient_{i:03d}"
        out_path   = tensor_dir / f"{patient_id}.pt"

        if out_path.exists() and not overwrite:
            skipped += 1
            continue

        arr = preprocess_volume(series_path, target_size)
        if arr is None:
            print(f"  FAILED: {series_path}")
            failed.append(series_path)
            continue

        tensor = torch.tensor(arr).unsqueeze(0)   # (1, D, H, W)
        torch.save(tensor, out_path)
        processed += 1

        if (i + 1) % 10 == 0:
            print(f"  Processed {i+1}/{len(series_dirs)}")

    summary = {"processed": processed, "skipped": skipped, "failed": failed}
    print(f"\nDone — processed: {processed}, skipped: {skipped}, failed: {len(failed)}")
    return summary
