"""Average raw data_1 coherent-state decay measurements into compact NPZ files.

The output files follow the ``delay_*_averaged_cf.npz`` schema used by
``helpers.io.load_processed_npz``.  Real and imaginary HDF5 files are paired
by their recorded ``wait_for_decay`` attribute, not by filename or timestamp.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np


DEFAULT_AXIS_SCALE = 1.603359375


def _records_by_delay(directory: Path, missing_delay: int = 0) -> dict[int, Path]:
    records: dict[int, Path] = {}
    missing_metadata: list[Path] = []
    for path in sorted(directory.glob("*.hdf5")):
        with h5py.File(path, "r") as file:
            delay = file.attrs.get("wait_for_decay")
        if delay is None:
            missing_metadata.append(path)
            continue
        delay = int(delay)
        if delay in records:
            raise ValueError(f"Duplicate wait_for_decay={delay} in {directory}")
        records[delay] = path
    if missing_metadata:
        if len(missing_metadata) != 1 or missing_delay in records:
            names = ", ".join(path.name for path in missing_metadata)
            raise ValueError(f"Cannot infer the delay for files without metadata in {directory}: {names}")
        # The data_1 zero-delay acquisition is the sole file with no
        # wait_for_decay attribute in each component directory.
        records[missing_delay] = missing_metadata[0]
    if not records:
        raise FileNotFoundError(f"No HDF5 files found in {directory}")
    return records


def _average_file(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    with h5py.File(path, "r") as file:
        samples = np.asarray(file["I"], dtype=float)
        x_axis = np.asarray(file["x_displace"], dtype=float)
        y_axis = np.asarray(file["y_displace"], dtype=float)
    if samples.ndim != 3:
        raise ValueError(f"{path} has I shape {samples.shape}; expected (repetition, y, x)")
    return samples.mean(axis=0), samples.std(axis=0), x_axis, y_axis


def _central_crop(image: np.ndarray, crop_size: int) -> np.ndarray:
    if crop_size <= 0 or crop_size > min(image.shape) or crop_size % 2 == 0:
        raise ValueError("crop_size must be a positive odd number no larger than the image dimensions")
    y_start = (image.shape[0] - crop_size) // 2
    x_start = (image.shape[1] - crop_size) // 2
    return image[y_start : y_start + crop_size, x_start : x_start + crop_size]


def _central_axis(axis: np.ndarray, crop_size: int) -> np.ndarray:
    start = (len(axis) - crop_size) // 2
    return axis[start : start + crop_size]


def average_data(raw_directory: Path, output_directory: Path, crop_size: int, axis_scale: float) -> None:
    real_files = _records_by_delay(raw_directory / "data_real")
    imag_files = _records_by_delay(raw_directory / "data_img")
    if real_files.keys() != imag_files.keys():
        raise ValueError("Real and imaginary files do not have the same wait_for_decay values")

    output_directory.mkdir(parents=True, exist_ok=True)
    for delay_index, delay in enumerate(sorted(real_files)):
        chi_real, chi_real_std, x_real, y_real = _average_file(real_files[delay])
        chi_imag, chi_imag_std, x_imag, y_imag = _average_file(imag_files[delay])
        if not (np.allclose(x_real, x_imag) and np.allclose(y_real, y_imag)):
            raise ValueError(f"Real and imaginary grids differ for wait_for_decay={delay}")

        alpha_real = axis_scale * _central_axis(x_real, crop_size)
        alpha_imag = axis_scale * _central_axis(y_real, crop_size)
        chi_real = _central_crop(chi_real, crop_size)
        chi_imag = _central_crop(chi_imag, crop_size)
        chi_real_std = _central_crop(chi_real_std, crop_size)
        chi_imag_std = _central_crop(chi_imag_std, crop_size)
        centre = crop_size // 2

        np.savez_compressed(
            output_directory / f"delay_{delay_index}_averaged_cf.npz",
            delay_index=delay_index,
            wait_for_decay_ns=delay,
            pair_indices=np.array([0], dtype=int),
            alpha_real=alpha_real,
            alpha_imag=alpha_imag,
            chi_real=chi_real,
            chi_imag=chi_imag,
            chi_real_std=chi_real_std,
            chi_imag_std=chi_imag_std,
            scale=float(axis_scale),
            beta=float(axis_scale / 2),
            amplitude_real=float(np.mean(chi_real)),
            amplitude_imag=float(np.mean(chi_imag)),
            crosshair_offset_real=float(chi_real[centre, centre]),
            crosshair_offset_imag=float(chi_imag[centre, centre]),
        )
        print(f"Wrote delay_{delay_index}_averaged_cf.npz for {delay} ns")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("raw_directory", type=Path, help="Path to raw data_1 containing data_real and data_img")
    parser.add_argument("output_directory", type=Path, help="Directory for averaged NPZ files")
    parser.add_argument("--crop-size", type=int, default=23, help="Odd central grid size to retain (default: 23)")
    parser.add_argument("--axis-scale", type=float, default=DEFAULT_AXIS_SCALE, help="Displacement-axis calibration scale")
    args = parser.parse_args()
    average_data(args.raw_directory, args.output_directory, args.crop_size, args.axis_scale)


if __name__ == "__main__":
    main()
