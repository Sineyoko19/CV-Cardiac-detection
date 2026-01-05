import cv2
import pydicom
import numpy as np
import pandas as pd
from pathlib import Path


def transform_data(data: pd.DataFrame, root_path: str, save_path: str) -> None:
    """
    Transforms DICOM medical images into resized, normalized NumPy arrays,
    splits them into training and validation sets, and computes dataset statistics.

    Parameters
    ----------
    data : pd.DataFrame
        A DataFrame containing patient identifiers under the column 'name'.
    root_path : str or pathlib.Path
        Path to the directory containing the raw DICOM (.dcm) files.
    save_path : str or pathlib.Path
        Path to the directory where transformed NumPy arrays will be saved.
        The data will be organized into 'train' and 'val' subfolders.

    Notes
    -----
    - Each DICOM file is resized to (224, 224) pixels and normalized to the range [0, 1].
    - The first 400 images (by index) are assigned to the training set;
      the rest are assigned to the validation set.
    - The function saves each processed image as a `.npy` file in the corresponding subfolder.
    """

    train_ids, val_ids = [], []

    for i, patient_id in enumerate(list(data["name"])):
        dcm_path = (root_path / str(patient_id)).with_suffix(
            ".dcm"
        )  # add extension to make it readable by pydicom

        dcm_file = pydicom.read_file(dcm_path)
        dcm_array = (cv2.resize(dcm_file.pixel_array, (224, 224)) / 255).astype(
            np.float16
        )  # normalization of the pixel values

        train_or_val = "train" if i < 400 else "val"

        if train_or_val == "train":
            train_ids.append(patient_id)
        else:
            val_ids.append(patient_id)

        final_save_path = save_path / train_or_val
        final_save_path.mkdir(parents=True, exist_ok=True)

        np.save(final_save_path / patient_id, dcm_array)

        if train_or_val == "train":
            np.save(f"{save_path}/{train_or_val}_subjects", train_ids)
        else:
            np.save(f"{save_path}/{train_or_val}_subjects", val_ids)


def calculate_mean_std(train_path: Path) -> tuple[float, float]:
    """
    Computes the mean and standard deviation of all images in the training set.

    Parameters
    ----------
    train_folder : str or Path
        Path to the folder containing the training .npy files.

    Returns
    -------
    mean_val : float
        Mean pixel value across all training images.
    std_val : float
        Standard deviation of pixel values across all training images.
    """

    npy_files = list(train_path.glob("*.npy"))
    sums, sums_squared = 0.0, 0.0
    num_images = len(npy_files)
    num_pixels = 224 * 224  # adjust if size changes

    for f in npy_files:
        arr = np.load(f)
        sums += np.sum(arr) / num_pixels
        sums_squared += np.sum(arr**2) / num_pixels

    mean_val = sums / num_images
    std_val = np.sqrt((sums_squared / num_images) - mean_val**2)

    return mean_val, std_val
