from __future__ import annotations

import h5py
import numpy as np
import torch
from atria.core.utilities.logging import get_logger

logger = get_logger(__name__)


class HFDataset:
    def __init__(self, filepath: str, mode="a"):
        self.filepath = filepath
        self.mode = mode
        self.hf = None

    def __enter__(self) -> HFSampleSaver:
        self.hf = h5py.File(self.filepath, self.mode)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.hf:
            self.hf.close()
            self.hf = None

    def save(self, key: str, data: torch.Tensor) -> None:
        """
        Saves the dataset to the HDF5 file under the given key.

        Args:
            key (str): The key to identify the data.
            data: The data to be saved.

        Raises:
            ValueError: If the data cannot be saved.
        """
        if key in self.hf:
            raise ValueError(f"Key '{key}' already exists in the file.")

        # Create a dataset with max shape to allow future resizing
        self.hf.create_dataset(
            key,
            data=data.detach().cpu().numpy(),
            compression="gzip",
        )

    def load(self):
        """
        Loads the dataset from the HDF5 file.

        Returns:
            dict: The data loaded from the file.
        """
        data = {}
        for key in self.hf:
            data[key] = torch.from_numpy(self.hf[key][:])

        return data

    def key_exists(self, key: str) -> np.ndarray:
        return key in self.hf


class HFSampleSaver:
    def __init__(self, filepath: str, mode="a", overwrite_if_exists: bool = True):
        self.filepath = filepath
        self.mode = mode
        self.hf = None
        self.overwrite_if_exists = overwrite_if_exists

    def __enter__(self) -> HFSampleSaver:
        self.hf = h5py.File(self.filepath, self.mode)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.hf:
            self.hf.close()
            self.hf = None

    def save_attribute(self, key: str, data, sample_key: str) -> None:
        if sample_key not in self.hf:
            self.hf.create_group(sample_key)
        if self.hf[sample_key].attrs.get(key) is None:
            self.hf[sample_key].attrs[key] = data

    def load_attribute(self, key: str, sample_key: str):
        if sample_key not in self.hf:
            return None

        return self.hf[sample_key].attrs.get(key)

    def sample_exists(self, sample_key: str):
        return sample_key in self.hf

    def key_exists(self, sample_key: str, key: str):
        return sample_key in self.hf and key in self.hf[sample_key]

    def get_keys(self, sample_key: str):
        if sample_key in self.hf:
            return list(self.hf[sample_key].keys())
        return []

    def save(self, key: str, data, sample_key: str) -> None:
        if sample_key not in self.hf:
            self.hf.create_group(sample_key)

        if key not in self.hf[sample_key]:
            # Create a dataset for the key
            if isinstance(data, np.ndarray):
                self.hf[sample_key].create_dataset(
                    key,
                    data=[data],
                    maxshape=(None, *data.shape),
                    compression="gzip",
                )
            else:
                try:
                    self.hf[sample_key].create_dataset(key, data=data)
                except Exception as e:
                    logger.exception(
                        f"Exception raised while creating dataset for data={key}: {data}"
                    )
                    exit()
        else:
            if self.overwrite_if_exists:
                logger.warning(
                    f"Data already exists on given key = {sample_key}/{key}. "
                    "Overwriting as overwrite_if_exists=True..."
                )
                # Create a dataset for the key
                if isinstance(data, np.ndarray):
                    # Overwrite the existing data
                    self.hf[sample_key][key][0] = data
                else:
                    self.hf[sample_key][key][...] = data
            else:
                logger.warning(
                    f"Data already exists on given key = {key}. "
                    "If you want to overwrite it set overwrite_if_exists=True."
                )

    def load(self, key: str, sample_key: str):
        if sample_key not in self.hf or key not in self.hf[sample_key]:
            return None

        return self.hf[sample_key][key][...]
