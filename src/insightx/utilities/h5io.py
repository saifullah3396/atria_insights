from __future__ import annotations

import h5py
import numpy as np


class HFDataset:
    def __init__(self, filepath: str, mode="a"):
        self.filepath = filepath
        self.mode = mode
        self.hf = None

    def __enter__(self) -> HFIO:
        self.hf = h5py.File(self.filepath, self.mode)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.hf:
            self.hf.close()
            self.hf = None

    def save(self, key: str, data: np.ndarray) -> None:
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
        self.hf.create_dataset(key, data=data.cpu().numpy())

    def load(self, key: str) -> np.ndarray:
        """
        Loads the dataset from the HDF5 file under the given key.

        Args:
            key (str): The key to identify the data.
            data: The data to be saved.
        """
        if key not in self.hf:
            raise KeyError(f"Key '{key}' does not exist under key '{key}'.")

        return self.hf[key][:]

    def key_exists(self, key: str) -> np.ndarray:
        return key in self.hf


class HFIO:
    def __init__(self, filepath: str, mode="a"):
        self.filepath = filepath
        self.mode = mode
        self.hf = None

    def __enter__(self) -> HFIO:
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

    def sample_exists(self, sample_key: str):
        return sample_key in self.hf

    def key_exists(self, sample_key: str, key: str):
        return sample_key in self.hf and key in self.hf[sample_key]

    def save(self, key: str, data, sample_key: str) -> None:
        if sample_key not in self.hf:
            self.hf.create_group(sample_key)

        if key not in self.hf[sample_key]:
            # Create a dataset for the key
            if isinstance(data, np.ndarray):
                self.hf[sample_key].create_dataset(
                    key, data=[data], maxshape=(None, *data.shape)
                )
            else:
                self.hf[sample_key].create_dataset(key, data=data)
        else:
            # Create a dataset for the key
            if isinstance(data, np.ndarray):
                # Overwrite the existing data
                self.hf[sample_key][key][0] = data
            else:
                self.hf[sample_key][key][...] = data

    def load(self, key: str, sample_key: str):
        if sample_key not in self.hf or key not in self.hf[sample_key]:
            return None

        return self.hf[sample_key][key][...]
