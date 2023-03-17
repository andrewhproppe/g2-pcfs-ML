from pathlib import Path
from tkinter import Image
from typing import Callable, List, Union, Optional, Dict
from functools import lru_cache

import h5py
import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import Dataset, random_split, DataLoader

from g2_pcfs.pipeline import transforms as t
from g2_pcfs.utils import paths


class ImageDataset(Dataset):
    def __init__(
        self,
        filepath: Union[str, Path],
        seed: int = 10236,
        transforms: Optional[List[Callable]] = None,
        fixed_scale: Optional[int] = None,
        scale_range: Optional[List[int]] = None,
        **kwargs
    ):
        super().__init__()
        self._filepath = filepath
        self.rng = np.random.default_rng(seed)
        self.df = self.data["df"][:]
        # do the exact same stuff for the noise-free g2s
        # except add noise
        if transforms:
            self.transforms = transforms
        else:
            # otherwise use the default stuff
            self.transforms = [
                t.NoisyInputs(self.rng, self.df),
                t.ArrayToTensor(),
                t.NormalizeBatch(),
                t.AddChannelDim(),
            ]
        # get the actual time grid
        self.timesteps = self.data["t"][:]

        self.fixed_scale = fixed_scale
        self.scale_range = scale_range

    @property
    def filepath(self) -> str:
        return self._filepath

    @property
    def num_params(self) -> int:
        return len([key for key in self.data.keys() if "parameter" in key])

    @lru_cache()
    def __len__(self) -> int:
        """
        Returns the total number of g2s in the dataset.
        Because the array is n-dimensional, the length is
        given as the product of the first four dimensions.

        Returns
        -------
        int
            Number of g2s in the dataset
        """
        g2_shape = self.data["g2s"].shape
        return g2_shape[0]

    @property
    @lru_cache()
    def indices(self) -> np.ndarray:
        return np.arange(len(self))

    @property
    def data(self):
        return h5py.File(self.filepath, "r")

    @property
    @lru_cache()
    def g2s(self) -> np.ndarray:
        """
        Return the g2s stored on disk, however reshaped such that
        we flatten the grid of parameters, and left with a 2D array
        with shape (num_g2s, timesteps).

        Returns
        -------
        np.ndarray
            NumPy 1D array containing photon counts
        """
        return self.data["g2s"]

    @property
    @lru_cache()
    def parameters(self):
        # params = [self.data[f"parameter_w{i}"][:] for i in range(self.num_params)]
        # grid = np.stack(np.meshgrid(*params, indexing="ij")).reshape(self.num_params, -1).T
        # return grid
        return self.data["params"][:]

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        target_image = self.g2s[index]
        # Use fixed scale, or scale range set before training
        if self.fixed_scale != None:
            scale = self.fixed_scale
        else:
            if self.scale_range != None:
                scale = 10 ** self.rng.uniform(self.scale_range[0], self.scale_range[1])
            else:
                scale = 10 ** self.rng.uniform(0, 3)
        # grab the data
        target_image *= scale
        input_image = target_image.copy()
        params = self.parameters[index]
        # get the time stuff
        time = self.timesteps
        ode_time = torch.linspace(0.0, 1.0, len(time))
        data = {
            "target": target_image,
            "input": input_image,
            "parameters": params,
            "scale": scale,
            "time": time,
            "ode_time": ode_time,
        }
        # run through transforms
        if self.transforms:
            for transform in self.transforms:
                data = transform(data)
        return data


class ODEDataModule(pl.LightningDataModule):
    def __init__(
        self,
        h5_path: Union[None, str] = None,
        batch_size: int = 64,
        seed: int = 120516,
        **kwargs
    ):
        super().__init__()
        # by default run with the devset
        if not h5_path:
            h5_path = paths.get("raw").joinpath("pcfs_g2_devset.h5")
        self.h5_path = paths.get("raw").joinpath(h5_path)
        self.batch_size = batch_size
        self.seed = seed
        self.data_kwargs = kwargs

    def setup(self, stage: Union[str, None] = None):
        full_dataset = ImageDataset(self.h5_path, self.seed, **self.data_kwargs)
        # use 10% of the data set a test set
        test_size = int(len(full_dataset) * 0.2)
        self.train_set, self.val_set = random_split(
            full_dataset,
            [len(full_dataset) - test_size, test_size],
            torch.Generator().manual_seed(self.seed),
        )

    @staticmethod
    def collate_ode_data(
        batch: List[Dict[str, torch.Tensor]]
    ) -> Dict[str, torch.Tensor]:
        keys = batch[0].keys()
        batched_data = {}
        for key in keys:
            if "time" not in key:
                data = [b.get(key) for b in batch]
                if isinstance(data[0], torch.Tensor):
                    batched_data[key] = torch.vstack(data)
                else:
                    batched_data[key] = torch.as_tensor(data)
        # now generate the random time steps to train with
        timesteps = batch[0].get("time")
        num_steps = np.random.randint(5, 20)
        indices = torch.randperm(len(timesteps))[:num_steps].sort()[0]
        batched_data["indices"] = indices
        batched_data["time"] = timesteps
        batched_data["ode_time"] = batch[0].get("ode_time")
        return batched_data

    def train_dataloader(self):
        return DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,
            collate_fn=self.collate_ode_data,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_set,
            batch_size=self.batch_size,
            num_workers=0,
            collate_fn=self.collate_ode_data,
        )
