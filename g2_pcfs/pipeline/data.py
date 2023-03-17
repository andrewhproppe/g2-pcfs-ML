from functools import lru_cache
from typing import Tuple, Type, Union

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, random_split
# from torchvision.transforms import Compose, ToTensor
import pytorch_lightning as pl

from g2_pcfs.pipeline import transforms
from g2_pcfs.utils import paths


class H5Dataset(Dataset):
    def __init__(self, filepath: str, seed: int = 10236, window_size: int = 5, **kwargs):
        super().__init__()
        self._filepath = filepath
        self.rng = np.random.default_rng(seed)
        self.df = self.data["df"][:]
        # do the exact same stuff for the noise-free g2s
        # except add noise
        self.transform = transforms.train_transform_pipeline(self.rng, self.df)
        self.target_transform = transforms.target_transform_pipeline()
        self.timesteps = self.data["t"][:]

        if 'fixed_scale' in kwargs:
            self.fixed_scale = kwargs['fixed_scale']
        else:
            self.fixed_scale = None

        if 'scale_range' in kwargs:
            self.scale_range = kwargs['scale_range']
        else:
            self.scale_range = None


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
        return np.prod(g2_shape[:-1])

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
        return np.reshape(self.data["g2s"], (len(self), -1))

    @property
    @lru_cache()
    def parameters(self):
        # params = [self.data[f"parameter_w{i}"][:] for i in range(self.num_params)]
        # grid = np.stack(np.meshgrid(*params, indexing="ij")).reshape(self.num_params, -1).T
        # return grid
        return self.data["params"][:] # 20220916 removed transpose after modifying data generation

    def __getitem__(self, index: int) -> Tuple[Type[torch.Tensor]]:
        """
        Returns a randomly chosen spectrum, along with its noisy
        counterpart.

        Parameters
        ----------
        index : int
            Not used; passed by a `DataLoader`

        Returns
        -------
        x : torch.Tensor
            Noisy spectrum
        y : torch.Tensor
            Noise-free spectrum
        """
        y = self.g2s[index]
        # y = self.g2s[0].copy()

        # Use fixed scale, or scale range set before training
        if self.fixed_scale != None:
            scale = self.fixed_scale
        else:
            if self.scale_range != None:
                scale = 10**self.rng.uniform(self.scale_range[0], self.scale_range[1])
            else:
                scale = 10**self.rng.uniform(0, 3)

        # rescale the noise free g2s

        y *= scale
        x = y.copy()

        # run through the usual pipeline; add noise and
        x = self.transform(x)
        y = self.target_transform(y)
        # scale back down to original size
        y /= scale
        x /= scale

        """ Performing normalization here instead of pipeline.transform to co-normalize the g2s together """
        # subtract baseline w.r.t. noisy data
        y = y-torch.min(x)
        x = x-torch.min(x)
        # normalize w.r.t. noisy data
        y = y/torch.max(x)
        x = x/torch.max(x)

        # pack data into arrays
        x = np.vstack([self.timesteps, x]).T
        y = np.vstack([self.timesteps, y]).T
        # get the parameters now
        params = self.parameters[index]

        x = x.astype(np.float32)
        y = y.astype(np.float32)
        params = params.astype(np.float32)

        return (x, y, params)


class H5Dataset2D(Dataset):
    def __init__(self, filepath: str, seed: int = 10236, window_size: int = 5, **kwargs):
        super().__init__()
        self._filepath = filepath
        self.rng = np.random.default_rng(seed)
        self.df = self.data["df"][:]
        # do the exact same stuff for the noise-free g2s
        # except add noise
        self.transform = transforms.train_transform_pipeline(self.rng, self.df)
        self.target_transform = transforms.target_transform_pipeline()
        self.timesteps = self.data["t"][:]

        if 'fixed_scale' in kwargs:
            self.fixed_scale = kwargs['fixed_scale']
        else:
            self.fixed_scale = None

        if 'scale_range' in kwargs:
            self.scale_range = kwargs['scale_range']
        else:
            self.scale_range = None

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
        # return np.prod(g2_shape[:-1])
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
        # return np.reshape(self.data["g2s"], (len(self), -1))

    @property
    @lru_cache()
    def parameters(self):
        # params = [self.data[f"parameter_w{i}"][:] for i in range(self.num_params)]
        # grid = np.stack(np.meshgrid(*params, indexing="ij")).reshape(self.num_params, -1).T
        # return grid
        return self.data["params"][:]

    def __getitem__(self, index: int) -> Tuple[Type[torch.Tensor]]:
        # FFFFF
        y = self.g2s[index]
        # y = self.g2s[0].copy()

        # Use fixed scale, or scale range set before training
        if self.fixed_scale != None:
            scale = self.fixed_scale
        else:
            if self.scale_range != None:
                scale = 10**self.rng.uniform(self.scale_range[0], self.scale_range[1])
            else:
                scale = 10**self.rng.uniform(0, 3)

        # rescale the noise free g2s

        y *= scale
        x = y.copy()

        # run through the usual pipeline; add noise and
        x = self.transform(x)
        y = self.target_transform(y)
        # scale back down to original size
        y /= scale
        x /= scale

        """ Performing normalization here instead of pipeline.transform to co-normalize the g2s together """
        # subtract baseline w.r.t. noisy data
        y = y-torch.min(x)
        x = x-torch.min(x)
        # normalize w.r.t. noisy data
        y = y/torch.max(x)
        x = x/torch.max(x)

        # pack data into arrays
        # unsqueeze to create 1 channel for Conv2D
        x = x.unsqueeze(0)
        y = y.unsqueeze(0)
        # x = np.vstack([self.timesteps, x]).T
        # y = np.vstack([self.timesteps, y]).T
        # get the parameters now
        params = self.parameters[index]

        # x = x.astype(np.float32)
        # y = y.astype(np.float32)

        x = x.type(torch.float32)
        y = y.type(torch.float32)

        params = params.astype(np.float32)

        return (x, y, params)


class NoiseFreeDataset(H5Dataset):
    def __init__(self, filepath: str, seed: int = 10236, window_size: int = 5):
        super().__init__(filepath, seed=seed, window_size=window_size)
    
    def __getitem__(self, index: int) -> Tuple[Type[torch.Tensor]]:
        y = self.g2s[index]
        y = self.target_transform(y)
        x = y.clone()
        x = np.vstack([self.timesteps, x]).T
        y = np.vstack([self.timesteps, y]).T
        params = self.parameters[index]
        return x, y, params


class g2DataModule(pl.LightningDataModule):
    def __init__(
        self, h5_path: Union[None, str] = None, batch_size: int = 64, seed: int = 120516, as_2d=False, **kwargs
    ):
        super().__init__()
        # by default run with the devset
        if not h5_path:
            h5_path = "pcfs_g2_devset.h5"
        self.h5_path = paths.get("raw").joinpath(h5_path)
        self.batch_size = batch_size
        self.seed = seed
        self.as_2d = as_2d
        self.data_kwargs = kwargs

    def setup(self, stage: Union[str, None] = None):
        if self.as_2d:
            full_dataset = H5Dataset2D(self.h5_path, **self.data_kwargs)
        else:
            full_dataset = H5Dataset(self.h5_path, **self.data_kwargs)
        # use 10% of the data set a test set
        test_size = int(len(full_dataset) * 0.2)
        self.train_set, self.val_set = random_split(
            full_dataset,
            [len(full_dataset) - test_size, test_size],
            torch.Generator().manual_seed(self.seed),
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=1
            # collate_fn=transforms.pad_collate_func,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_set,
            batch_size=self.batch_size,
            num_workers=1
            # collate_fn=transforms.pad_collate_func,
        )


class NoiseFreeg2DataModule(g2DataModule):
    def __init__(self, h5_path: Union[None, str] = None, batch_size: int = 64, seed: int = 120516, **kwargs):
        super().__init__(h5_path=h5_path, batch_size=batch_size, seed=seed, **kwargs)

    def setup(self, stage: Union[str, None] = None):
        full_dataset = NoiseFreeDataset(self.h5_path, **self.data_kwargs)
        # use 10% of the data set a test set
        test_size = int(len(full_dataset) * 0.2)
        self.train_set, self.val_set = random_split(
            full_dataset,
            [len(full_dataset) - test_size, test_size],
            torch.Generator().manual_seed(self.seed),
        )


def get_test_batch(batch_size: int = 32, h5_path: Union[None, str] = None, seed: int = 120516, noise: bool = True) -> Tuple[torch.Tensor]:
    """
    Convenience function to grab a batch of validation data using the same
    pipeline as in the training process.

    Parameters
    ----------
    batch_size : int, optional
        Number of g2s to grab, by default 32
    h5_path : Union[None, str], optional
        Name of the HDF5 file containing g2s, by default None
    seed : int, optional
        random seed, by default 120516, same as training
    noise : bool, optional
        Whether to use the noisy data set, by default True

    Returns
    -------
    Tuple[torch.Tensor]
        3-tuple of data
    """
    if noise:
        target_module = g2DataModule
    else:
        target_module = NoiseFreeg2DataModule
    data = target_module(h5_path, batch_size, seed)
    data.setup()
    return next(iter(data.val_dataloader()))

# Debugging
if __name__ == '__main__':
    from matplotlib import pyplot as plt
    from torch import nn

    # data = g2DataModule('pcfs_g2_devset2.h5', batch_size=128, window_size=1)
    data = g2DataModule('pcfs_g2_2d_test1.h5', batch_size=128, window_size=1, as_2d=True)
    data.setup()
    (x, y, params) = data.train_set.__getitem__(1)

    input = torch.randn(20, 1, 100, 140)
    from g2_pcfs.models.base import Conv2DAutoEncoder
    ae = Conv2DAutoEncoder(kernel1=15, kernel2=3)
    output = ae.forward(input)[0]

    #
    # conv_shape = get_conv_output_shape(ae.encoder, input)
    # flat_shape = get_conv_flat_shape(ae.encoder, input)
    #
    # z_dim = 32
    # flattener = nn.Flatten()
    # linear_bottleneck_1 = nn.Linear(flat_shape[0], z_dim)
    # linear_bottleneck_2 = nn.Linear(z_dim, flat_shape[0])
    # reshaper = Reshape(-1, conv_shape[1], conv_shape[2], conv_shape[3])
    #
    #
    # encoded     = ae.encoder(input)
    # flattened   = flattener(encoded)
    # z           = linear_bottleneck_1(flattened)
    # unflattened = linear_bottleneck_2(z)
    # reshaped    = reshaper(unflattened)
    # decoded     = ae.decoder(reshaped)




    # output = encoder.model(input)
    # conv_shape = get_conv_output_shape(encoder.model, input)
    # flat = nn.Flatten()
    # output_flat = flat(output)
    # flat_size = output_flat.shape[-1]
    # flat_to_z = nn.Linear(flat_size, 16)
    # z = flat_to_z(output_flat)
    # z_to_flat = nn.Linear(16, flat_size)
    # flat_new = z_to_flat(z)
    # flat_to_conv = Reshape(-1, conv_shape[1], conv_shape[2], conv_shape[3])
    # conv_new = flat_to_conv(flat_new)

    """FFFF"""
    #
    #
    # class ConvEncoder2D(nn.Module):
    #     def __init__(
    #         self,
    #         input_channels: int,
    #         output_channels: int,
    #         num_layers: int,
    #         kernel1: int = (5, 5),
    #         kernel2: int = (3, 3),
    #         pool_size: int = 2,
    #     ):
    #         super().__init__()
    #         modules = []
    #         channels = np.interp(
    #             np.linspace(0.0, 1.0, num_layers+1), [0, 1], [input_channels, output_channels]
    #         ).astype(int)
    #         for idx in range(num_layers):
    #             # kernel_size = kernel1
    #             if idx == num_layers-1:
    #                 kernel_size = kernel1
    #                 pad = 0
    #             else:
    #                 kernel_size = kernel2
    #                 pad = 1
    #             modules.extend(
    #                 [
    #                     nn.Conv2d(channels[idx], channels[idx + 1], kernel_size, stride=(2, 2), padding=pad),
    #                     nn.SiLU(),
    #                     nn.BatchNorm2d(channels[idx + 1]),
    #                     # nn.MaxPool2d(pool_size),
    #                 ]
    #             )
    #
    #         # modules.append(nn.Flatten())
    #         self.model = nn.Sequential(*modules)
    #
    #
    # class ConvDecoder2D(nn.Module):
    #     def __init__(
    #         self,
    #         z_channels: int,
    #         input_channels: int,
    #         output_channels: int,
    #         flat_size: int,
    #         conv_shape,
    #         num_layers: int,
    #         kernel1: int = (5, 5),
    #         kernel2: int = (3, 3),
    #     ):
    #         super().__init__()
    #         modules = []
    #         modules.append(nn.Linear(z_channels, flat_size))
    #         modules.append(Reshape(-1, conv_shape[1], conv_shape[2], conv_shape[3]))
    #         channels = np.interp(
    #             np.linspace(0.0, 1.0, num_layers+1), [0, 1], [input_channels, output_channels]
    #         ).astype(int)
    #
    #         for idx in (range(num_layers)):
    #             # kernel_size = kernel1
    #             if idx == 0:
    #                 kernel_size = kernel1
    #                 pad = 0
    #             else:
    #                 kernel_size = kernel2
    #                 pad = 1
    #             modules.extend(
    #                 [
    #                     nn.ConvTranspose2d(channels[idx], channels[idx + 1], kernel_size, stride=(2, 2), padding=pad, output_padding=pad),
    #                     nn.SiLU(),
    #                 ]
    #             )
    #
    #         self.model = nn.Sequential(*modules)
    #
    #
    # nlayers = 3
    #
    # encoder = ConvEncoder2D(input_channels=1, output_channels=32, num_layers=nlayers)
    #
    # output = encoder.model(input)
    # conv_shape = get_conv_output_shape(encoder.model, input)
    # flat = nn.Flatten()
    # output_flat = flat(output)
    # flat_size = output_flat.shape[-1]
    # flat_to_z = nn.Linear(flat_size, 16)
    # z = flat_to_z(output_flat)
    # # z_to_flat = nn.Linear(16, flat_size)
    # # flat_new = z_to_flat(z)
    # # flat_to_conv = Reshape(-1, conv_shape[1], conv_shape[2], conv_shape[3])
    # # conv_new = flat_to_conv(flat_new)
    #
    # decoder = ConvDecoder2D(z_channels=16, input_channels=32, output_channels=1, flat_size=flat_size, num_layers=nlayers, conv_shape=conv_shape)
    # decoded = decoder.model(z)
