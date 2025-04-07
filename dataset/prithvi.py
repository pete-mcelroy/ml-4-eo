"""
DataModule for the Olds College Label data.
"""

from rasterio.warp import transform_bounds
import lightning as L
import pandas as pd
import numpy as np
from rasterio.warp import transform_bounds
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2
import yaml
from pathlib import Path

from utils.zarr import load_from_zarr
from utils.embeddings import normalize_latlon, normalize_timestamp
from utils.constants import WGS84


class PrithviDataset(Dataset):
    """
    Dataset for the Prithvi foundation model.

    Args:
        chip_dir (str): Directory containing the image chips.
    """

    def __init__(self, chip_dir: str, metadata: dict):
        self.chip_dir = Path(chip_dir)
        self.metadata = metadata
        self.label_dtype = torch.long

        # Load chip and label file names
        self.chips = [chip_path for chip_path in self.chip_dir.glob("*.zarr")]
        print(f"Found {len(self.chips)} chips to process for {chip_dir}")

        sample_chip = load_from_zarr(self.chips[0])
        
        band_names = list(sample_chip.band.values)
        band_names.remove("label")
                
        # Load statistics from metadata
        s2_mean = []
        meta_band_order = []
        for k, v in metadata["sentinel-2-l2a"].bands.mean.items():
            if k in band_names:
                meta_band_order.append(k)
                s2_mean.append(v)
        assert meta_band_order == band_names, f"Band order for means in metadata {meta_band_order} does not match that of input chip {band_names}."
        
        s2_std = []
        meta_band_order = []
        for k, v in metadata["sentinel-2-l2a"].bands.std.items():
            if k in band_names:
                meta_band_order.append(k)
                s2_std.append(v)
        assert meta_band_order == band_names, f"Band order for stds in metadata {meta_band_order} does not match that of input chip {band_names}."

        self.transform = self.create_transforms(
            mean=s2_mean,
            std=s2_std,
        )
        
    def __getitem__(self, idx):
        """
        Get a sample from the dataset.

        Args:
            idx (int): Index of the sample.

        Returns:
            dict: A dictionary containing the image, label, and additional information.
        """
        chip = load_from_zarr(self.chips[idx])

        x = chip.drop_sel(band="label").astype("float32")
        y = chip.sel(band="label").astype("float32")[0] # Label is the same for each time step
        
        chip_bounds = chip.rio.bounds()
        bounds_wgs84 = transform_bounds(src_crs=chip.rio.crs, dst_crs=WGS84, left=chip_bounds[0], bottom=chip_bounds[1], right=chip_bounds[2], top=chip_bounds[3])

        times = [normalize_timestamp(pd.to_datetime(dat)) for dat in chip.time.values]
        week_norm = [dat[0] for dat in times]
        hour_norm = [dat[1] for dat in times]

        lat, lon = bounds_wgs84[-1], bounds_wgs84[0]

        latlons = [normalize_latlon(lat, lon)] * len(times)
        lat_norm = [dat[0] for dat in latlons]
        lon_norm = [dat[1] for dat in latlons]
        
        normalized_image = self.transform(torch.from_numpy(x.to_numpy()))

        sample = {
            "x": normalized_image.transpose(0,1), # (c, t, x, y)
            "mask": torch.from_numpy(y.to_numpy()).type(self.label_dtype),
            "temporal_coords": torch.tensor(np.hstack((week_norm, hour_norm)), dtype=torch.float32),
            "location_coords": torch.tensor(np.hstack((lat_norm, lon_norm)), dtype=torch.float32)
        }
        return sample
    

    def create_transforms(self, mean, std):
        """
        Create normalization transforms.

        Args:
            mean (list): Mean values for normalization.
            std (list): Standard deviation values for normalization.

        Returns:
            torchvision.transforms.Compose: A composition of transforms.
        """
        return v2.Compose(
            [
                v2.Normalize(mean=mean, std=std),
            ],
        )

    def __len__(self):
        return len(self.chips)
    

class PrithviDataModule(L.LightningDataModule):
    """
    DataModule class for the Prithvi dataset.

    Args:
        train_chip_dir (str): Directory containing training image chips.
        val_chip_dir (str): Directory containing validation image chips.
        metadata_path (str): Path to the metadata file.
        batch_size (int): Batch size for data loading.
        num_workers (int): Number of workers for data loading.
    """

    def __init__(  # noqa: PLR0913
        self,
        train_chip_dir,
        val_chip_dir,
        test_chip_dir,
        metadata_path,
        batch_size,
        num_workers,
    ):
        super().__init__()
        self.train_chip_dir = train_chip_dir
        self.val_chip_dir = val_chip_dir
        self.test_chip_dir = test_chip_dir
        self.metadata = Box(yaml.safe_load(open(metadata_path)))
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self):
        """
        Setup datasets for training, validation and testing.
        """
        self.train_dataset = PrithviDataset(
            self.train_chip_dir,
            self.metadata,
        )
        self.val_dataset = PrithviDataset(
            self.val_chip_dir,
            self.metadata,
        )
        self.test_dataset = PrithviDataset(
            self.test_chip_dir,
            self.metadata,
        )

    def train_dataloader(self):
        """
        Create DataLoader for training data.

        Returns:
            DataLoader: DataLoader for training dataset.
        """
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        """
        Create DataLoader for validation data.

        Returns:
            DataLoader: DataLoader for validation dataset.
        """
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )
    
    def test_dataloader(self):
        """
        Create DataLoader for test data.

        Returns:
            DataLoader: DataLoader for test dataset.
        """
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )
    