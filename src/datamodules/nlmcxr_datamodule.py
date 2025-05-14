import torch
import hydra
import numpy as np

from pytorch_lightning import LightningDataModule
from torchvision import transforms
from transformers import AutoTokenizer
from torch.utils.data import Dataset, DataLoader, random_split
from typing import Optional, Tuple, Dict, Any

from .nlmcxr_dataset import NLMCXRDataSet

class NLMCXRDataModule(LightningDataModule):
    """Example of LightningDataModule for MNIST dataset.

    A DataModule implements 5 key methods:

        def prepare_data(self):
            # things to do on 1 GPU/TPU (not on every GPU/TPU in DDP)
            # download data, pre-process, split, save to disk, etc...
        def setup(self, stage):
            # things to do on every process in DDP
            # load data, set variables, etc...
        def train_dataloader(self):
            # return train dataloader
        def val_dataloader(self):
            # return validation dataloader
        def test_dataloader(self):
            # return test dataloader
        def teardown(self):
            # called on every process in DDP
            # clean up after fit or test

    This allows you to share a full dataset without explaining how to download,
    split, transform and process the data.

    Read the docs:
        https://pytorch-lightning.readthedocs.io/en/latest/data/datamodule.html
    """

    def __init__(
        self,
        nlmcxr_dataset_file: str,
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
        text_model_name: str = '',
        train_val_split: Tuple[float, float] = [0.95,0.05],
        normalize_mean_values: Tuple[float, float, float] = [0.485, 0.456, 0.406],
        normalize_std_values: Tuple[float, float, float] = [0.229, 0.224, 0.225],
        seed: int = 42,
        subset_percentage: Optional[float] = None,
    ):
        super(NLMCXRDataModule,).__init__()
        
        self.save_hyperparameters(logger=False)

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((224, 224)),
            # transforms.RandomResizedCrop(size=224, ratio=(0.6, 1.0)),
            # transforms.RandomAffine(degrees=(-20, 20), translate=(0.1, 0.1), scale=(0.95, 1.05)),
            # transforms.ColorJitter(contrast=(0.6, 1.4), brightness=(0.6, 1.4)),
            # transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 3.0)),
            transforms.Normalize(normalize_mean_values, normalize_std_values)
        ])

        self.tokenizer = AutoTokenizer.from_pretrained(text_model_name, use_fast=True)
  
        self.train_dataset: Optional[Dataset] = None
        self.val_dataset: Optional[Dataset] = None
        self.test_dataset: Optional[Dataset] = None

        self.prepare_data_per_node = False

        self.nlmcxr_dataset_file = nlmcxr_dataset_file

        self.seed = seed
        self.subset_percentage = subset_percentage

    def prepare_data(self):
        """Download data if needed.
        Do not use it to assign state (self.x = y).
        """
        return
    
    def setup(self, stage: str):
        if stage == "fit":
            self.train_dataset = NLMCXRDataSet(
                self.nlmcxr_dataset_file,
                self.transform,
                self.tokenizer
            )

            total_samples = len(self.train_dataset)
            train_number_samples = int(self.hparams.train_val_split[0] * len(self.train_dataset))
            train_val_split_count = [train_number_samples, total_samples-train_number_samples]

            self.train_dataset, self.val_dataset = random_split(
                    dataset=self.train_dataset,
                    lengths=train_val_split_count,
                    generator=torch.Generator().manual_seed(self.seed),
            )

            ## Subset after split option
            if self.subset_percentage is not None:
                indices = np.random.choice(
                    len(self.train_dataset),
                    int(self.subset_percentage * len(self.train_dataset)),
                    replace=False,
                )

                if len(indices) != 0:
                    self.train_dataset = torch.utils.data.Subset(self.train_dataset, indices)

            print(f"Training Dataset Size: {len(self.train_dataset)}")
            print(f"Validation Dataset Size: {len(self.val_dataset)}")
        
        elif stage == "test":
            self.test_dataset = NLMCXRDataSet(
                self.nlmcxr_dataset_file,
                self.transform,
                self.tokenizer
            )
            print(f"Test Dataset Size: {len(self.test_dataset)}")

    def train_dataloader(self, num_workers=None):
        return DataLoader(
            dataset = self.train_dataset,
            batch_size = self.hparams.batch_size,
            num_workers = self.hparams.num_workers if num_workers is None else num_workers,
            pin_memory = self.hparams.pin_memory,
            shuffle = True,
        )

    def val_dataloader(self, num_workers=None):
        return DataLoader(
            dataset = self.val_dataset,
            batch_size = self.hparams.batch_size,
            num_workers = self.hparams.num_workers if num_workers is None else num_workers,
            pin_memory = self.hparams.pin_memory,
            shuffle = False,
        )

    def test_dataloader(self, num_workers=None):
        return DataLoader(
            dataset = self.test_dataset,
            batch_size = self.hparams.batch_size,
            num_workers = self.hparams.num_workers if num_workers is None else num_workers,
            pin_memory = self.hparams.pin_memory,
            shuffle = False,
        )

    def teardown(self, stage: Optional[str] = None):
        """Clean up after fit or test."""
        pass

    def state_dict(self):
        """Extra things to save to checkpoint."""
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Things to do when loading checkpoint."""
        pass


if __name__ == "__main__":
    import hydra
    import omegaconf
    import pyrootutils

    root = pyrootutils.setup_root(__file__, pythonpath=True)
    cfg = omegaconf.OmegaConf.load(root / "configs" / "datamodule" / "nlmcxr.yaml")
    print(cfg.pretty())
    _ = hydra.utils.instantiate(cfg)