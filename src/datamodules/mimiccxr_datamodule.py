from typing import Any, Dict, Optional, Tuple

import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from transformers import AutoTokenizer
from sklearn.model_selection import GroupShuffleSplit

import numpy as np
import hydra

from .mimiccxr_dataset import MIMICCXRDataset, TransformSubset
from .components.masking_transforms import ImageMaskingTransform, TextMaskingTransform

REQUIRED_TRANSFORMS = ['torchvision.transforms.Resize',
                       'torchvision.transforms.ToTensor',
                       'torchvision.transforms.Normalize']

class MIMICCXRDataModule(LightningDataModule):
    """LightningDataModule for MIMIC-CXR dataset.

    Handles loading and preprocessing of chest X-ray images and their associated reports.
    Implements patient-aware train/validation splitting to ensure no patient overlap.
    Provides options for image transformations, text tokenization, and masking transforms
    for both modalities.

    Key Methods:
        prepare_data(): Not used as data is assumed to be preprocessed
        setup(): Loads data and splits into train/val sets ensuring patient separation
        train_dataloader(): Returns train dataloader with augmentations
        val_dataloader(): Returns validation dataloader with base transforms

    Args:
        mimic_cxr_dataset_file: Path to the MIMIC-CXR dataset file
        batch_size: Batch size for the dataloaders
        num_workers: Number of workers for the dataloaders
        pin_memory: Whether to pin memory in the dataloaders
        n_sentences: Number of sentences to use from the report
        text_model_name: Name of the text model to use for tokenization
        train_val_split: Number of samples to use for train and validation sets
        seed: Random seed for the train/val split
        transform_list: List of transforms for image preprocessing
        img_mask_transform: Masking transform for images
        text_mask_transform: Masking transform for text
    """

    def __init__(
        self,
        mimic_cxr_dataset_file:str,
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
        n_sentences: int=1,
        text_model_name:str = '',
        train_val_split: Tuple[int, int] = [-1, 5000],
        seed: int = 42,
        transform_list: Optional[Any] = None,
        img_mask_transform: ImageMaskingTransform = None,
        text_mask_transform: TextMaskingTransform = None,
        #cfg: dict = {},
    ):
        super(MIMICCXRDataModule,).__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        # due to _recursive_=false instantiation of the dataloader, we instantiate the transforms here
        if img_mask_transform is not None and text_mask_transform is not None:
            self.hparams.img_mask_transform = hydra.utils.instantiate(img_mask_transform)
            self.hparams.text_mask_transform = hydra.utils.instantiate(text_mask_transform)
            print("Image and text masking transforms are provided")
        elif img_mask_transform is None and text_mask_transform is None:
            pass
        else:
            raise ValueError("Both img_mask_transform and text_mask_transform must be provided")

        # Check if the first transform is torchvision.transforms.ToTensor since otherwise
        # we see unexpected behavior with the transforms. This is because the dataset returns
        # a PIL image.
        if self.hparams.transform_list[0]._target_ != 'torchvision.transforms.ToTensor':
            raise ValueError("First transform must be torchvision.transforms.ToTensor") 

        ## Image transformations
        for req_transform in REQUIRED_TRANSFORMS:
            if req_transform not in [t._target_ for t in self.hparams.transform_list]:
                raise ValueError(f"{req_transform} transformation is required")
        
        # We need to differentiate between transforms that are required and those that belong to
        # data augmentation. We will use the required transforms for the validation set and all
        # transforms for the training set.
        all_transform_instances = []
        req_transform_instances = []
        for t in self.hparams.transform_list:
            # There is a bug in torchvision when initializing ColorJitter with hydra.
            # See https://github.com/pytorch/vision/issues/5646
            # we have to add it manually 
            if t._target_ == 'torchvision.transforms.ColorJitter':
                # have to convert hydras tuple to list
                color_jitter = transforms.ColorJitter(brightness=list(t.brightness),
                                                        contrast=list(t.contrast))
                all_transform_instances.append(color_jitter)
            else:
                instance = hydra.utils.instantiate(t)
                if t._target_ in REQUIRED_TRANSFORMS:
                    req_transform_instances.append(instance)
                all_transform_instances.append(instance)
        self.req_transforms = transforms.Compose(req_transform_instances)
        self.all_transforms = transforms.Compose(all_transform_instances)

        ## Transforms
        self.tokenizer = AutoTokenizer.from_pretrained(text_model_name, use_fast=True)

        self.train_dataset: Optional[Dataset] = None
        self.val_dataset: Optional[Dataset] = None

        self.prepare_data_per_node = False

    def prepare_data(self):
        """Download data if needed.
        Do not use it to assign state (self.x = y).
        """
        return

    def setup(self, stage: str):
        """
        Load data and split it into train and val sets. Ensure that patients are not split between
        train and validation sets.

        This method is called by lightning with both `trainer.fit()` and `trainer.test()`, so be
        careful not to execute things like random split twice!

        In Zhang et al. 2022, the dataset has a validation set of 5000 samples. We will use the same
        split.
        """
        # if train and val datasets are already split then do nothing
        if self.train_dataset is not None and self.val_dataset is not None:
            return

        # Load the dataset
        dataset = MIMICCXRDataset(self.hparams.mimic_cxr_dataset_file,
                                  transforms=None, # We will apply transforms later
                                  n_sentences=self.hparams.n_sentences,
                                  tokenizer=self.tokenizer)

        # Prepare data for grouped splitting
        patient_ids = dataset.df['patient_id'].values
        indices = np.arange(len(dataset))

        # Use GroupShuffleSplit to split based on patient_id
        gss = GroupShuffleSplit(
            n_splits=1,
            test_size=self.hparams.train_val_split[1] / len(dataset),
            random_state=self.hparams.seed
        )

        # Perform the split
        train_idx, val_idx = next(gss.split(indices, groups=patient_ids))

        # assert that patients are not split between train and val
        train_patients = dataset.df['patient_id'].iloc[train_idx].unique()
        val_patients = dataset.df['patient_id'].iloc[val_idx].unique()
        assert len(set(train_patients).intersection(set(val_patients))) == 0,\
                ("Patients are split between train and val sets")
        # print(f"Train patients: {len(train_patients)}\nVal patients: {len(val_patients)}")

        if self.hparams.train_val_split[0] != -1:
            # Use the provided split
            train_idx = np.random.choice(train_idx, self.hparams.train_val_split[0], replace=False)

        # Create train and validation datasets
        self.train_dataset = TransformSubset(dataset,
                                             train_idx,
                                             self.all_transforms,
                                             img_mask_transform=self.hparams.img_mask_transform,
                                             text_mask_transform=self.hparams.text_mask_transform)
        self.val_dataset = TransformSubset(dataset,
                                           val_idx,
                                           self.req_transforms,
                                           img_mask_transform=self.hparams.img_mask_transform,
                                           text_mask_transform=self.hparams.text_mask_transform)

        # Update and log train/val split counts
        # With seed 42 -> Train/Val split: [355634, 5044]
        self.hparams.train_val_split = [len(self.train_dataset), len(self.val_dataset)]
        # print(f"Train/Val split: {self.hparams.train_val_split}")

    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
        )
    
    def val_dataloader(self):
        return DataLoader(
            dataset=self.val_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
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
    cfg = omegaconf.OmegaConf.load(root / "configs" / "datamodule" / "mimic_cxr.yaml")
    print(cfg.pretty())
    _ = hydra.utils.instantiate(cfg)
