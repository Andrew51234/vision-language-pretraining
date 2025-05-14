import os
import pickle

from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from .components.masking_transforms import ImageMaskingTransform, TextMaskingTransform

class MIMICCXRDataset(Dataset):
    def __init__(self,
                 df_file:str,
                 config=None,
                 n_sentences=False,
                 transforms=None,
                 tokenizer=None):
        super(MIMICCXRDataset, self).__init__()

        self.config = config
        self.n_sentences = n_sentences
        assert os.path.exists(df_file) and os.path.splitext(df_file)[1].lower()==".pkl", f"Check file path exists and has the extension .pkl\n Given file: {df_file}"

        with open(df_file, 'rb') as f:
            self.df = pickle.load(f)
        # aimspace got moved to miltank
        self.df['resized_image_file'] = self.df['resized_image_file'].apply(lambda x: x.replace('aimspace', 'miltank'))

        self.transforms = transforms
        self.tokenizer = tokenizer

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        """
        Retrieves the image and corresponding text data for the given index.
        
        Args:
            idx (int): Index of the data sample to retrieve.
        
        Returns:
            Sample (dict): A dictionary containing the image and text data.
                keys:
                    image (PIL.Image or torch.Tensor): The image data (transformed if transforms are provided).
                    text (str): The raw text data.
                    tokenized_data (dict, optional): The tokenized text data if tokenizer is provided.
                    masked_img (torch.Tensor, optional): Masked version of the image if img_mask_transform provided.
                    img_mask (torch.Tensor, optional): Binary mask indices if img_mask_transform provided.
                    masked_tokenized_data (dict, optional): Masked tokenized text if text_mask_transform provided.
                    text_mask (torch.Tensor, optional): Binary mask indices if text_mask_transform provided.
        """
        record = self.df.iloc[int(idx)]
        item_out = {}

        #### Load and prepare the image
        image = Image.open(record['resized_image_file']).convert('RGB')
        assert image is not None, f"Failed to load image {record['resized_image_file']}"

        if self.transforms:
            image = self.transforms(image)
        
        item_out['image'] = image

        #### Load and prepare the text data
        # sample a random sentence from the findings/impressions
        findings = record['finding_sentences']
        impressions = record['impression_sentences']
        find_impres = findings + impressions
        assert len(find_impres)!=0,f"Issue findings/impression of {record['patient_folder']}/{record['patient_id']}/{record['study_id']}"

        if self.n_sentences >= len(find_impres) or self.n_sentences == -1:
            text = ' '.join(find_impres)
        else:
            indices = np.random.choice(len(find_impres), self.n_sentences, replace=False)
            indices = np.sort(indices)
            text = ' '.join([find_impres[i] for i in indices])
            
        item_out['text'] = text

        if self.tokenizer:
            # Tokenize the raw text data
            text_tokenized = self.tokenizer(text,
                                            max_length=128,
                                            padding="max_length",
                                            truncation=True,
                                            return_tensors="pt",
                                            add_special_tokens=True)
            text_tokenized_data = {k: v.squeeze() for k, v in text_tokenized.items()}
            item_out['tokenized_data'] = text_tokenized_data

        return item_out

class TransformSubset(torch.utils.data.Dataset):
    """
    A dataset wrapper that applies transformations to a subset of a given dataset. Optionally, it
    also applies image masking.

    Args:
        - dataset (torch.utils.data.Dataset): The original dataset.
        - indices (list): List of indices to create the subset.
        - transforms (callable, optional): A function/transform that takes in an image and returns
        a transformed version. Default is None.
        - img_mask_transform (ImageMaskingTransform, optional): A function/transform that takes in
        an image and returns a masked image and mask. Default is None.
        - text_mask_transform (TextMaskingTransform, optional): A function/transform that takes in
        a tokenized text and returns a masked tokenized text and mask. Default is None.

    Note:
        - If the `transforms` argument is provided, it will be applied to the image data.
        - If the `img_mask_transform` argument is provided, it will be applied to the image data.
        - If the `text_mask_transform` argument is provided, it will be applied to the tokenized
        text data.

    Raises:
        AssertionError: If the original dataset already has transformations applied.
    """

    def __init__(self,
                 dataset,
                 indices,
                 transforms=None,
                 img_mask_transform: ImageMaskingTransform=None,
                 text_mask_transform: TextMaskingTransform=None):
        super(TransformSubset, self).__init__()
        assert dataset.transforms == None, "Transforms should be applied only once"
        self.subset = torch.utils.data.Subset(dataset, indices)
        self.transforms = transforms
        self.img_mask_transform = img_mask_transform
        self.text_mask_transform = text_mask_transform
        if self.text_mask_transform is not None:
            self.mask_token_id = self.subset.dataset.tokenizer.mask_token_id

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        sample = self.subset[idx]

        if self.transforms and 'image' in sample.keys():
            sample['image'] = self.transforms(sample['image'])

        if self.img_mask_transform and 'image' in sample.keys():
            masked_img, mask = self.img_mask_transform(sample['image'])
            sample['masked_img'] = masked_img
            sample['img_mask'] = mask

        if self.text_mask_transform and 'tokenized_data' in sample.keys():
            masked_text, mask = self.text_mask_transform(sample['tokenized_data'],
                                                         mask_token_id=self.mask_token_id)
            sample['masked_tokenized_data'] = masked_text
            sample['text_mask'] = mask

        return sample

if __name__ == "__main__":
    print("TODO: Need to write a basic test")
