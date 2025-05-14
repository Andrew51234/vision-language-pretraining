import os
import pickle
import pandas as pd
from PIL import Image
import torch
import numpy as np
from torch.utils.data import Dataset

class NLMCXRDataSet(Dataset):
    def __init__(self, dataset_file, transform=None, tokenizer=None):
        """
        Args:
            image_dir (str): Path to the directory containing PNG images.
            captions_file (str): Path to the CSV file containing the captions.
            transform (callable, optional): Optional transform to be applied
                on a sample (e.g., image transforms).
        """
        super(NLMCXRDataSet, self).__init__()
        self.transform = transform
        self.tokenizer = tokenizer

        assert os.path.exists(dataset_file) and os.path.splitext(dataset_file)[1].lower()==".pkl", f"Check file path exists and has the extension .pkl\n Given file: {dataset_file}"
        
        with open(dataset_file, 'rb') as f:
            self.df = pickle.load(f)

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        """
        Retrieves the image and corresponding text data for the given index.
        
        Args:
            idx (int): Index of the data sample to retrieve.
        
        Returns:
            Sample: A dictionary containing the image, selected text, and tokenized data.
        """

        record = self.df.iloc[int(idx)]

        # Load the image
        image_path = record['image_file_path_resized'].replace('aimspace', 'miltank')
        image = Image.open(image_path).convert('RGB')
        assert image is not None, f"Failed to load image {image_path}"

        findings = record['findings_tokenized']
        impressions = record['impression_tokenized']
        find_impres = findings + impressions
        find_impres = ' '.join(find_impres)

        # text = np.random.choice(find_impres)

        if self.transform:
            image = self.transform(image)

        if self.tokenizer:
            unsqueezed_tokenized_input_data = self.tokenizer(find_impres, is_split_into_words=False, max_length=128, padding="max_length", truncation=True, return_tensors="pt")
            tokenized_input_data = {k: v.squeeze() for k, v in unsqueezed_tokenized_input_data.items()}

        decoded_texts = self.retrieve_text_from_tokenized_data(unsqueezed_tokenized_input_data)
        return {'image': image, 'text': decoded_texts, 'tokenized_data': tokenized_input_data}

    def retrieve_text_from_tokenized_data(self, tokenized_data):
        """Decodes the tokenized input data back to the original text."""
        return self.tokenizer.batch_decode(tokenized_data['input_ids'], skip_special_tokens=True)