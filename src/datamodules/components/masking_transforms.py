import torch
import torch.nn as nn
import random
import math
from typing import Tuple
from copy import deepcopy

from torchvision.transforms import ToTensor

from PIL import Image

class ImageMaskingTransform(nn.Module):
    """
    A PyTorch module that applies a masking transform to an input image by dividing it into patches 
    and randomly masking a fraction of these patches.

    Args:
        patch_size (int): Size of each patch to mask. Default is 32.
        mask_ratio (float): Fraction of patches to mask. Default is 0.6.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
            - masked_image (torch.Tensor): Masked version of input image, shape (C, H, W)
            - mask (torch.Tensor): Binary mask of shape (H//patch_size, W//patch_size)
              where 1 indicates masked patches and 0 indicates unmasked patches

    Note:
        The input image dimensions must be divisible by the patch_size.
    """
    def __init__(
        self,
        patch_size: int = 32,  # Size of each patch to mask
        mask_ratio: float = 0.6,  # Fraction of the image to mask
    ):
        super().__init__()
        self.patch_size = patch_size
        self.mask_ratio = mask_ratio

    def forward(self, image: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Applies the masking transform to the input image.
        
        Args:
            image (Union[torch.Tensor, PIL.Image]): Input image:
                - If tensor: Shape (C, H, W)
                - If PIL Image: Will be converted to tensor
        
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
                - masked_image (torch.Tensor): Masked version of input image, shape (C, H, W)
                - mask (torch.Tensor): Binary mask of shape (1, H, W) where 1 indicates masked
                patches and 0 indicates unmasked patches
        
        Raises:
            AssertionError: If image dimensions are not divisible by patch_size
        """
        # Ensure the input image is a tensor
        if isinstance(image, Image.Image):
            image = ToTensor()(image)

        C, H, W = image.shape
        
        # Ensure the image dimensions are divisible by the patch size
        assert H % self.patch_size == 0 and W % self.patch_size == 0, \
            "Image dimensions must be divisible by patch_size"
        
        num_patches_h = H // self.patch_size
        num_patches_w = W // self.patch_size
        num_patches = num_patches_h * num_patches_w

        # Determine the number of patches to mask
        num_masked_patches = math.ceil(self.mask_ratio * num_patches)

        # Create a mask grid (1 means masked, 0 means unmasked)
        mask = torch.zeros((num_patches_h * num_patches_w), device=image.device, dtype=torch.bool)
        mask_indices = torch.randperm(num_patches)[:num_masked_patches]
        mask[mask_indices] = True
        mask = mask.view(num_patches_h, num_patches_w)
        
        # Expand mask to the full image size
        mask_full = mask.repeat_interleave(self.patch_size, dim=0).repeat_interleave(self.patch_size, dim=1)
        mask_full = mask_full.unsqueeze(0)  # Add channel dimension

        # Apply the mask to the image
        masked_image = image * (1 - mask_full.float())

        return masked_image, mask_full


class TextMaskingTransform(nn.Module):
    """
    A PyTorch module for masking a portion of tokens in text sequences.

    Args:
        - mask_ratio (float): The ratio of tokens to be masked in the sentence, excluding the [CLS]
        and [SEP] token. Default is 0.3.

    Note:
        The first and last token ([CLS], [SEP]) is never masked in the implementation.
    """
    def __init__(self, mask_ratio: float = 0.3):
        super().__init__()
        self.mask_ratio = mask_ratio

    def forward(self, tokens: dict, mask_token_id: int) -> Tuple[dict, torch.Tensor]:
        """
        Applies the masking transform to the input tokens.

        Args:
            tokens (dict): A dictionary containing the tokenized sentence.
            mask_token_id (torch.Tensor): The ID of the mask token.

        Returns:
            Tuple[dict, torch.Tensor]: A tuple containing the masked tokens and the mask tensor.
        """
        # copy tokens to avoid modifying the original
        masked_tokens = deepcopy(tokens)

        # Get the input tokens
        input_ids = masked_tokens['input_ids']
        mask = torch.zeros_like(input_ids, dtype=torch.bool)

        # Determine the number of tokens to mask
        # -2 to exclude the [CLS] and [SEP] token at the beginning and end
        num_tokens = masked_tokens['attention_mask'].sum().item() - 2
        num_masked_tokens = math.ceil(self.mask_ratio * num_tokens)

        # Randomly select token indices to mask
        indices = random.sample(range(num_tokens), num_masked_tokens)
        for idx in indices:
            mask[idx+1] = True
            input_ids[idx+1] = mask_token_id

        return masked_tokens, mask

