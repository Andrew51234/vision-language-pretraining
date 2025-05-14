import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
from torch import Tensor

class ConVIRTContrastiveCriterion(nn.Module):
    """
    Implements ConVIRT contrastive loss using cosine similarities between image and text embeddings.
    """
    def __init__(self, temperature: float, lamda: float):
        """
        Args:
            temperature (float): Temperature scaling for softmax.
            lamda (float): Weight balancing for image-to-text and text-to-image terms.
        """
        super().__init__()
        self.temperature = temperature
        self.lamda = lamda
    
    def forward(self, image_v, text_u):
        """
        Computes contrastive loss based on the diagonal elements of log-softmaxed similarities.

        Args:
            image_v (Tensor): Image embeddings.
            text_u (Tensor): Text embeddings.

        Returns:
            Tensor: Mean contrastive loss value.
        """
        sim_matrix = F.cosine_similarity(image_v.unsqueeze(1), text_u.unsqueeze(0), dim=-1)  # Shape (N, N)

        l_vu = torch.diag(-F.log_softmax(sim_matrix / self.temperature, dim=1))
        l_uv = torch.diag(-F.log_softmax(sim_matrix.T / self.temperature, dim=1))  # Reuse transposed similarity

        return torch.mean(self.lamda * l_vu + (1 - self.lamda) * l_uv)
    
class ITCLoss(nn.Module):
    """
    Computes Image-Text Contrastive (ITC) loss by comparing batchwise projected image-text features.
    """
    def __init__(self, temperature: float, lamda: float):
        """
        Initialize the ITC loss.
        Args:
            temperature (float): Temperature scaling factor.
        """
        super().__init__()
        self.temperature = temperature
        self.lamda = lamda

    def forward(self, z_im, z_txt):
        """
        Forward pass to compute the loss.
        Args:
            z_im: Tensor of image embeddings of shape (N, D).
            z_txt: Tensor of text embeddings of shape (N, D).
        Returns:
            Scalar loss value.
        Note:
            z_im and z_txt should be normalized before passing to this function.
        """
        # Compute similarity matrix
        sim_matrix = torch.matmul(z_im, z_txt.T) / self.temperature  # Shape: (N, N)

        # Diagonal elements (positive pairs)
        positives = torch.diag(sim_matrix)

        # Logits for image-to-text and text-to-image directions
        log_im_to_txt = positives - torch.logsumexp(sim_matrix, dim=1)
        log_txt_to_im = positives - torch.logsumexp(sim_matrix.T, dim=1)

        # Compute the loss
        loss = -torch.mean(log_im_to_txt * self.lamda + log_txt_to_im * (1 - self.lamda))

        return loss

class MLMLoss(nn.Module):
    """
    Computes the Masked Language Modeling (MLM) loss using cross-entropy on masked tokens.
    """
    def __init__(self):
        super().__init__()

    def forward(
        self,
        text_token_classifier_output: Tensor,
        text_tokenized: Tensor,
        masked_positions: Tensor
    ) -> Tensor:
        """
        Processes only masked logits and calculates cross-entropy with the original tokens.

        Args:
            text_token_classifier_output (Tensor): Predicted token logits [B, t_seq, vocab_size].
            text_tokenized (Tensor): Actual token IDs (B, t_seq).
            masked_positions (Tensor): Boolean mask which positions are masked (B, t_seq).

        Returns:
            Tensor: Scalar MLM loss.
        """
        logits = text_token_classifier_output.flatten(0, 1)  # Flatten (B, t_seq, vocab_size) → (B*t_seq, vocab_size)
        targets = text_tokenized.flatten() # (B*t_seq)
        masked_positions = masked_positions.flatten() # (B*t_seq)

        # Compute the loss only for masked tokens
        mlm_loss = F.cross_entropy(logits, targets, reduction='none')
        mlm_loss = mlm_loss * masked_positions
        mlm_loss = (mlm_loss * masked_positions).sum() / masked_positions.count_nonzero()
        
        return mlm_loss

class MIMLoss(nn.Module):
    """
    Computes the Masked Image Modeling (MIM) loss using mean absolute error on masked patches.
    """
    def __init__(self, image_patch_size: int):
        """
        Args:
            image_patch_size (int): Size of each square patch for masking.
        """
        super().__init__()
        self.image_patch_size = image_patch_size
        self.l1_loss = nn.L1Loss()

    def forward(
        self,
        image_cross_modality_decoder_output: Tensor,
        image: Tensor,
        image_mask: Tensor
    ) -> Tensor:
        """
        Calculates MAE on the reconstructed image patches corresponding to masked regions.

        Args:
            image_cross_modality_decoder_output (Tensor): Reconstructed grayscale patches (B, 1+ #patches, patch_size * patch_size * 1).
            image (Tensor): Original image (B, C, H, W).
            image_mask (Tensor): Masked image (B, 1, H, W).

        Returns:
            Tensor: Scalar MIM loss.
        """
        batch_size, channels, height, width = image.shape
        patch_size = self.image_patch_size

        # Calculate number of patches
        num_patches_h = height // patch_size
        num_patches_w = width // patch_size
        num_patches = num_patches_h * num_patches_w

        # remove the cls token
        image_cross_modality_decoder_output = image_cross_modality_decoder_output[:, 1:, :]

        # Extract the predicted and original masked image patches
        predicted_masked_patches = image_cross_modality_decoder_output.view(
            batch_size, 1, height, width
        )[image_mask]
        original_image_patches = image[:, 0, :, :].unsqueeze(1)[image_mask]

        # Compute the mean absolute error between the predicted and original masked image patches
        mim_loss = self.l1_loss(predicted_masked_patches, original_image_patches)
        return mim_loss



class ITMLoss(nn.Module):
    """
    Image-Text Matching Loss.
    """
    def __init__(self):
        super().__init__()

    def forward(self, logits, y_itm):
        """
        Forward pass to compute ITM Loss.
        Args:
            logits: Tensor of shape (N,), output of the ITM head.
            y_itm: Ground truth labels for alignment (1 for aligned, 0 for not aligned), shape (N,).
        Returns:
            Scalar loss value.
        """
        # Compute binary cross-entropy loss
        loss = F.binary_cross_entropy_with_logits(logits, y_itm.float())
        return loss
