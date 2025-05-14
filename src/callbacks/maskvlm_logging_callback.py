from matplotlib import pyplot as plt
import seaborn as sns
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
import torch
from transformers import AutoTokenizer
import wandb

from src.models.maskvlm_module import MaskVLMModule


class MaskVLMLoggingCallback(Callback):
    def __init__(self, text_model_name: str):
        """
        Callback for logging MaskVLM model outputs during validation.

        Args:
            text_model_name (str): Name of the tokenizer model.
        """
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(text_model_name, use_fast=True)
        self.val_iter = None
        self.train_iter = None

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        """
        Runs at the end of each validation epoch to log reconstructed images and text.
        """
        if not isinstance(pl_module, MaskVLMModule):
            return

        if self.val_iter is None:
            self.val_iter = iter(trainer.datamodule.val_dataloader())

        batch, self.val_iter = self._fetch_batch(self.val_iter, trainer.datamodule.val_dataloader, trainer)
        if batch is None:
            return

        txt_logits, reconstructed_img, z_img, z_txt = self._forward_pass(pl_module, batch)

        idx = torch.randint(0, batch['image'].shape[0], (1,)).item()
        self._prepare_and_log_outputs(trainer, batch, txt_logits, reconstructed_img, z_img, z_txt, idx, state='val')

    def on_train_epoch_end(self, trainer, pl_module):
        """
        Runs at the end of each validation epoch to log reconstructed images and text.
        """
        if not isinstance(pl_module, MaskVLMModule):
            return

        if self.train_iter is None:
            self.train_iter = iter(trainer.datamodule.train_dataloader())

        batch, self.train_iter = self._fetch_batch(self.train_iter, trainer.datamodule.train_dataloader, trainer)
        if batch is None:
            return

        txt_logits, reconstructed_img, z_img, z_txt = self._forward_pass(pl_module, batch)

        idx = torch.randint(0, batch['image'].shape[0], (1,)).item()
        self._prepare_and_log_outputs(trainer, batch, txt_logits, reconstructed_img, z_img, z_txt, idx, state='train')

    def _fetch_batch(self, data_iter, dataloader_getter, trainer):
        """Fetches a batch and moves it to the device."""
        device = trainer.strategy.root_device
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader_getter())  # Reset iterator when exhausted
            batch = next(data_iter)

        return trainer.strategy.batch_to_device(batch, device), data_iter

    def _forward_pass(self, pl_module: pl.LightningModule, batch: dict):
        """Runs a forward pass through the model in evaluation mode."""
        with torch.inference_mode():
            txt_logits, reconstructed_img, z_img, z_txt, _, _ = pl_module(batch)
        return txt_logits, reconstructed_img, z_img, z_txt

    def _prepare_and_log_outputs(self, trainer, batch, txt_logits, reconstructed_img, z_img, z_txt, idx, state='val'):
        """Prepares and logs images and text reconstruction results to Weights & Biases."""
        original_image = batch['image'][idx]
        reconstructed_image = reconstructed_img[idx]
        image_mask = batch['img_mask'][idx]
        text_input_ids = batch['tokenized_data']['input_ids'][idx]
        text_mask = batch['text_mask'][idx]

        diff_img, reconstructed_image, text, text_reconstructed = process_reconstructed_outputs(
            original_image, reconstructed_image, image_mask, self.tokenizer, txt_logits[idx], text_input_ids
        )

        similarity_matrix_img = self._compute_and_log_similarity_heatmap(z_img, z_txt)
        self._log_to_wandb(trainer, original_image, reconstructed_image, diff_img, similarity_matrix_img, text, text_reconstructed, text_mask, state)

    def _compute_and_log_similarity_heatmap(self, z_img, z_txt):
        """Computes and logs cosine similarity heatmap between z_img and z_txt."""
        similarity_matrix = torch.nn.functional.cosine_similarity(z_img[:, None, :], z_txt[None, :, :], dim=-1).cpu().numpy()

        fig, ax = plt.subplots(figsize=(10, 10))
        sns.heatmap(similarity_matrix, cmap='coolwarm', ax=ax, vmin=-1, vmax=1)
        plt.tight_layout()
        img = wandb.Image(fig, caption="Cosine Similarity between z_img and z_txt")
        plt.close(fig)
        return img

    def _log_to_wandb(self, trainer, original_image, reconstructed_image, diff_img, similarity_matrix_img, text, text_reconstructed, text_mask, state='val'):
        """Logs images and text reconstruction results to Weights & Biases."""
        original_image = wandb.Image(original_image.squeeze().cpu().numpy().transpose(1, 2, 0), caption="Original Image")
        reconstructed_image = wandb.Image(reconstructed_image.squeeze().cpu().numpy(), caption="Reconstructed Image")
        diff_img = wandb.Image(diff_img.squeeze().cpu().numpy(), caption="Difference Image")

        text_mask_str = ''.join(str(int(i)) for i in text_mask.squeeze().cpu().numpy()[:len(text)])
        table = wandb.Table(columns=['Original Text', 'Reconstructed Text', 'Mask'], data=[[text, text_reconstructed, text_mask_str]])

        for logger in trainer.loggers:
            if isinstance(logger, pl.loggers.WandbLogger):
                logger.experiment.log({
                    f'media_{state}/original_image': original_image,
                    f'media_{state}/reconstructed_image': reconstructed_image,
                    f'media_{state}/diff_img': diff_img,
                    f'media_{state}/similarity_matrix': similarity_matrix_img,
                    f'media_{state}/text': table
                })


def process_reconstructed_outputs(
        original_image: torch.Tensor, # (C, H, W)
        reconstructed_image: torch.Tensor, # ( 1 + #patches, patch_size * patch_size * 1)
        image_mask: torch.Tensor, # (H, W)
        tokenizer: AutoTokenizer,
        text_logits: torch.Tensor,
        text_input_ids: torch.Tensor
    ):
    """
    Processes the original and reconstructed text/images for readable logging.
    """
    text = tokenizer.decode(text_input_ids.squeeze())
    pad_idx = text.find(tokenizer.pad_token)
    text = text[:pad_idx] if pad_idx != -1 else text
    text_reconstructed = tokenizer.decode(text_logits.squeeze().argmax(dim=-1))
    text_reconstructed = text_reconstructed[:pad_idx] if pad_idx != -1 else text_reconstructed

    C, H, W = original_image.shape
    reconstructed_image = reconstructed_image[1:, :].view(1, H, W)  # Remove CLS token

    diff_img = torch.where(image_mask,
                           (original_image[0, :, :] - reconstructed_image).abs(),
                           torch.tensor(0.0, device=original_image.device, dtype=original_image.dtype))
    diff_img *= 255

    return diff_img, reconstructed_image, text, text_reconstructed
