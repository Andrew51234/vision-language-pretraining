import torch
from typing import Any, List
from pytorch_lightning import LightningModule
from torchmetrics import MeanMetric
import wandb
import os
import time


class NLMCXRLitModule(LightningModule):
    """Example of LightningModule for NLMCXR Image-Text Retrieval.

    A LightningModule organizes your PyTorch code into 6 sections:
        - Computations (init)
        - Train loop (training_step)
        - Validation loop (validation_step)
        - Test loop (test_step)
        - Prediction Loop (predict_step)
        - Optimizers and LR Schedulers (configure_optimizers)

    Docs:
        https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html
    """

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        criterion: torch.nn.modules.loss,
    ):    
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False, ignore=["model", "criterion"])
        
        self.model = model
        self.criterion = criterion

        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        # Initialize embedding storage
        self.val_image_embs = []
        self.val_text_embs = []

    def forward(self, x: torch.Tensor):
        return self.model(x)

    def step(self, batch: Any):
        image_embeddings, text_embeddings = self.forward(batch)

        # Compute similarity scores
        similarity = torch.matmul(image_embeddings, text_embeddings.t())

        # Compute loss
        loss = self.criterion(image_embeddings, text_embeddings)

        return loss, similarity, image_embeddings, text_embeddings

    def training_step(self, batch: Any, batch_idx: int):
        loss, _, _, _ = self.step(batch)

        # update and log metrics
        self.train_loss(loss)
        
        # Log Loss
        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True,prog_bar=True) 
        # we can return here dict with any tensors
        # and then read it in some callback or in `training_epoch_end()` below
        # remember to always return loss from `training_step()` or backpropagation will fail!
        return {"loss": loss}  
    
    def log_retrieval_results(self, batch, similarity):
        top_k = 3
        images, texts = batch["image"], batch["text"]

        # ---- Image-to-Text Retrieval ----
        top_text_matches = torch.argsort(similarity, dim=1, descending=True)[:, :top_k]

        for i in range(min(3, len(images))):
            img = images[i].cpu().permute(1, 2, 0).numpy()
            true_label = texts[0][i]            
            matched_texts = [texts[0][j] for j in top_text_matches[i].cpu().numpy()]

            # Log text and image using WandB
            self.logger.experiment.log({
            f"Example {i}/Image": wandb.Image(img, caption=f"Top Matches: {matched_texts} | True Label: {true_label}")
        })

        # # ---- Text-to-Image Retrieval ----
        # top_image_matches = torch.argsort(similarity, dim=0, descending=True)[:top_k, :]

        # for i in range(min(3, len(texts[0]))):
        #     text = texts[0][i]
        #     matched_images = [images[j].cpu().permute(1, 2, 0).numpy() for j in top_image_matches[:, i].cpu().numpy()]

        #     self.logger.experiment.log({
        #         f"Example {i}/Text-to-Image Images": [
        #             wandb.Image(img, caption=f"Matched Image {j+1}") for j, img in enumerate(matched_images)
        #         ],
        #         f"Example {i}/Text": text
        #     })


    def plot_similarity(self, similarity):
        """Save and log similarity matrix as a heatmap compatible with WandB."""
        import matplotlib.pyplot as plt

        plt.figure(figsize=(8, 6))
        plt.imshow(similarity.cpu().detach().numpy(), cmap="viridis")
        plt.colorbar()
        plt.title("Similarity Matrix")
        plt.xlabel("Text Embeddings")
        plt.ylabel("Image Embeddings")
        plt.tight_layout()

        # Save and log with WandB


        timestamp = int(time.time())  # Get current time in seconds
        filename = f"similarity_epoch_{self.current_epoch}_{timestamp}.png"
        plt.savefig(filename)
        self.logger.experiment.log({
            "Similarity Heatmap": wandb.Image(filename, caption="Similarity Matrix")
        })
        plt.close()
        os.remove(filename)

    def validation_step(self, batch: Any, batch_idx: int):
        loss, similarity, image_emb, text_emb = self.step(batch)

        self.val_image_embs.append(image_emb.detach().cpu())
        self.val_text_embs.append(text_emb.detach().cpu())

        if batch_idx == 0:  
            self.plot_similarity(similarity)
            self.log_retrieval_results(batch, similarity)
        
        self.val_loss(loss)
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)

        return {"loss": loss}  

    def validation_epoch_end(self, outputs):
        # Concatenate all collected embeddings
        image_embs = torch.cat(self.val_image_embs).to(self.device)
        text_embs = torch.cat(self.val_text_embs).to(self.device)
        
        # Calculate full similarity matrix
        sim_matrix = torch.matmul(image_embs, text_embs.T)
        
        # Compute recalls for both directions
        img2text_recall = self._calculate_recall(sim_matrix)
        text2img_recall = self._calculate_recall(sim_matrix.T)
        
        # Log metrics
        self.log_dict({
            'val/recall_img2text@1': img2text_recall[0],
            'val/recall_img2text@5': img2text_recall[1],
            'val/recall_text2img@1': text2img_recall[0],
            'val/recall_text2img@5': text2img_recall[1],
        }, prog_bar=True)
            

    def _calculate_recall(self, sim_matrix, ks=(1, 5)):
        """Calculate recall@k for given similarity matrix"""
        num_queries = sim_matrix.size(0)
        targets = torch.arange(num_queries, device=sim_matrix.device)
        
        _, topk_indices = sim_matrix.topk(max(ks), dim=1)
        recalls = []
        for k in ks:
            correct = (topk_indices[:, :k] == targets.unsqueeze(1)).any(dim=1)
            recalls.append(correct.float().mean().item())
        return recalls
    
    def on_validation_epoch_start(self):
        # Initialize embedding storage
        self.val_image_embs = []
        self.val_text_embs = []

    def training_epoch_end(self, outputs: List[Any]):
        # `outputs` is a list of dicts returned from `training_step()`
        pass        

    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
        """
        optimizer = self.hparams.optimizer(params=self.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)   
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    # "interval": "step",
                    },
            }
        return {"optimizer": optimizer}
  

if __name__ == "__main__":
    import hydra
    import omegaconf
    import pyrootutils

    root = pyrootutils.setup_root(__file__, pythonpath=True)
    cfg = omegaconf.OmegaConf.load(root / "configs" / "model" / "nlmcxr_finetune_model.yaml")
    _ = hydra.utils.instantiate(cfg)
