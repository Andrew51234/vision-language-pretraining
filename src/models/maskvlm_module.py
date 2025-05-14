
from itertools import chain
import pytorch_lightning as pl
import torch
from torch import optim
from torchmetrics import MeanMetric

from .maskvlm_model import MaskVLM
from .criterion import ITMLoss, MIMLoss, MLMLoss

class MaskVLMModule(pl.LightningModule):
    def __init__(
            self,
            net: MaskVLM,
            itc_loss: torch.nn.Module,
            image_patch_size=16,
            lr_encoder=1e-5,
            lr_cross=3e-4,
            lr_other=3e-4,
            warmup_epochs=5,
            weight_decay=0.05,
            max_epochs=100,
            loss_weight_itc=1.0,
            loss_weight_itm=1.0,
            loss_weight_mlm=.5,
            loss_weight_mim=.5,
            ):
        super().__init__()
        self.save_hyperparameters(logger=False, ignore=['net', 'itc_loss'])

        self.net = net
        self.itc_loss = itc_loss

        self.itm_loss = ITMLoss()
        self.mim_loss = MIMLoss(self.hparams.image_patch_size)
        self.mlm_loss = MLMLoss()

        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.train_mlm_loss = MeanMetric()
        self.train_mim_loss = MeanMetric()
        self.train_itc_loss = MeanMetric()
        self.train_itm_loss = MeanMetric()

        self.val_loss = MeanMetric()
        self.val_mlm_loss = MeanMetric()
        self.val_mim_loss = MeanMetric()
        self.val_itc_loss = MeanMetric()
        self.val_itm_loss = MeanMetric()


    def forward(self, batch):
        return self.net(batch)
    
    def _update_train_loss(self, loss_itc, loss_itm, loss_mlm, loss_mim, loss):
        self.train_itc_loss.update(loss_itc)
        self.train_itm_loss.update(loss_itm)
        self.train_mlm_loss.update(loss_mlm)
        self.train_mim_loss.update(loss_mim)
        self.train_loss.update(loss)

    def _update_val_loss(self, loss_itc, loss_itm, loss_mlm, loss_mim, loss):
        self.val_itc_loss.update(loss_itc)
        self.val_itm_loss.update(loss_itm)
        self.val_mlm_loss.update(loss_mlm)
        self.val_mim_loss.update(loss_mim)
        self.val_loss.update(loss)

    def _get_loss(self, outputs, batch):
        txt_logits, reconstructed_img, z_img, z_txt, itm_logits, itm_labels = outputs
        loss_itc = self.itc_loss(z_img, z_txt)
        loss_itm = self.itm_loss(itm_logits, itm_labels)
        loss_mlm = self.mlm_loss(txt_logits,
                                    batch['tokenized_data']['input_ids'],
                                    batch['text_mask'])
        loss_mim = self.mim_loss(reconstructed_img,
                                    batch['image'],
                                    batch['img_mask'])
        loss = self._weighted_loss(loss_itc, loss_itm, loss_mlm, loss_mim)
        return loss_itc, loss_itm, loss_mlm, loss_mim, loss

    def _weighted_loss(self, loss_itc, loss_itm, loss_mlm, loss_mim):
        loss = self.hparams.loss_weight_itc * loss_itc \
                + self.hparams.loss_weight_itm * loss_itm \
                + self.hparams.loss_weight_mlm * loss_mlm \
                + self.hparams.loss_weight_mim * loss_mim
        return loss

    def training_step(self, batch, batch_idx):
        outputs = self(batch)
        loss_itc, loss_itm, loss_mlm, loss_mim, loss = self._get_loss(outputs, batch)

        self._update_train_loss(loss_itc, loss_itm, loss_mlm, loss_mim, loss)

        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/loss_itc", self.train_itc_loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("train/loss_itm", self.train_itm_loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("train/loss_mlm", self.train_mlm_loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("train/loss_mim", self.train_mim_loss, on_step=False, on_epoch=True, prog_bar=False)

        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        outputs = self(batch)
        loss_itc, loss_itm, loss_mlm, loss_mim, loss = self._get_loss(outputs, batch)

        self._update_val_loss(loss_itc, loss_itm, loss_mlm, loss_mim, loss)

        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/loss_itc", self.val_itc_loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("val/loss_itm", self.val_itm_loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("val/loss_mlm", self.val_mlm_loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("val/loss_mim", self.val_mim_loss, on_step=False, on_epoch=True, prog_bar=False)

        return {'loss': loss}

    def configure_optimizers(self):
        # Separate encoder and cross-modality parameters
        encoder_params = chain(self.net.image_model.parameters(),
                               self.net.text_model.parameters())
        cross_modality_params = chain(self.net.img_cross_encoder.parameters(),
                                      self.net.txt_cross_encoder.parameters(),
                                      self.net.img_cross_decoder.parameters())
        other_params = chain(self.net.txt_classifier.parameters(),
                             self.net.itc_head.parameters(),
                             self.net.itm_head.parameters())

        # Define two parameter groups with different learning rates
        optimizer = optim.AdamW(
            [
                {"params": encoder_params, "lr": self.hparams.lr_encoder},
                {"params": cross_modality_params, "lr": self.hparams.lr_cross},
                {"params": other_params, "lr": self.hparams.lr_other}
            ],
            weight_decay=self.hparams.weight_decay
        )

        # Warm-up from 0 to lr_cross in warmup_epochs
        def lr_lambda(epoch):
            if epoch < self.hparams.warmup_epochs:
                return float(epoch + 1) / float(self.hparams.warmup_epochs)
            return 1.0

        warmup_scheduler = {
            "scheduler": optim.lr_scheduler.LambdaLR(optimizer, lr_lambda),
            "interval": "epoch",
            "frequency": 1,
        }

        # Cosine anneal from lr_cross to lr_encoder after warm-up
        scheduler = {
            "scheduler": optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.hparams.max_epochs,
                eta_min=self.hparams.lr_encoder
            ),
            "interval": "epoch",
            "frequency": 1,
            "name": "Charts"
        }

        scheduler = {
            "scheduler": optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=0.8,
                patience=3,
                verbose=True
            ),
            "interval": "epoch",
            "frequency": 1,
            "monitor": "val/loss",
            "name": "Charts"
        }

        return [optimizer], [warmup_scheduler, scheduler] #, plateau_scheduler]