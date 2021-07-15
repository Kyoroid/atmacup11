from abc import ABC
import torch
from torch.optim import Adam
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
import pytorch_lightning as pl


class BaseRegressor(pl.LightningModule, ABC):
    def configure_optimizers(self):
        init_lr = 5e-4
        optimizer = Adam(self.parameters(), lr=init_lr)
        scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.25)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "monitor": "loss/val",
            },
        }

    def training_step(self, train_batch, batch_idx):
        x, y_gt = train_batch
        y_pred = self.forward(x)
        loss_rmse = torch.sqrt(F.mse_loss(y_pred, y_gt) + 1e-8)
        return loss_rmse

    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x["loss"] for x in outputs]).mean()
        self.log("loss/train", avg_loss)

    def validation_step(self, val_batch, batch_idx):
        x, y_gt = val_batch
        y_pred = self.forward(x)
        loss = torch.sqrt(F.mse_loss(y_pred, y_gt) + 1e-8)
        categorical_pred = torch.clamp((y_pred - 1550) / 100, min=0, max=3)
        categorical_gt = torch.clamp((y_gt - 1550) / 100, min=0, max=3)
        score = torch.sqrt(F.mse_loss(categorical_pred, categorical_gt) + 1e-8)
        return {"val_loss": loss, "score": score}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([y["val_loss"] for y in outputs]).mean()
        avg_score = torch.stack([y["score"] for y in outputs]).mean()
        self.log("loss/val", avg_loss)
        self.log("score/val", avg_score)
