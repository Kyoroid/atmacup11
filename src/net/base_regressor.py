from abc import ABC
from pathlib import Path
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
import pytorch_lightning as pl
from loss.rmse import RMSELoss


class BaseRegressor(pl.LightningModule, ABC):
    def __init__(
        self,
        cls_name: str,
        learning_rate: float = 5e-4,
        ckpt_path: Path = None,
        *args,
        **kwargs
    ):
        super().__init__()
        self.save_hyperparameters()
        self.lr = None
        self.learning_rate = learning_rate
        self.rmse = RMSELoss()

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=(self.lr or self.learning_rate))
        scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: 0.95 ** epoch)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
            },
        }

    def training_step(self, train_batch, batch_idx):
        x, y_gt = train_batch
        y_pred = self.forward(x)
        loss_rmse = self.rmse(y_pred, y_gt)
        return loss_rmse

    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x["loss"] for x in outputs]).mean()
        self.log("loss/train", avg_loss)

    def validation_step(self, val_batch, batch_idx):
        x, y_gt = val_batch
        y_pred = self.forward(x)
        loss = self.rmse(y_pred, y_gt)
        categorical_pred = torch.clamp((y_pred - 1550) / 100, min=0, max=3)
        categorical_gt = torch.clamp((y_gt - 1550) / 100, min=0, max=3)
        score = self.rmse(categorical_pred, categorical_gt)
        return {"val_loss": loss, "score": score}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([y["val_loss"] for y in outputs]).mean()
        avg_score = torch.stack([y["score"] for y in outputs]).mean()
        self.log("loss/val", avg_loss)
        self.log("score/val", avg_score)
