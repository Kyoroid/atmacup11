import torch
from torch.nn import Linear, Sequential, Flatten, Dropout, Sigmoid
from torch.optim import Adam
from torch.optim.lr_scheduler import OneCycleLR
import torch.nn.functional as F
from torchvision.models.resnet import resnet18
import pytorch_lightning as pl


class ResNet18Mixer(pl.LightningModule):
    def __init__(self, n_categories=1):
        super().__init__()
        self.encoder = resnet18(pretrained=False)
        self.head = Sequential(
            Flatten(), Dropout(p=0.5), Linear(1000, 1 + n_categories)
        )
        self.sigmoid = Sigmoid()
        self.n_categories = n_categories

    def forward(self, x):
        x = self.encoder(x)
        y = self.head(x)
        y_reg, y_clf = y[:, :1], self.sigmoid(y[:, 1:])
        return y_reg, y_clf

    def configure_optimizers(self):
        init_lr = 5e-4
        pct_start = 0.1
        steps_per_epoch = len(self.train_dataloader())
        optimizer = Adam(self.parameters(), lr=init_lr)
        scheduler = OneCycleLR(
            optimizer,
            max_lr=init_lr,
            steps_per_epoch=steps_per_epoch,
            epochs=self.trainer.max_epochs,
            pct_start=pct_start,
            anneal_strategy="cos",
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }

    def training_step(self, train_batch, batch_idx):
        x, y_gt = train_batch
        y_reg_gt, y_clf_gt = y_gt[:, :1], y_gt[:, 1:]
        y_reg_pred, y_clf_pred = self.forward(x)
        rmse_loss = torch.sqrt(F.mse_loss(y_reg_pred, y_reg_gt) + 1e-8)
        ce_loss = torch.sqrt(F.mse_loss(y_clf_pred, y_clf_gt) + 1e-8)
        loss = rmse_loss + ce_loss * 100
        return {"loss": loss, "rmse": rmse_loss, "ce": ce_loss}

    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x["loss"] for x in outputs]).mean()
        avg_rmse_loss = torch.stack([x["rmse"] for x in outputs]).mean()
        avg_ce_loss = torch.stack([x["ce"] for x in outputs]).mean()
        self.log("loss/train", avg_loss)
        self.log("rmse_loss/train", avg_rmse_loss)
        self.log("ce_loss/train", avg_ce_loss)

    def validation_step(self, val_batch, batch_idx):
        x, y_gt = val_batch
        y_reg_gt, y_clf_gt = y_gt[:, :1], y_gt[:, 1:]
        y_reg_pred, y_clf_pred = self.forward(x)
        rmse_loss = torch.sqrt(F.mse_loss(y_reg_pred, y_reg_gt) + 1e-8)
        ce_loss = torch.sqrt(F.mse_loss(y_clf_pred, y_clf_gt) + 1e-8)
        loss = rmse_loss + ce_loss * 100
        categorical_pred = torch.clamp((y_reg_pred - 1550) / 100, min=0, max=3)
        categorical_gt = torch.clamp((y_reg_gt - 1550) / 100, min=0, max=3)
        score = torch.sqrt(F.mse_loss(categorical_pred, categorical_gt) + 1e-8)
        return {
            "val_loss": loss,
            "val_rmse": rmse_loss,
            "val_ce": ce_loss,
            "score": score,
        }

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([y["val_loss"] for y in outputs]).mean()
        avg_rmse_loss = torch.stack([x["val_rmse"] for x in outputs]).mean()
        avg_ce_loss = torch.stack([x["val_ce"] for x in outputs]).mean()
        avg_score = torch.stack([y["score"] for y in outputs]).mean()
        self.log("loss/val", avg_loss)
        self.log("rmse_loss/val", avg_rmse_loss)
        self.log("ce_loss/val", avg_ce_loss)
        self.log("score/val", avg_score)
