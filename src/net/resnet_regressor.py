import torch
from torch.nn import Linear, Sequential, Flatten, Dropout
from torch.optim import Adam
from torch.optim.lr_scheduler import OneCycleLR
import torch.nn.functional as F
from torchvision.models.resnet import resnet18, resnet50
import pytorch_lightning as pl


class ResNet18Regressor(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.encoder = resnet18(pretrained=False)
        self.regressor = Sequential(Flatten(), Dropout(p=0.5), Linear(1000, 1))

    def forward(self, x):
        x = self.encoder(x)
        y = self.regressor(x)
        return y

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


class ResNet50Regressor(ResNet18Regressor):
    def __init__(self):
        super().__init__()
        self.encoder = resnet50(pretrained=False)
        self.regressor = Sequential(Flatten(), Dropout(p=0.5), Linear(1000, 1))

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
