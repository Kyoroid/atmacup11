import torch
from torch.nn import Linear, Sequential, Flatten, ReLU, Dropout
from torch.optim import SGD
import torch.nn.functional as F
from torchvision.models.resnet import resnet18
import pytorch_lightning as pl


class ResNet18Regressor(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.encoder = resnet18(pretrained=False)
        self.regressor = Sequential(Flatten(), Dropout(p=0.1), Linear(1000, 1))

    def forward(self, x):
        x = self.encoder(x)
        y = self.regressor(x)
        return y

    def configure_optimizers(self):
        optimizer = SGD(self.parameters(), lr=1e-4)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        x, y_gt = train_batch
        y_pred = self.forward(x)
        loss = torch.sqrt(F.mse_loss(y_pred, y_gt) + 1e-8)
        return loss

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
