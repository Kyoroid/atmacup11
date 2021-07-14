from torch.nn import Linear, Sequential, Flatten, Dropout
from torch.optim import Adam
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR
from efficientnet_pytorch import EfficientNet

from net.base_regressor import BaseRegressor


class EfficientNetB0Regressor(BaseRegressor):
    def __init__(self):
        super().__init__()
        self.encoder = EfficientNet.from_name("efficientnet-b0", num_classes=1000)
        self.regressor = Sequential(Flatten(), Dropout(p=0.5), Linear(1000, 1))

    def forward(self, x):
        x = self.encoder(x)
        y = self.regressor(x)
        return y

    def configure_optimizers(self):
        init_lr = 5e-5
        optimizer = Adam(self.parameters(), lr=init_lr)
        scheduler = LambdaLR(optimizer, lr_lambda=lambda epoch: 0.95 ** epoch)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
            },
        }
