from pathlib import Path
from torch.nn import Linear, Sequential, Dropout
from torchvision.models.resnet import resnet18, resnet50
from net.base_regressor import BaseRegressor


class ResNet18Regressor(BaseRegressor):
    def __init__(
        self,
        learning_rate: float = 1e-3,
        dropout_rate=0.5,
        n_features=1000,
        ckpt_path: Path = None,
    ):
        super().__init__(
            self.__class__.__name__,
            learning_rate,
            dropout_rate=dropout_rate,
            n_features=n_features,
            ckpt_path=ckpt_path,
        )
        self.encoder = resnet18(pretrained=False)
        self.encoder.fc = Linear(512, n_features)
        self.regressor = Sequential(Dropout(p=dropout_rate), Linear(n_features, 1))

    def forward(self, x):
        x = self.encoder(x)
        y = self.regressor(x)
        return y


class ResNet50Regressor(BaseRegressor):
    def __init__(
        self,
        learning_rate: float = 4e-3,
        dropout_rate=0.5,
        n_features=1000,
        ckpt_path: Path = None,
    ):
        super().__init__(
            self.__class__.__name__,
            learning_rate,
            dropout_rate=dropout_rate,
            n_features=n_features,
            ckpt_path=ckpt_path,
        )
        self.encoder = resnet50(pretrained=False)
        self.encoder.fc = Linear(2048, n_features)
        self.regressor = Sequential(Dropout(p=dropout_rate), Linear(n_features, 1))

    def forward(self, x):
        x = self.encoder(x)
        y = self.regressor(x)
        return y
