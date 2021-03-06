from pathlib import Path
from torch.nn import Linear, Sequential, Dropout
import timm
from net.base_regressor import BaseRegressor


class EfficientNetB0Regressor(BaseRegressor):
    def __init__(
        self,
        learning_rate: float = 7e-3,
        dropout_rate=0.5,
        n_features=1280,
        ckpt_path: Path = None,
    ):
        super().__init__(
            self.__class__.__name__,
            learning_rate,
            dropout_rate=dropout_rate,
            n_features=n_features,
            ckpt_path=ckpt_path,
        )
        self.encoder = timm.create_model("efficientnet_b0", pretrained=False)
        self.encoder.classifier = Linear(1280, n_features)
        self.regressor = Sequential(Dropout(p=dropout_rate), Linear(n_features, 1))

    def forward(self, x):
        x = self.encoder(x)
        y = self.regressor(x)
        return y
