from .base_regressor import BaseRegressor
from .resnet_regressor import ResNet18Regressor, ResNet34Regressor, ResNet50Regressor
from .efficientnet_regressor import EfficientNetB0Regressor
from .resnet_ssl import (
    ResNet18SimSiamRegressor,
    ResNet18TransferRegressor,
)
from .vit_regressor import ViTRegressor
from .vit_ssl import ViTDinoRegressor, ViTTransferRegressor
