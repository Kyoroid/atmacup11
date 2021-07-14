from torch.nn import Linear, Sequential, Flatten, Dropout
from torchvision.models.resnet import resnet18, resnet50
from net.base_regressor import BaseRegressor


class ResNet18Regressor(BaseRegressor):
    def __init__(self):
        super().__init__()
        self.encoder = resnet18(pretrained=False)
        self.regressor = Sequential(Flatten(), Dropout(p=0.5), Linear(1000, 1))

    def forward(self, x):
        x = self.encoder(x)
        y = self.regressor(x)
        return y


class ResNet50Regressor(ResNet18Regressor):
    def __init__(self):
        super().__init__()
        self.encoder = resnet50(pretrained=False)
        self.regressor = Sequential(Flatten(), Dropout(p=0.5), Linear(1000, 1))

    def forward(self, x):
        x = self.encoder(x)
        y = self.regressor(x)
        return y
