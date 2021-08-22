import torch
from torch.nn import MSELoss


class RMSELoss(MSELoss):
    def __init__(self, reduction: str = "mean", eps: float = 1e-5):
        super().__init__(reduction=reduction)
        self.eps = eps

    def forward(self, input, target):
        criterion = super().forward(input, target)
        loss = torch.sqrt(criterion + self.eps)
        return loss
