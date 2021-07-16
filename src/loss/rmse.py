import torch
from torch.nn import Module, MSELoss


class RMSELoss(Module):
    def __init__(self):
        super(RMSELoss, self).__init__()

    def forward(self, x, y):
        criterion = MSELoss()
        loss = torch.sqrt(criterion(x, y) + 1e-8)
        return loss
