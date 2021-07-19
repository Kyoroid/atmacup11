import torch
from torchvision.transforms import Normalize


class Denormalize(Normalize):
    def __init__(self, mean, std):
        self.mean = torch.as_tensor(mean)
        self.std = torch.as_tensor(std)
        std_inv = 1 / (self.std + 1e-9)
        mean_inv = -self.mean * std_inv
        super().__init__(mean_inv, std_inv)
