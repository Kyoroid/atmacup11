import torchvision.transforms.functional as F


class SquarePad:
    def __init__(self, width, height):
        self.width = width
        self.height = height

    def __call__(self, x):
        height, width = x.shape[1], x.shape[2]
        padding = (0, 0, max(0, self.width - width), max(0, self.height - height))
        return F.pad(x, padding, fill=0, padding_mode="constant")
