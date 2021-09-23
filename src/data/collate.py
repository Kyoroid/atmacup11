import torch
import torch.nn as nn
from typing import List
import numpy as np
import albumentations as A


atmacup_normalize = {
    "mean": [0.77695272, 0.74355234, 0.67019692],
    "std": [0.16900558, 0.16991152, 0.17102272],
}


class BaseCollateFunction(nn.Module):
    """Base class for other collate implementations.

    Takes a batch of images as input and transforms each image into two
    different augmentations with the help of random transforms. The images are
    then concatenated such that the output batch is exactly twice the length
    of the input batch.

    Attributes:
        transform:
            A set of torchvision transforms which are randomly applied to
            each image.

    """

    def __init__(self, transform: A.Compose):

        super(BaseCollateFunction, self).__init__()
        self.transform = transform

    def forward(self, batch: List[tuple]):
        """Turns a batch of tuples into a tuple of batches.

        Args:
            batch:
                A batch of tuples of images, labels, and filenames which
                is automatically provided if the dataloader is built from
                a LightlyDataset.

        Returns:
            A tuple of images, labels, and filenames. The images consist of
            two batches corresponding to the two transformations of the
            input images.

        Examples:
            >>> # define a random transformation and the collate function
            >>> transform = ... # some random augmentations
            >>> collate_fn = BaseCollateFunction(transform)
            >>>
            >>> # input is a batch of tuples (here, batch_size = 1)
            >>> input = [(img, 0, 'my-image.png')]
            >>> output = collate_fn(input)
            >>>
            >>> # output consists of two random transforms of the images,
            >>> # the labels, and the filenames in the batch
            >>> (img_t0, img_t1), label, filename = output

        """
        batch_size = len(batch)

        # list of transformed images
        transforms = [
            self.transform(image=np.array(batch[i % batch_size][0]))[
                "image"
            ].unsqueeze_(0)
            for i in range(2 * batch_size)
        ]
        # list of labels
        labels = torch.LongTensor([item[1] for item in batch])
        # list of filenames
        fnames = [item[2] for item in batch]

        # tuple of transforms
        transforms = (
            torch.cat(transforms[:batch_size], 0),
            torch.cat(transforms[batch_size:], 0),
        )

        return transforms, labels, fnames
