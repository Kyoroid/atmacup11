from pathlib import Path
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from vit_pytorch import ViT
from net.base_regressor import BaseRegressor


class ViTRegressor(BaseRegressor):
    def __init__(
        self,
        learning_rate: float = 1e-3,
        dropout_rate=0.5,
        n_features=1024,
        ckpt_path: Path = None,
    ):
        super().__init__(
            self.__class__.__name__,
            learning_rate,
            dropout_rate=dropout_rate,
            n_features=n_features,
            ckpt_path=ckpt_path,
        )
        self.model = ViT(
            image_size=224,
            patch_size=32,  # (224 // 32) ** 2 = 49
            num_classes=1,  # Number of classes to classify.
            dim=n_features,  # Last dimension of output tensor after linear transformation.
            depth=5,  # Number of Transformer blocks.
            heads=16,  # Number of heads in Multi-head Attention layer.
            mlp_dim=2048,  # Dimension of the MLP (FeedForward) layer.
            channels=3,  # Number of image's channels.
            dropout=dropout_rate,  # Dropout rate.
            pool="cls",  # token pooling
        )

    def forward(self, x):
        y = self.model(x)
        return y

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=(self.lr or self.learning_rate))
        steps_per_epoch = len(self.train_dataloader())
        scheduler = OneCycleLR(
            optimizer,
            max_lr=self.learning_rate,
            steps_per_epoch=steps_per_epoch,
            epochs=self.trainer.max_epochs,
            pct_start=0.1,
            anneal_strategy="cos",
        )
        self.logger.experiment.tag({"optimizer": optimizer.__class__.__name__})
        self.logger.experiment.tag({"scheduler": scheduler.__class__.__name__})
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }
