from pathlib import Path
import torch
from torch.optim import SGD, AdamW
from torch.optim.lr_scheduler import OneCycleLR
from vit_pytorch import ViT, Dino
from net.base_regressor import BaseRegressor


class ViTDinoRegressor(BaseRegressor):
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
        self.automatic_optimization = False
        model = ViT(
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
        self.learner = Dino(
            model,
            image_size=224,
            hidden_layer="to_latent",  # hidden layer name or index, from which to extract the embedding
            projection_hidden_size=256,  # projector network hidden dimension
            projection_layers=4,  # number of layers in projection network
            num_classes_K=65336,  # output logits dimensions (referenced as K in paper)
            student_temp=0.9,  # student temperature
            teacher_temp=0.04,  # teacher temperature, needs to be annealed from 0.04 to 0.07 over 30 epochs
            local_upper_crop_scale=0.4,  # upper bound for local crop - 0.4 was recommended in the paper
            global_lower_crop_scale=0.5,  # lower bound for global crop - 0.5 was recommended in the paper
            moving_average_decay=0.9,  # moving average of encoder - paper showed anywhere from 0.9 to 0.999 was ok
            center_moving_average_decay=0.9,  # moving average of teacher centers - paper showed anywhere from 0.9 to 0.999 was ok
        )

    def forward(self, x):
        y = self.learner(x)
        return y

    def configure_optimizers(self):
        optimizer = SGD(self.parameters(), lr=(self.lr or self.learning_rate))
        self.logger.experiment.tag({"optimizer": optimizer.__class__.__name__})
        return {
            "optimizer": optimizer,
        }

    def training_step(self, train_batch, batch_idx):
        x, _ = train_batch
        loss = self.forward(x)
        self.optimizers().zero_grad()
        self.manual_backward(loss)
        self.optimizers().step()
        self.learner.update_moving_average()
        return {"loss": loss}


class ViTTransferRegressor(BaseRegressor):
    def __init__(
        self,
        learning_rate: float = 0.001,
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
        ssl_model = ViTDinoRegressor().load_from_checkpoint(checkpoint_path=ckpt_path)
        ssl_model = ssl_model.learner.net
        self.model.load_state_dict(ssl_model.state_dict())

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
