from abc import ABC
from pathlib import Path
import math
import torch
from torch.nn import Sequential
from torch.optim import SGD
import pytorch_lightning as pl
from torchvision.models.resnet import resnet18
import lightly
from net.resnet_regressor import ResNet18Regressor


class ResNet18SimSiamRegressor(pl.LightningModule, ABC):
    def __init__(
        self,
        learning_rate: float = 1e-3,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.lr = None
        self.learning_rate = learning_rate
        self.criterion = lightly.loss.SymNegCosineSimilarityLoss()
        classifier = resnet18(pretrained=False)
        self.encoder = Sequential(*list(classifier.children())[:-1])
        self.model = lightly.models.SimSiam(
            backbone=self.encoder,
            num_ftrs=512,
            proj_hidden_dim=128,
            pred_hidden_dim=128,
            out_dim=128,
        )

    def configure_optimizers(self):
        optimizer = SGD(
            self.parameters(),
            lr=(self.lr or self.learning_rate),
            momentum=0.9,
            weight_decay=5e-4,
        )
        self.logger.experiment.tag({"optimizer": optimizer.__class__.__name__})
        return optimizer

    def forward(self, x0, x1):
        y0, y1 = self.model(x0, x1)
        return y0, y1

    def training_step(self, train_batch, batch_idx):
        (x0, x1), _, _ = train_batch
        y0, y1 = self.forward(x0, x1)
        loss = self.criterion(y0, y1)

        # calculate the per-dimension standard deviation of the outputs
        # we can use this later to check whether the embeddings are collapsing
        output, _ = y0
        output = output.detach()
        output = torch.nn.functional.normalize(output, dim=1)

        output_std = torch.std(output, 0)
        output_std = output_std.mean()
        return {"loss": loss, "output_std": output_std}

    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x["loss"] for x in outputs]).mean()
        avg_output_std = torch.stack([x["output_std"] for x in outputs]).mean()
        collapse_level = max(0.0, 1 - math.sqrt(self.model.out_dim) * avg_output_std)
        self.log("loss/train", avg_loss)
        self.log("collapse_level/train", collapse_level)


class ResNet18TransferRegressor(ResNet18Regressor):
    def __init__(
        self,
        learning_rate: float = 0.001,
        dropout_rate=0.5,
        n_features=1000,
        ckpt_path: Path = None,
    ):
        super().__init__(
            learning_rate=learning_rate,
            dropout_rate=dropout_rate,
            n_features=n_features,
        )
        ssl_model = ResNet18SimSiamRegressor().load_from_checkpoint(
            checkpoint_path=ckpt_path
        )
        ssl_encoder = ssl_model.model.backbone
        layer_prefix_dict = {
            "0": "conv1",
            "1": "bn1",
            "2": "relu",
            "3": "maxpool",
            "4": "layer1",
            "5": "layer2",
            "6": "layer3",
            "7": "layer4",
            "8": "avgpool",
        }
        pretrained_state_dict = {}
        for ssl_layer_name, weight in ssl_encoder.state_dict().items():
            layer_name_part = ssl_layer_name.split(".")
            main_layer_name = ".".join(
                [layer_prefix_dict[layer_name_part[0]], *layer_name_part[1:]]
            )
            pretrained_state_dict[main_layer_name] = weight
        encoder_state_dict = self.encoder.state_dict()
        encoder_state_dict.update(pretrained_state_dict)
        self.encoder.load_state_dict(encoder_state_dict)
        for param in self.encoder.parameters():
            param.requires_grad = False
