import torch
import pytorch_lightning as pl
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics import Accuracy

class Evaluator(pl.LightningModule):
    def __init__(self, model, strategy_name="baseline"):
        super().__init__()
        self.model = model
        self.criterion = nn.CrossEntropyLoss()
        self.acc = Accuracy(task='multiclass', num_classes=self.model.classifier.out_features)
        self.strategy_name = strategy_name

    def training_step(self, batch, batch_idx):
        out = self.model(batch.x, batch.edge_index, batch.batch)
        loss = self.criterion(out, batch.y)
        acc = self.acc(out, batch.y)
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", acc, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        out = self.model(batch.x, batch.edge_index, batch.batch)
        loss = self.criterion(out, batch.y)
        acc = self.acc(out, batch.y)
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        out = self.model(batch.x, batch.edge_index, batch.batch)
        loss = self.criterion(out, batch.y)
        acc = self.acc(out, batch.y)
        self.log("test_loss", loss, prog_bar=True)
        self.log("test_acc", acc, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001, weight_decay=5e-4)
        return optimizer
