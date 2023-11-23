from typing import Optional

import torch
import torch.nn.functional as F
from torchmetrics import MeanAbsoluteError, MetricCollection, Accuracy
from torchmetrics.classification import MulticlassAccuracy
import pytorch_lightning as pl
from coral_pytorch.losses import corn_loss
from coral_pytorch.dataset import corn_label_from_logits


class PyTorchRNN(torch.nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        num_layers,
        dropout,
        num_classes
    ) -> None:
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.num_classes = num_classes

        self.recurrent_block = torch.nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, dropout=dropout, batch_first=True)
        if self.num_classes > 1:
            self.output_layer = torch.nn.Linear(hidden_dim, num_classes - 1)  # CORN output layer uses num_classes-1
        else:
            self.output_layer = torch.nn.Linear(hidden_dim, num_classes)  # 1: regression

    def forward(self, X, hidden_states: Optional[tuple] = None):
        if hidden_states is None:
            output, (hidden_layer, hidden_cell) = self.recurrent_block(X)
        else:
            output, (hidden_layer, hidden_cell) = self.recurrent_block(X, hidden_states)

        assert torch.equal(output[:,-1,:], hidden_layer[-1])

        linear_out = self.output_layer(hidden_layer[-1])  # output dim: (sequence len, num_classes-1)
        if self.num_classes in (1, 2):
            logits = linear_out.squeeze()
        elif self.num_classes > 2:
            logits = linear_out.view(-1, (self.num_classes-1))  # logits dim: (sequence len, num_classes-1)

        return logits, (hidden_layer, hidden_cell)


class LightningCORN(pl.LightningModule):
    def __init__(self, model, learning_rate: float) -> None:
        super().__init__()

        self.learning_rate = learning_rate
        self.model = model

        self.input_dim = model.input_dim
        self.hidden_dim = model.hidden_dim
        self.num_classes = model.num_classes

        self.save_hyperparameters(ignore=["model"])

        metrics = MetricCollection([
            MeanAbsoluteError(),
        ])

        self.train_metrics = metrics.clone(prefix="train_")
        self.valid_metrics = metrics.clone(prefix="valid_")

    def forward(self, X, hidden_states: Optional[tuple] = None):
        return self.model(X, hidden_states)

    def step(self, batch: tuple, stage: str = "fit"):
        x, y = batch
        logits, _ = self.forward(x)
        return y, logits

    def predict_step(self, batch, batch_idx):
        y, logits = self.step(batch)
        y_pred = corn_label_from_logits(logits)
        return y_pred, logits

    def training_step(self, batch, batch_idx):
        y, logits = self.step(batch)
        loss = corn_loss(logits, y, num_classes=self.num_classes)
        y_pred = corn_label_from_logits(logits)

        batch_size = y.shape[0]

        self.log("train_loss", loss, batch_size=batch_size, on_epoch=True)
        metrics = self.train_metrics(preds=y_pred, target=y, logits=logits)
        self.log_dict(metrics, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        y, logits = self.step(batch)
        loss = corn_loss(logits, y, num_classes=self.num_classes)
        y_pred = corn_label_from_logits(logits)

        batch_size = y.shape[0]

        self.log("valid_loss", loss, batch_size=batch_size, on_epoch=True)
        metrics = self.valid_metrics(preds=y_pred, target=y, logits=logits)
        self.log_dict(metrics, on_epoch=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer


class LightningBinaryClassifier(pl.LightningModule):
    def __init__(self, model, learning_rate: float) -> None:
        super().__init__()

        self.learning_rate = learning_rate
        self.model = model

        self.input_dim = model.input_dim
        self.hidden_dim = model.hidden_dim
        self.num_classes = model.num_classes

        self.save_hyperparameters(ignore=["model"])

        metrics = MetricCollection([
            Accuracy(task="binary", num_classes=2),
            MulticlassAccuracy(num_classes=2, top_k=1, average="weighted"),
        ])

        self.train_metrics = metrics.clone(prefix="train_")
        self.valid_metrics = metrics.clone(prefix="valid_")

    def forward(self, X, hidden_states: Optional[tuple] = None):
        return self.model(X, hidden_states)

    def step(self, batch: tuple, stage: str = "fit"):
        x, y = batch
        logits, _ = self.forward(x)
        return y, logits

    def predict_step(self, batch, batch_idx):
        _, logits = self.step(batch)
        y_pred = torch.sigmoid(logits) > 0.5
        return y_pred, logits

    def training_step(self, batch, batch_idx):
        y, logits = self.step(batch)
        loss = F.binary_cross_entropy_with_logits(logits, y.to(torch.float))
        y_pred = torch.sigmoid(logits)

        self.log("train_loss", loss, batch_size=y.shape[0], on_epoch=True)
        metrics = self.train_metrics(preds=y_pred, target=y, logits=logits)
        self.log_dict(metrics, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        y, logits = self.step(batch)
        loss = F.binary_cross_entropy_with_logits(logits, y.to(torch.float))
        y_pred = torch.sigmoid(logits)

        self.log("valid_loss", loss, batch_size=y.shape[0], on_epoch=True)
        metrics = self.valid_metrics(preds=y_pred, target=y, logits=logits)
        self.log_dict(metrics, on_epoch=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer
