from typing import Optional

from hamilton.function_modifiers import config, parameterize_sources, extract_fields
import pandas as pd
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers.wandb import WandbLogger
import torch
from torch.utils.data import Dataset, DataLoader

from src.ordinal_lstm import PyTorchRNN, LightningBinaryClassifier, LightningCORN
from src.tabular_model import _binarize_labels


torch.set_float32_matmul_precision("medium")
pl.seed_everything(seed=0, workers=True)


SEQUENCE_FEATURES = [
  "act_in_vehicle",
  "act_on_bike",
  "act_on_foot",
  "act_running",
  "act_still",
  "act_tilting",
  "act_unknown",
  "act_walking",
  "audio_convo_duration",
  "audio_convo_num",
  "audio_voice",
  "call_in_duration",
  "call_in_num",
  "call_miss_num",
  "call_out_duration",
  "call_out_num",
  "loc_dist",
  "loc_visit_num",
  "sleep_duration",
  "sleep_end",
  "sleep_start",
  "sms_in_num",
  "sms_out_num",
  "unlock_duration",
  "unlock_num",
  "month_sp_0",
  "month_sp_1",
  "month_sp_2",
  "month_sp_3",
  "month_sp_4",
  "month_sp_5",
  "day_of_month_sp_0",
  "day_of_month_sp_1",
  "day_of_month_sp_2",
  "day_of_month_sp_3",
  "day_of_month_sp_4",
  "day_of_month_sp_5",
  "day_of_month_sp_6",
  "day_of_month_sp_7",
  "day_of_month_sp_8",
  "day_of_month_sp_9",
  "day_of_week_sp_0",
  "day_of_week_sp_1",
  "day_of_week_sp_2",
  "is_weekend",
]


@extract_fields(dict(
    train_data=pd.DataFrame,
    eval_data=pd.DataFrame
))
@config.when(mode="development")
def load_train_eval__development(data_dir: str, offset: int) -> dict[str, pd.DataFrame]:
    offset_data_dir = f"{data_dir}/offset_{offset}"

    train_sequence_df = pd.read_parquet(f"{offset_data_dir}/train_sequence_df.parquet")
    validation_sequence_df = pd.read_parquet(f"{offset_data_dir}/validation_sequence_df.parquet")

    return dict(
        train_data=train_sequence_df,
        eval_data=validation_sequence_df
    )


@extract_fields(dict(
    train_data=pd.DataFrame,
    eval_data=pd.DataFrame
))
@config.when(mode="evaluation")
def load_train_eval__evaluation(data_dir: str, offset: int) -> dict[str, pd.DataFrame]:
    offset_data_dir = f"{data_dir}/offset_{offset}"

    train_macro_sequence_df = pd.read_parquet(f"{offset_data_dir}/train_macro_sequence_df.parquet")
    test_sequence_df = pd.read_parquet(f"{offset_data_dir}/test_sequence_df.parquet")

    return dict(
        train_data=train_macro_sequence_df,
        eval_data=test_sequence_df
    )


@parameterize_sources(
    X_train=dict(data="train_data"),
    X_eval=dict(data="eval_data"),
)
def X_prep(data: pd.DataFrame, features: list[str] = SEQUENCE_FEATURES) -> np.ndarray:
    Xs = []
    for _, sequence in data.groupby("seq_idx"):
        X = sequence[features].to_numpy()
        Xs.append(X)

    return np.asarray(Xs).astype(np.float32)


@parameterize_sources(
    y_train_prep=dict(data="train_data", label="label"),
    y_eval_prep=dict(data="eval_data", label="label"),
)
def y_prep(data: pd.DataFrame, label: str) -> np.ndarray:
    ys = []
    for _, sequence in data.groupby("seq_idx"):
        y = sequence[label].tail(1).to_numpy()
        ys.append(y)

    return np.asarray(ys).astype(np.float32)


@parameterize_sources(
    y_train=dict(y_prep="y_train_prep", label="label"),
    y_eval=dict(y_prep="y_eval_prep", label="label"),
)
@config.when(task="binary_classification")
def y__binary(y_prep: np.ndarray, label: str) -> np.ndarray:
    return _binarize_labels(y_prep, label)


@parameterize_sources(
    y_train=dict(y_prep="y_train_prep"),
    y_eval=dict(y_prep="y_eval_prep"),
)
@config.when(task="ordinal_regression")
def y__ordinal(y_prep: np.ndarray) -> np.ndarray:
    return y_prep


class TimeseriesDataset(Dataset):
    """Convert numpy packed sequence matrix into torch Dataset"""
    def __init__(self, X: np.ndarray, y: np.ndarray):
        """converts input numpy matrix into torch Tensor"""
        self.X = torch.tensor(X).to(torch.float32)
        self.y = torch.tensor(y.reshape(-1)).to(torch.int32)

    def __len__(self) -> int:
        """get the number examples"""
        return len(self.y)

    def __getitem__(self, idx) -> tuple[torch.Tensor, torch.Tensor]:
        """get a single example tuple (features, label)"""
        return self.X[idx], self.y[idx]


class RNNDataModule(pl.LightningDataModule):
    def __init__(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_eval: np.ndarray,
        y_eval: np.ndarray,
        batch_size: int,
        num_workers: int,
    ):
        super().__init__()
        self.X_train, self.y_train = X_train, y_train
        self.X_eval, self.y_eval = X_eval, y_eval
        self.batch_size = batch_size
        self.num_workers = num_workers

    def prepare_data(self):
        pass

    def setup(self, stage = None):
        self.train_dataset = TimeseriesDataset(self.X_train, self.y_train)
        self.val_dataset = TimeseriesDataset(self.X_eval, self.y_eval)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )
    

def pl_data_module(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_eval: np.ndarray,
    y_eval: np.ndarray,
    batch_size: int = 8092,
    num_workers: int = 8,
) -> pl.LightningDataModule:
    return RNNDataModule(
        X_train=X_train,
        y_train=y_train,
        X_eval=X_eval,
        y_eval=y_eval,
        batch_size=batch_size,
        num_workers=num_workers,
    )


@config.when(task="binary_classification")
def pl_callbacks__binary() -> list:
    return [
        ModelCheckpoint(
            filename="crosscheck-binaryclf-{epoch:02d}-{valid_loss:.03f}",
            monitor="valid_loss",
            mode="min",
            save_last=True,
            save_on_train_epoch_end=True,
        ),
    ]


@config.when(task="ordinal_regression")
def pl_callbacks__ordinal() -> list:
    return [
        ModelCheckpoint(
            filename="crosscheck-corn-{epoch:02d}-{valid_MeanAbsoluteError:.03f}",
            monitor="valid_MeanAbsoluteError",
            mode="min",
            save_last=True,
            save_on_train_epoch_end=True,
        ),
    ]


def pl_logger(
    logger_config_override: Optional[dict] = None
) -> WandbLogger:
    config = dict(
        project="masters-src"
    )
    if logger_config_override:
        config.update(**logger_config_override)

    return WandbLogger(**config)


def pl_trainer(
    pl_callbacks: list,
    pl_logger: WandbLogger | bool,
    trainer_config_override: Optional[dict] = None
) -> pl.Trainer:
    return pl.Trainer(
        max_epochs=65,
        accelerator="gpu",
        devices=1,
        auto_scale_batch_size=False,
        auto_lr_find=False,
        gradient_clip_algorithm="norm",
        deterministic=False,
        precision=32,
        logger=pl_logger,
        callbacks=pl_callbacks,
    )


def model_config(
    features: list[str] = SEQUENCE_FEATURES,
    model_config_override: Optional[dict] = None,
) -> dict:
    config = dict(
        input_dim=len(features),
        hidden_dim=128,
        num_layers=2,
        dropout=0.1,
    )
    if model_config_override:
        config.update(**model_config_override)

    return config


@config.when(task="binary_classification")
def base_model__binary(
    model_config: dict,
    learning_rate: float = 0.0015,
) -> pl.LightningModule:
    lstm_module = PyTorchRNN(num_classes=2, **model_config)
    return LightningBinaryClassifier(lstm_module, learning_rate=learning_rate)


@config.when(task="ordinal_regression")
def base_model__ordinal(
    model_config: dict,
    learning_rate: float = 0.0015,
) -> pl.LightningModule:
    lstm_module = PyTorchRNN(num_classes=4, **model_config)
    return LightningCORN(lstm_module, learning_rate=learning_rate)


def best_model(
    base_model: pl.LightningModule,
    pl_data_module: pl.LightningDataModule,
    pl_trainer: pl.Trainer,
) -> pl.Trainer:
    pl_trainer.fit(base_model, pl_data_module)  

    return pl_trainer


@extract_fields(dict(
    y_pred_eval=np.ndarray,
    y_logits_eval=np.ndarray,
))
def best_pred(
    best_model: pl.Trainer,
    pl_data_module: pl.LightningDataModule,
    eval_ckpt: str = "last"
) -> dict[str, np.ndarray]:
    model_out = best_model.predict(
        ckpt_path=eval_ckpt,
        dataloaders=pl_data_module.val_dataloader(),
    )
    return dict(
        y_pred_eval=model_out[0][0].cpu().detach().numpy(),
        y_logits_eval=model_out[0][1].cpu().detach().numpy(),
    )
