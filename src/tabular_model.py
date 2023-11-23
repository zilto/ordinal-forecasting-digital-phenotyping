from typing import Optional, Callable

import pandas as pd
import numpy as np
import optuna
from optuna.distributions import IntDistribution, FloatDistribution
from sklearn.model_selection import StratifiedKFold
import xgboost

from hamilton.function_modifiers import (
    config,
    parameterize_sources,
    extract_fields,
)

from src.xgboost_ordinal import OptimizedRounder


OPTUNA_DISTRIBUTIONS = dict(
    n_estimators=IntDistribution(low=250, high=700, step=150),
    learning_rate=FloatDistribution(low=0.01, high=0.2, log=True),
    max_depth=IntDistribution(low=3, high=10),
    gamma=FloatDistribution(low=0.01, high=20, log=True),
    colsample_bytree=FloatDistribution(low=0.06, high=1),
    min_child_weight=IntDistribution(low=1, high=20, log=True),
    max_delta_step=IntDistribution(low=0, high=10),
)


TABULAR_FEATURES = [
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
  "month",
  "day_of_week",
  "day_of_month",
  "is_weekend",
]


def _drop_splines(df):
    df = df.loc[:, [col for col in df.columns if not col.startswith(("day_", "month_"))]]
    df["month"] = df.timestamp.dt.month
    df["day_of_month"] = df.timestamp.dt.day
    df["day_of_week"] = df.timestamp.dt.dayofweek
    return df


def _binarize_labels(y: np.ndarray, label: str) -> np.ndarray:
    if label in ["ema_CALM", "ema_HOPEFUL", "ema_SLEEPING", "ema_THINK"]:
        return np.where(y>=3, 1, 0)
    elif label in ["ema_DEPRESSED", "ema_HARM", "ema_SEEING_THINGS", "ema_STRESSED", "ema_VOICES"]:
        return np.where(y>=1, 1, 0)
    elif label == "ema_SOCIAL":
        return np.where(y>=2, 1, 0)
    else:
        raise ValueError(f"Invalid `label`. Received {label}")


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
        train_data=_drop_splines(train_sequence_df),
        eval_data=_drop_splines(validation_sequence_df)
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
        train_data=_drop_splines(train_macro_sequence_df),
        eval_data=_drop_splines(test_sequence_df)
    )


@parameterize_sources(
    X_train=dict(data="train_data"),
    X_eval=dict(data="eval_data"),
)
def X_prep(data: pd.DataFrame, features: list[str] = TABULAR_FEATURES) -> np.ndarray:
    Xs = []
    for _, group in data.groupby("seq_idx"):
        X = pd.pivot_table(
            group[features + ["seq_idx", "user_id", "epoch", "timestamp"]],
            index=["seq_idx", "user_id"],
            columns=["epoch", "timestamp"]
        ).to_numpy()
        Xs.append(X)

    return np.concatenate(Xs, axis=0)


@parameterize_sources(
    y_train_prep=dict(data="train_data", label="label"),
    y_eval_prep=dict(data="eval_data", label="label"),
)
def y_prep(data: pd.DataFrame, label: str) -> np.ndarray:
    ys = []
    for _, group in data.groupby("seq_idx"):
        y = group[label].iloc[-1]
        ys.append(y)

    return np.asarray(ys)


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


def model_config(model_config_override: Optional[dict] = None) -> dict:
    """define the model config for an XGBoost regressor; allows for overrides to be passed"""
    config = dict(
        booster="gbtree",
        learning_rate=0.05,  # alias: eta; typical 0.01 to 0.2
        max_depth=3,  # typical 3 to 10; will lead to overfitting
        gamma=0.1,  # alias: min_split_loss; 0 to +inf
        n_estimators=200,
        colsample_bytree=1,  # typical 0.5 to 1
        subsample=1,  # typical 0.6 to 1
        min_child_weight=1,  # 0 to +inf; prevent overfitting; too high underfit
        max_delta_step=0,  # 0 is no constraint; used in imbalanced logistic reg; typical 1 to 10;
        reg_alpha=0,  # alias alpha; default 0
        reg_lambda=1,  # alias lambda; default 1
        tree_method="gpu_hist",
        enable_categorical=True,
        max_cat_to_onehot=None,
        # learning parameters
        objective="reg:squarederror",
        seed=0,
        # others
        verbosity=2,  # 0: silent, 1: warning, 2: info, 3: debug
        callbacks=None,
    )
    if model_config_override:
        config.update(**model_config_override)
    return config


@config.when(task="binary_classification")
def predictor__binary() -> Callable:
    def _predictor(model: xgboost.XGBModel, X: np.ndarray, **kwargs) -> np.ndarray:
        return model.predict(X)
        
    return _predictor


@config.when(task="ordinal_regression")
def predictor__ordinal() -> Callable:
    def _predictor(
        model: xgboost.XGBModel,
        X: np.ndarray,
        y: np.ndarray,
        **kwargs
    ) -> np.ndarray:
        continuous_pred = model.predict(X)
        rounder = OptimizedRounder()
        rounder.fit(continuous_pred, y)
        return rounder.predict(continuous_pred, rounder.coefficients)
        
    return _predictor


def study(higher_is_better: bool) -> optuna.study.Study:
    if higher_is_better:
        return optuna.create_study(direction="maximize")
    else:
        return optuna.create_study(direction="minimize")


@config.when(task="binary_classification")
def base_model__binary(model_config: dict) -> xgboost.XGBModel:
    return xgboost.XGBClassifier(**model_config)


@config.when(task="ordinal_regression")
def base_model__ordinal(model_config: dict) -> xgboost.XGBModel:
    model_config.update(objective="reg:absoluteerror")
    return xgboost.XGBRegressor(**model_config)


@extract_fields(dict(
    study_results=pd.DataFrame,
    best_hyperparameters=dict,
))
def hyperparameter_search(
    X_train: np.ndarray,
    y_train: np.ndarray,
    base_model: xgboost.XGBModel,
    predictor: Callable,
    scorer: Callable,
    study: optuna.study.Study,
    n_trials: int,
    n_folds: int = 5,
    optuna_distributions: dict = OPTUNA_DISTRIBUTIONS,
) -> dict:
    model = base_model

    for _ in range(n_trials):
        trial = study.ask(optuna_distributions)
        model.set_params(**trial.params)

        fold_scores = []
        folds = StratifiedKFold(n_splits=n_folds, shuffle=True)
        for train_idx, val_idx in folds.split(X_train, y_train):
            X_train_fold, y_train_fold = X_train[train_idx], y_train[train_idx]
            X_val_fold, y_val_fold = X_train[val_idx], y_train[val_idx]

            model.fit(
                X_train_fold, y_train_fold,
                eval_set=[(X_val_fold, y_val_fold)],
                verbose=False,
            )
            y_val_pred = predictor(model=model, X=X_val_fold, y=y_val_fold)
            fold_score = scorer(y_true=y_val_fold, y_pred=y_val_pred)

            fold_scores.append(fold_score)
        
        mean_fold_score = np.mean(fold_scores)
        study.tell(trial, mean_fold_score)  # type: ignore

    return dict(
        study_results=study.trials_dataframe(),
        best_hyperparameters=study.best_params,
    )


def best_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    base_model: xgboost.XGBModel,
    best_hyperparameters: dict,
) -> xgboost.XGBModel:
    model = base_model
    model = model.set_params(
        early_stopping_rounds=None,
        **best_hyperparameters,
    )
    model.fit(X_train, y_train)

    return model


@parameterize_sources(
    y_pred_train=dict(X="X_train", y="y_train", best_model="best_model"),
    y_pred_eval=dict(X="X_eval", y="y_eval", best_model="best_model"),
)
def best_pred(
    X: np.ndarray,
    y: np.ndarray,
    best_model: xgboost.XGBModel,
    predictor: Callable,
) -> np.ndarray:
    return predictor(model=best_model, X=X, y=y)
