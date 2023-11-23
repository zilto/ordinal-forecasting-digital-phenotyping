from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from hamilton import driver, base
from hamilton.io.materialization import to
from hamilton.plugins import xgboost_extensions
import hydra
from hydra.core.config_store import ConfigStore


def base_hydra_config() -> dict:
    return dict(
        job=dict(chdir=True),
        run=dict(dir="./outputs/tabular/${offset}/${label}/${now:%Y-%m-%d_%H-%M-%S}"),
        sweep=dict(
            dir="./multirun/${now:%Y-%m-%d_%H-%M-%S}/tabular",
            subdir="${task}/${offset}/${label}/",
        )
    )


@dataclass
class TrainTabularConfig:
    hydra: dict = field(default_factory=base_hydra_config)
    data_dir: str = "/home/tjean/projects/masters/data/preprocessed"
    mode: str = "evaluation"
    task: str = "ordinal_regression"
    label: str = "ema_CALM"
    offset: int = 0
    n_trials: int = 10
    n_folds: int = 2


cs = ConfigStore.instance()
cs.store(name="train_tabular", node=TrainTabularConfig)


@hydra.main(version_base=None, config_name="train_tabular")
def main(cfg):
    from src import tabular_model, model_evaluation

    config = dict(
        task=cfg.task,
        mode=cfg.mode,
    )

    dr = (
        driver.Builder()
        .with_modules(tabular_model, model_evaluation)
        .with_config(config)
        .with_adapter(base.DefaultAdapter())
        .build()
    )

    inputs = dict(
        test_alpha=0.05,
        data_dir=cfg.data_dir,
        offset=cfg.offset,
        label=cfg.label,
        n_trials=cfg.n_trials,
        n_folds=cfg.n_folds,
    )

    offset_data_dir = f"{cfg.data_dir}/offset_{cfg.offset}"
    # try to read cache
    if config["mode"] == "development":
        if not all((
            Path(f"{offset_data_dir}/X_train.npy").exists(),
            Path(f"{offset_data_dir}/X_validation.npy").exists(),
        )):
            to_cache = dr.execute(final_vars=["X_train", "X_eval", "y_eval"], inputs=inputs)
            np.save(f"{offset_data_dir}/X_train.npy", to_cache["X_train"])
            np.save(f"{offset_data_dir}/X_validation.npy", to_cache["X_eval"])
    else:
        if not all((
            Path(f"{offset_data_dir}/X_train_macro.npy").exists(),
            Path(f"{offset_data_dir}/X_test.npy").exists(),
        )):
            to_cache = dr.execute(final_vars=["X_train", "X_eval", "y_eval"], inputs=inputs)
            np.save(f"{offset_data_dir}/X_train_macro.npy", to_cache["X_train"])
            np.save(f"{offset_data_dir}/X_test.npy", to_cache["X_eval"])

    # load cached data       
    if cfg.mode == "development":
        overrides = dict(
            X_train=np.load(f"{offset_data_dir}/X_train.npy"),
            X_eval=np.load(f"{offset_data_dir}/X_validation.npy"),
        )
    else:
        overrides = dict(
            X_train=np.load(f"{offset_data_dir}/X_train_macro.npy"),
            X_eval=np.load(f"{offset_data_dir}/X_test.npy"),
        )

    materializers = [
        to.json(
            id="best_model_json", dependencies=["best_model"], path="./xgboost_model.json"
        ),
        to.json(
            id="eval_stats_json",
            dependencies=["performance_metrics"],
            path="./eval_stats.json"
        ),
        to.parquet(
            id="bootstrap_scores_parquet",
            dependencies=["bootstrap_scores"],
            combine=base.PandasDataFrameResult(),
            path="./bootstrap_scores.parquet",
        ),
        to.parquet(
            id="optuna_study_parquet", dependencies=["study_results"], path="./optuna_study.parquet"
        ),
        to.parquet(
            id="y_pred_eval_parquet",
            dependencies=["y_pred_eval", "y_eval"],
            combine=base.PandasDataFrameResult(),
            path="./y_pred_eval.parquet",
        ),
    ]

    dr.materialize(
        *materializers,
        inputs=inputs,
        overrides=overrides,
    )

    dr.visualize_materialization(
        *materializers,
        inputs=inputs,
        overrides=overrides,
        output_file_path="./dag",
        render_kwargs={"format": "png", "view": False},
    )


if __name__ == "__main__":
    main()
