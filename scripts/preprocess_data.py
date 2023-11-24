from dataclasses import dataclass, field

import pandas as pd
from hamilton import driver
from hamilton.io.materialization import to
import hydra
from hydra.core.config_store import ConfigStore


pd.options.mode.chained_assignment = None


def base_hydra_config() -> dict:
    return dict(
        job=dict(chdir=True),
        run=dict(dir="./outputs/${now:%Y-%m-%d_%H-%M}/preprocess_data/offset_${offset}"),
        sweep=dict(
            dir="./multirun/${now:%Y-%m-%d_%H-%M}/preprocess_data",
            subdir="offset_${offset}/",
        )
    )


@dataclass
class PreprocessDataConfig:
    hydra: dict = field(default_factory=base_hydra_config)
    raw_data_path: str = "/home/tjean/projects/masters/src/data/raw/CrossCheck_Daily_Data.csv"
    data_dir: str = "/home/tjean/projects/masters/data/preprocessed"
    offset: int = 0
    minimum_n_ema: int = 21
    dataset: str = "crosscheck"


cs = ConfigStore.instance()
cs.store(name="preprocess_data", node=PreprocessDataConfig)


@hydra.main(version_base=None, config_name="preprocess_data")
def main(cfg):
    from src import crosscheck, preprocessing

    dr = (
        driver.Builder()
        .enable_dynamic_execution(allow_experimental_mode=True)
        .with_modules(crosscheck, preprocessing)
        .build()
    )

    inputs = dict(
        raw_data_path=cfg.raw_data_path,
        offset=cfg.offset,
        minimum_n_ema=cfg.minimum_n_ema,
    )
    
    offset_dir = f"{cfg.data_dir}/offset_{cfg.offset}"
    materializers = [
        to.parquet(
            path=f"{offset_dir}/train_df.parquet",
            id="train_parquet",
            dependencies=["train_scaled_df"],
        ),
        to.parquet(
            path=f"{offset_dir}/validation_df.parquet",
            id="validation_parquet",
            dependencies=["validation_scaled_df"],
        ),
        to.parquet(
            path=f"{offset_dir}/train_macro_df.parquet",
            id="train_macro_parquet",
            dependencies=["train_macro_scaled_df"],
        ),
        to.parquet(
            path=f"{offset_dir}/test_df.parquet",
            id="test_parquet",
            dependencies=["test_scaled_df"],
        ),
    ]
    
    dr.materialize(
        *materializers,
        inputs=inputs,
    )


if __name__ == "__main__":
    main()
