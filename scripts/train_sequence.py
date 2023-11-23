from dataclasses import dataclass, field

from hamilton import driver, base
from hamilton.io.materialization import to
import hydra
from hydra.core.config_store import ConfigStore


def base_hydra_config() -> dict:
    return dict(
        job=dict(chdir=True),
        run=dict(dir="./outputs/sequence/${offset}/${label}/${now:%Y-%m-%d_%H-%M-%S}"),
        sweep=dict(
            dir="./multirun/${now:%Y-%m-%d_%H-%M-%S}/sequence",
            subdir="${task}/${offset}/${label}/",
        )
    )


@dataclass
class TrainSequenceConfig:
    hydra: dict = field(default_factory=base_hydra_config)
    data_dir: str = "/home/tjean/projects/masters/data/preprocessed"
    mode: str = "evaluation"
    task: str = "ordinal_regression"
    label: str = "ema_CALM"
    offset: int = 0
    eval_ckpt: str = "test"


cs = ConfigStore.instance()
cs.store(name="train_sequence", node=TrainSequenceConfig)


@hydra.main(version_base=None, config_name="train_sequence")
def main(cfg):
    from src import sequence_model, model_evaluation

    config = dict(
        task=cfg.task,
        mode=cfg.mode,
    )

    dr = (
        driver.Builder()
        .with_modules(sequence_model, model_evaluation)
        .with_config(config)
        .with_adapter(base.DefaultAdapter())
        .build()
    )

    inputs = dict(
        data_dir=cfg.data_dir,
        offset=cfg.offset,
        label=cfg.label,
        eval_ckpt=cfg.eval_ckpt,
    )

    materializers = [
        to.json(
            id="eval_stats_json",
            dependencies=["performance_metrics", "association_metrics"],
            combine=base.DictResult(),
            path="./eval_stats.json"
        ),
        to.parquet(
            id="y_pred_eval_parquet",
            dependencies=["y_eval", "y_pred_eval"],
            combine=base.PandasDataFrameResult(),
            path="./y_pred_eval.parquet",
        ),
    ]

    dr.materialize(
        *materializers,
        inputs=inputs,
    )

    dr.visualize_materialization(
        *materializers,
        inputs=inputs,
        output_file_path="./dag",
        render_kwargs={"format": "png", "view": False},
    )


if __name__ == "__main__":
    main()
