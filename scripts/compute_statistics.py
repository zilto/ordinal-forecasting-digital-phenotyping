from dataclasses import dataclass, field

from hamilton import driver
from hamilton.io.materialization import to
import hydra
from hydra.core.config_store import ConfigStore


def base_hydra_config() -> dict:
    return dict(
        job=dict(chdir=True),
        run=dict(dir="./posthoc")
    )


@dataclass
class StatisticsConfig:
    hydra: dict = field(default_factory=base_hydra_config)
    run_name: str = "2023-09-24"
    run_dir: str = "/home/tjean/projects/masters/multirun/${run_name}/"
    data_dir: str = "/home/tjean/projects/masters/data/"


cs = ConfigStore.instance()
cs.store(name="statistics", node=StatisticsConfig)


@hydra.main(version_base=None, config_name="statistics")
def main(cfg):
    from src import statistics

    config = dict(
        correction="bonferroni"
    )

    dr = (
        driver.Builder()
        .enable_dynamic_execution(allow_experimental_mode=True)
        .with_config(config)
        .with_modules(statistics)
        .build()
    )

    inputs = dict(
        run_dir=cfg.run_dir,
        data_dir=cfg.data_dir,
        plot_kwargs=dict(format="png"),
        n_comparisons_correction=120,
    )

    materializers = [
        to.parquet(
            id="eval_stats_parquet",
            dependencies=["eval_stats_df"],
            path=f"{cfg.run_dir}/eval_stats.parquet"
        ),
        to.parquet(
            id="statistical_tests_parquet",
            dependencies=["statistical_tests_df"],
            path=f"{cfg.run_dir}/statistical_tests.parquet"
        ),
        to.json(
            id="statistical_tests_json",
            dependencies=["compare_model_types", "compare_offsets", "compare_labels"],
            path=f"{cfg.run_dir}/statistical_tests.json",
        )
    ]

    dr.materialize(
        *materializers,
        additional_vars=[
            "splits_plot",
            "binary_performance_plot",
            "ordinal_performance_plot",
            "imbalance_performance_plot",
            "binary_ordinal_error_plot",
            "binary_ordinal_residuals",
        ],
        inputs=inputs,
    )


if __name__ == "__main__":
    main()
