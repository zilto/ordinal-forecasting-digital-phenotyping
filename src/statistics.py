import itertools
import json
from typing import Callable

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as stats
import scikit_posthocs as sp

from hamilton.function_modifiers import parameterize, value, source, group
from hamilton.htypes import Parallelizable, Collect


TASKS = ["ordinal_regression", "binary_classification"]
MODEL_TYPES = ["tabular", "sequence"]
OFFSETS = [0, 1, 7]
LABELS = [
    "ema_CALM", "ema_HOPEFUL", "ema_SLEEPING", "ema_SOCIAL", "ema_THINK",
    "ema_DEPRESSED", "ema_HARM", "ema_SEEING_THINGS", "ema_STRESSED", "ema_VOICES",
]

LABELS_DISPLAY = {
    "ema_CALM": "CALM",  #"Feeling CALM",
    "ema_DEPRESSED": "DEPRESSED",
    "ema_HARM": "HARM", # "Worried about HARM",
    "ema_HOPEFUL": "HOPEFUL", #  "HOPEFUL about future",
    "ema_SEEING_THINGS": "SEEING THINGS",
    "ema_SLEEPING": "SLEEPING",  #"SLEEPING well",
    "ema_SOCIAL": "SOCIAL",
    "ema_STRESSED": "STRESSED", #"Feeling STRESSED",
    "ema_THINK": "THINK", # "THINK CLEARLY",
    "ema_VOICES":  "VOICES",  # "Bothered by VOICES"
}


def model_specs(
    tasks: list[str] = TASKS,
    model_types: list[str] = MODEL_TYPES,
    offsets: list[int] = OFFSETS,
    labels: list[str] = LABELS,
) -> Parallelizable[dict]:
    keys = ["task", "model_type", "offset", "label"]
    for specs in itertools.product(tasks, model_types, offsets, labels):
        yield dict(zip(keys, specs))


def model_dir(run_dir: str, model_specs: dict) -> str:
    task = model_specs['task']
    model_type = model_specs['model_type']
    offset = model_specs['offset']
    label = model_specs['label']
    return f"{run_dir}/{model_type}/{task}/{offset}/{label}"


def _y_df(run_dir: str, model_specs: dict) -> pd.DataFrame:
    task = model_specs['task']
    model_type = model_specs['model_type']
    offset = model_specs['offset']
    label = model_specs['label']

    y_tabular = pd.read_parquet(f"{run_dir}/tabular/{task}/{offset}/{label}/y_pred_eval.parquet")

    if model_type == "sequence":
        y_tabular["y_pred_eval"] = pd.read_parquet(f"{run_dir}/sequence/{task}/{offset}/{label}/y_pred_eval.parquet")["y_pred_eval"]

    return y_tabular.astype(float)


def bootstrap_scores(run_dir: str, model_specs: dict) -> np.ndarray:
    task = model_specs['task']
    offset = model_specs['offset']
    label = model_specs['label']

    bootstrap_dist = pd.read_parquet(f"{run_dir}/tabular/{task}/{offset}/{label}/bootstrap_scores.parquet")
    return bootstrap_dist["bootstrap_scores"].to_numpy()


def eval_stats(model_dir: str, model_specs: dict) -> dict:
    file_name = "eval_stats.json"
    stats = json.load(open(f"{model_dir}/{file_name}"))
    stats.update(model_specs)
    return stats


def eval_stats_collection(eval_stats: Collect[dict]) -> list[dict]:
    return list(eval_stats)


def eval_stats_df(eval_stats_collection: list[dict]) -> pd.DataFrame:
    return pd.DataFrame.from_records(eval_stats_collection)


def _paired_mean_rank(data: pd.DataFrame) -> pd.Series:
    _data = data.copy()
    for i in range(len(_data)):
        _data.iloc[i] = stats.rankdata(_data.iloc[i])
    return _data.mean(axis=0)


def _unpaired_mean_rank(data: pd.DataFrame):
    _data = data.copy()
    ranked = stats.rankdata(_data.to_numpy())
    reshaped = ranked.reshape(_data.shape)
    _data.iloc[:, :] = reshaped
    return _data.mean(axis=0)


def _friedman_test(
    data: pd.DataFrame,
    groupby_col: str,
    metric_col: str,
) -> dict:
    """paired 2+ samples"""
    samples = {
        g_id: g.to_numpy() for g_id, g in 
        data.groupby(groupby_col)[metric_col]
    }
    data = pd.DataFrame.from_dict(samples)
    result, pvalue = stats.friedmanchisquare(*samples.values())
    mean_ranks: pd.Series = _paired_mean_rank(data)
    posthoc_pairwise: pd.DataFrame = sp.posthoc_nemenyi_friedman(data)

    return dict(
        test="friedman",
        result=result,
        pvalue=pvalue,
        mean_ranks=mean_ranks.to_dict(),
        posthoc_pairwise=posthoc_pairwise.to_dict(),
    )


def _kruskal_test(
    data: pd.DataFrame,
    groupby_col: str,
    metric_col: str,
) -> dict:
    """unpaired 2+ samples"""
    samples = {
        g_id: g.to_numpy() for g_id, g in 
        data.groupby(groupby_col)[metric_col]
    }
    data = pd.DataFrame.from_dict(samples)
    result, pvalue = stats.kruskal(*samples.values())
    mean_ranks: pd.Series = _unpaired_mean_rank(data)
    posthoc_pairwise: pd.DataFrame = sp.posthoc_nemenyi(data.T.to_numpy())

    return dict(
        test="kruskall",
        result=result,
        pvalue=pvalue,
        mean_ranks=mean_ranks.to_dict(),
        posthoc_pairwise=posthoc_pairwise.to_dict(),
    )


def _wilcoxon_test(
    data: pd.DataFrame,
    groupby_col: str,
    metric_col: str,
) -> dict:
    """paired 2 samples"""
    samples = {
        g_id: g.to_numpy() for g_id, g in 
        data.groupby(groupby_col)[metric_col]
    }
    data = pd.DataFrame.from_dict(samples)
    x, y = tuple(samples.values())
    result, pvalue = stats.wilcoxon(x, y)
    mean_ranks: pd.Series = _paired_mean_rank(data)

    return dict(
        test="wilcoxon",
        result=result,
        pvalue=pvalue,
        mean_ranks=mean_ranks.to_dict(),
    )


def _mannwhitney_test(
    data: pd.DataFrame,
    groupby_col: str,
    metric_col: str,
) -> dict:
    """unpaired 2 samples"""
    samples = {
        g_id: g.to_numpy() for g_id, g in 
        data.groupby(groupby_col)[metric_col]
    }
    data = pd.DataFrame.from_dict(samples)
    x, y = tuple(samples.values())
    result, pvalue = stats.mannwhitneyu(x, y)
    mean_ranks: pd.Series = _unpaired_mean_rank(data)

    return dict(
        test="mannwhitney",
        result=result,
        pvalue=pvalue,
        mean_ranks=mean_ranks.to_dict(),
    )


@parameterize(
    compare_model_types=dict(
        eval_stats_df=source("eval_stats_df"),
        groupby_col=value("model_type"),
        statistical_tests=group(value(_wilcoxon_test), value(_mannwhitney_test))
    ),
    compare_offsets=dict(
        eval_stats_df=source("eval_stats_df"),
        groupby_col=value("offset"),
        statistical_tests=group(value(_friedman_test), value(_kruskal_test))
    ),
    compare_labels=dict(
        eval_stats_df=source("eval_stats_df"),
        groupby_col=value("label"),
        statistical_tests=group(value(_friedman_test), value(_kruskal_test))
    ),
)
def statistical_tests(
    eval_stats_df: pd.DataFrame,
    groupby_col: str,
    statistical_tests: list[Callable],
    metric_col: str = "performance",
    tasks: list[str] = TASKS,
) -> list:
    results = []
    for task in tasks:
        filtered_df = eval_stats_df.loc[eval_stats_df.task==task]
        for statistical_test in statistical_tests:
            test_results = statistical_test(
                data=filtered_df,
                groupby_col=groupby_col,
                metric_col=metric_col,
            )
            test_results.update(name=statistical_test.__name__, task=task)
            results.append(test_results)        

    return results


def statistical_tests_df(
    compare_model_types: list,
    compare_offsets: list,
    compare_labels: list,
) -> pd.DataFrame:
    records = []
    for comparison in [compare_model_types, compare_offsets, compare_labels]:
        for results in comparison:
            for result in results:
                record = dict(
                    name=result["name"],
                    test=result["test"],
                    result=result["result"],
                    pvalue=result["pvalue"],
                )
                records.append(record)

    return pd.DataFrame.from_records(records)
     

def _plot_bootstrap(ax, vertical_lines: list[tuple], bootstrap_dist: np.ndarray, n_bins: int = 20):   
    hist, edges = np.histogram(bootstrap_dist, bins=n_bins)
    ax.stairs(hist, edges, fill=True, label="bootstrap", color="grey", alpha=0.5)

    ax.tick_params(left=False)

    for idx, (label, value) in enumerate(vertical_lines[::-1]):
        if label == "alpha":
            ax.vlines(value, [0], [hist.max()], colors=["black"], linestyles="dashed", label="\u03B1=0.05/120", alpha=0.5)
        else:
            vcolor = "C0" if idx == 1 else "C3"
            ax.vlines(value, [0], [hist.max()], colors=[f"{vcolor}"], linestyles="dotted", label=label)
        
        justify = "left" if value <0.75 else "right"
        plt.text(value, .2+idx*0.3, np.round(value, 2), ha=justify, transform=ax.get_xaxis_transform())   


def _plot_pred_distplots(
    ax,
    y_eval: np.ndarray,
    y_eval_pred: np.ndarray,
    colors: list,
    class_labels: list[str],
):   
    contingency = pd.crosstab(y_eval_pred, y_eval, rownames=["y_pred"], colnames=["y_true"])
    ax.set(
        ylim=(0+1e-5, 360),
    )
    
    bottom = np.zeros(len(contingency.columns))
    for true_val, heights in contingency.iterrows():
        true_val = int(true_val)
        ax.bar(
            x=heights.index,
            height=heights,
            width=0.7,
            align="center",
            color=[colors[true_val]] * len(heights),
            edgecolor="black",
            label=class_labels[true_val],
            bottom=bottom
        )
    
        bottom += heights

common_performance_plot = dict(run_dir=source("run_dir"), eval_stats_df=source("eval_stats_df"))
@parameterize(
    binary_performance_plot=dict(task=value("binary_classification"), **common_performance_plot),
    ordinal_performance_plot=dict(task=value("ordinal_regression"), **common_performance_plot)
)
def performance_plot(
    run_dir: str,
    eval_stats_df: pd.DataFrame,
    task: str,
    plot_kwargs: dict,
    labels: list[str] = LABELS,
    offsets: list[int] = OFFSETS,
    labels_display: dict = LABELS_DISPLAY,
) -> None:
    # create main frame
    width_in_cm = 19.0 * 2
    height_in_cm = 23.0 * 2

    width_in_inches = width_in_cm / 2.54
    height_in_inches = height_in_cm / 2.54

    fig = plt.figure(figsize=(width_in_inches, height_in_inches))

    offset_labels = ["Same day", "Next day", "Next week"]
    cmap = matplotlib.colormaps["plasma"]

    if task == "binary_classification":
        bootstrap_title = "BAcc"
        bootstrap_range = (0.30, 0.9)
        class_labels = ["Lower", "Higher"]
        colors = [cmap(0.1), cmap(0.9)]
        legend_anchor = (0.5, -1.3)

    else:
        bootstrap_title = "MAMAE"
        bootstrap_range = (0.4, 1.7)
        class_labels = ["Not at all", "A little", "Moderately", "Extremely"]
        colors = [cmap(0.1), cmap(0.4), cmap(0.6), cmap(0.9)]
        legend_anchor = (0.5, -1.3)

    # create 10 labels * 3 offsets grid
    outer_grid = plt.GridSpec(10, 3, wspace=0.25, figure=fig)
    for row in range(outer_grid.nrows):
        for col in range(outer_grid.ncols):
            outer_ax = fig.add_subplot(outer_grid[row, col])
            outer_ax.axis(False)
            
            # create an inner 1 * 3 grid (xgboost, lstm, bootstrap)
            inner_grid = outer_grid[row, col].subgridspec(1, 3, wspace=0.05)

            # query the relevant runs
            label = labels[row]
            offset = offsets[col]

            tabular_entry = eval_stats_df.loc[(
                (eval_stats_df.model_type=="tabular")
                & (eval_stats_df.task==task)
                & (eval_stats_df.label==label)
                & (eval_stats_df.offset==offset)
            )]

            sequence_entry = eval_stats_df.loc[(
                (eval_stats_df.model_type=="sequence")
                & (eval_stats_df.task==task)
                & (eval_stats_df.label==label)
                & (eval_stats_df.offset==offset)
            )]


            # (xgboost, ..., ...)
            tabular_ax = fig.add_subplot(inner_grid[0])
            _y_df = _y_df(run_dir=run_dir, model_specs=dict(model_type="tabular", task=task, offset=offset, label=label))
            _plot_pred_distplots(
                tabular_ax,
                y_eval=_y_df["y_eval"].to_numpy(),
                y_eval_pred=_y_df["y_pred_eval"].to_numpy(),
                class_labels=class_labels,
                colors=colors,
            )
            
            is_sig = tabular_entry["performance_is_significant"].values[0]
            annot = f"{'n.s.' if not is_sig else ''}"
            tabular_ax.text(0.9, 0.9, annot, ha="right", va='top', transform=tabular_ax.transAxes)

            # (..., lstm, ...)
            sequence_ax = fig.add_subplot(inner_grid[1], sharey=tabular_ax)
            _y_df = _y_df(run_dir=run_dir, model_specs=dict(model_type="sequence", task=task, offset=offset, label=label))
            _plot_pred_distplots(
                sequence_ax,
                y_eval=_y_df["y_eval"].to_numpy(),
                y_eval_pred=_y_df["y_pred_eval"].to_numpy(),
                class_labels=class_labels,
                colors=colors,
            )
            sequence_ax.yaxis.set_visible(False)

            is_sig = sequence_entry["performance_is_significant"].values[0]
            annot = f"{'n.s.' if not is_sig else ''}"
            sequence_ax.text(0.9, 0.9, annot, ha="right", va='top', transform=sequence_ax.transAxes)

            # (..., ..., bootstrap)
            bootstrap_ax = fig.add_subplot(inner_grid[2])
            _plot_bootstrap(
                bootstrap_ax,
                vertical_lines=[
                    ("XGBoost", tabular_entry["performance_result"].values[0]),
                    ("LSTM", sequence_entry["performance_result"].values[0]),
                    ("alpha", tabular_entry["performance_confidence_criterion"].values[0])
                ],
                bootstrap_dist=pd.read_parquet(f"{run_dir}/tabular/{task}/{offset}/{label}/bootstrap_scores.parquet").to_numpy(),
            )
            bootstrap_ax.set(yticklabels=[], xlim=bootstrap_range)


            # outer left col
            # if col == 0:
            tabular_ax.set_ylabel(labels_display[label])
                
            # outer top row
            if row == 0:
                outer_ax.set_title(offset_labels[col], pad=30)
                tabular_ax.set_title("XGBoost")
                sequence_ax.set_title("LSTM")
                bootstrap_ax.set_title(bootstrap_title)

            # except outer bottom
            if row != 9:
                tabular_ax.tick_params(axis='x', which='both',
                    bottom=False, top=False, labelbottom=False)
                sequence_ax.tick_params(axis='x', which='both',
                    bottom=False, top=False, labelbottom=False)
                bootstrap_ax.xaxis.set_visible(False)

            # outer bottom
            if row == 9:
                tabular_ax.set_xticks(np.arange(len(class_labels)), class_labels)
                plt.setp(tabular_ax.get_xticklabels(), rotation=70, ha="right", rotation_mode="anchor")

                sequence_ax.set_xticks(np.arange(len(class_labels)), class_labels)
                plt.setp(sequence_ax.get_xticklabels(), rotation=70, ha="right", rotation_mode="anchor")

                sequence_ax.tick_params(axis='x', which='both',
                    bottom=False, top=False, labelbottom=False)
                sequence_ax.legend(
                    handles=[matplotlib.patches.Patch(color=c, label=l)
                            for l, c in zip(class_labels, colors)],
                    loc="lower center",
                    bbox_to_anchor=legend_anchor,
                    title="Predicted"
                )

                bootstrap_ax.legend(loc="lower center", bbox_to_anchor=legend_anchor)

    fig.savefig(f"{task}_performance.{plot_kwargs.get('format', 'png')}", **plot_kwargs)

    # return fig


def label_distributions(data_dir: str) -> pd.DataFrame:
    def _binary_agg(label: str, group_df: pd.DataFrame) -> tuple:
        if label in ["ema_CALM", "ema_HOPEFUL", "ema_SLEEPING", "ema_THINK"]:
            lower = group_df[["0", "1", "2"]].sum(axis=1)
            higher = group_df["3"]
        elif label in ["ema_DEPRESSED", "ema_HARM", "ema_SEEING_THINGS", "ema_STRESSED", "ema_VOICES"]:
            lower = group_df["0"]
            higher = group_df[["1", "2", "3"]].sum(axis=1)
        elif label == "ema_SOCIAL":
            lower = group_df[["0", "1"]].sum(axis=1)
            higher = group_df[["2", "3"]].sum(axis=1)
        else:
            raise ValueError(f"Invalid `label`. Received {label}")
        
        return lower, higher

    records = list()
    for offset in [0, 1, 7]:
        offset_dir = f"{data_dir}/preprocessed/offset_{offset}"
        for split in ["train", "validation", "test"]:
            df = pd.read_parquet(f"{offset_dir}/{split}_sequence_df.parquet")
            for ema in LABELS:
                vals = df[[ema]].value_counts().to_dict()
                records.append({
                    "offset": offset, "split": split, "label": ema,
                    0: vals.get((0, ), 0),
                    1: vals.get((1, ), 0),
                    2: vals.get((2, ), 0),
                    3: vals.get((3, ), 0),
                })
    
    ordinal_splits = pd.DataFrame(records)
    for g_id, g in ordinal_splits.groupby("label"):
        lower, higher = _binary_agg(g_id, g)
        ordinal_splits.loc[ordinal_splits.label==g_id, "lower"] = lower
        ordinal_splits.loc[ordinal_splits.label==g_id, "higher"] = higher

    ordinal_splits[["lower", "higher"]] = ordinal_splits[["lower", "higher"]].astype(int)
    return ordinal_splits


def _plot_splits_dist(ax, splits_df, label, offset, task) -> None:
    mask = (splits_df.label == label) & (splits_df.offset == offset)
    for split_idx, split in enumerate(["train", "validation", "test"]):
        row = splits_df.loc[mask & (splits_df.split == split)]

        if task == "binary_classification":
            u = np.array([0, 1])
            c = row[["lower", "higher"]].to_numpy()[0]
        else:
            u = np.array([0, 1, 2, 3])
            c = row[["0", "1", "2", "3"]].to_numpy()[0]

        bar_width = 0.28
        x_shift = -bar_width + bar_width*split_idx
        cmap = matplotlib.colormaps["viridis"]
        colors = [cmap(0.15), cmap(0.5), cmap(0.85)]

        ax.bar(
            u+x_shift,
            height=c/c.sum() * 100,
            width=bar_width,
            label=["Train", "Validation", "Test"][split_idx],
            color=colors[split_idx],
            edgecolor="black",
        )


def splits_plot(
    label_distributions: pd.DataFrame,
    plot_kwargs: dict,
    labels: list[str] = LABELS,
    offsets: list[int] = OFFSETS,
    labels_display: dict = LABELS_DISPLAY,
) -> None:
    width_in_cm = 19.0 * 2
    height_in_cm = 24.0 * 2

    width_in_inches = width_in_cm / 2.54
    height_in_inches = height_in_cm / 2.54

    fig = plt.figure(figsize=(width_in_inches, height_in_inches))
    outer_grid = plt.GridSpec(1, 2, width_ratios=[2, 1], figure=fig)
    for outer_col in range(outer_grid.ncols):
        grid = outer_grid[0, outer_col].subgridspec(10, 3, wspace=0.05)
            
        if outer_col == 0:
            binary = False
            task = "ordinal_regression"
        else:
            binary = True
            task = "binary_classification"

        for row in range(grid.nrows):
            label = labels[row]
            for col in range(grid.ncols):
                offset = offsets[col]
                ax = fig.add_subplot(grid[row, col])
                _plot_splits_dist(ax, label_distributions, label, offset, task)
                
                ax.set(ylim=(0+1e-10, 85))

                # special conditions for outer frame
                # left
                if col == 0:
                    display_ylabel = labels_display[label]
                    ax.set_ylabel(display_ylabel)
                else:
                    ax.set(yticklabels=[])
                    
                # # top
                if row == 0:
                    offset_labels = ["Same day", "Next day", "Next week"]
                    ax.set_title(offset_labels[col], pad=30)

                # # except bottom
                if row != 9:
                    ax.tick_params(axis='x', which='both',
                        bottom=False, top=False, labelbottom=False)

                # bottom
                if row == 9:
                    if binary:
                        class_labels = np.array(["Lower", "Higher"])
                        legend_anchor = (0.5, -1)
                        ax.set_xticks(np.arange(len(class_labels)), class_labels)
                        plt.setp(ax.get_xticklabels(), rotation=30, ha="right", rotation_mode="anchor")
                    else:
                        class_labels = np.array(["Not at all", "A little", "Moderately", "Extremely"])
                        legend_anchor = (0.5, -1)
                        ax.set_xticks(np.arange(len(class_labels)), class_labels)
                        plt.setp(ax.get_xticklabels(), rotation=30, ha="right", rotation_mode="anchor")

                if (row == 9) and (col == 1):
                    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -1.15))

    fig.savefig(f"splits_dist.{plot_kwargs.get('format', 'png')}", **plot_kwargs)


def scaled_performance_df(eval_stats_df: pd.DataFrame) -> pd.DataFrame:
    df = eval_stats_df.copy()
    df["scaled_error"] = 0
    df.loc[df.task=="binary_classification", "scaled_error"] = 1 - df.loc[df.task=="binary_classification", "performance_result"]
    df.loc[df.task!="binary_classification", "scaled_error"] = df.loc[df.task!="binary_classification", "performance_result"] / 3
    return df


def binary_ordinal_residuals(scaled_performance_df: pd.DataFrame) -> pd.DataFrame:
    x = scaled_performance_df.loc[scaled_performance_df.task=="binary_classification", "scaled_error"]
    y = scaled_performance_df.loc[scaled_performance_df.task=="ordinal_regression", "scaled_error"]
    
    diagonal = np.linspace(0, 1, 2)
    perfect_predictions = np.polyval(np.polyfit(diagonal, diagonal, 1), x)

    df = scaled_performance_df.loc[scaled_performance_df.task=="binary_classification", ["model_type", "offset", "label", "scaled_error"]].copy()
    df["residuals"] = y.to_numpy() - perfect_predictions
    df["abs_residuals"] = np.abs(df["residuals"])
    df["ranked_residuals"] = stats.rankdata(df["abs_residuals"])
    return df


def binary_ordinal_error_plot(
    scaled_performance_df: pd.DataFrame,
    plot_kwargs: dict,
    labels: list[str] = LABELS,
    labels_display: dict = LABELS_DISPLAY,
) -> None:
    cmap1 = matplotlib.colormaps["viridis"]
    cmap1 = matplotlib.colors.ListedColormap(cmap1(np.linspace(0.2, 0.9, 256)))
    cmap2 = matplotlib.colormaps["plasma_r"]
    colors1 = [cmap1(v) for v in np.linspace(0, 1, 5)]
    colors2 = [cmap2(v) for v in np.linspace(0, 1, 5)]
    colors = colors1 + colors2

    colors[5] = "#fcce25"  # overwrite Depressed
    colors[8] = "#7F2447"  # overwrite Stressed
    colors[9] = "#590278"  # overwrite Voices

    x = scaled_performance_df.loc[scaled_performance_df.task=="binary_classification", "scaled_error"] * 100
    y = scaled_performance_df.loc[scaled_performance_df.task=="ordinal_regression", "scaled_error"] * 100
    diagonal = np.linspace(min(min(x), min(y)), max(max(x), max(y)), 2)

    width_in_cm = 9.0 * 2
    height_in_cm = 8.0 * 2

    width_in_inches = width_in_cm / 2.54
    height_in_inches = height_in_cm / 2.54

    fig = plt.figure(figsize=(width_in_inches, height_in_inches))
    plt.scatter(
        x,
        y,
        c=[colors[labels.index(l)] for l in scaled_performance_df.loc[scaled_performance_df.task=="binary_classification", "label"]],
        label=[labels_display[l] for l in scaled_performance_df.loc[scaled_performance_df.task=="binary_classification", "label"]],
        edgecolors="black"
    )
    plt.plot(diagonal, diagonal, linestyle="--", c="grey")

    plt.xlabel("Binary scale error (%)")
    plt.xticks(np.arange(25, 46, 5))
    plt.ylabel("Ordinal scale error (%)")
    plt.yticks(np.arange(25, 46, 5))
    plt.axis("equal")

    legend_handles = [matplotlib.patches.Patch(color=c, label=labels_display[l]) for c, l in zip(colors, labels)]
    plt.legend(handles=legend_handles, bbox_to_anchor=(1.05, 1), loc='upper left') 

    fig.savefig(f"binary_ordinal_performance.{plot_kwargs.get('format', 'png')}", **plot_kwargs)


def _binary_imbalance(df: pd.DataFrame) -> pd.Series:
    """% of majority class - % of minority class"""
    binary_cols = ["lower", "higher"]
    return (df[binary_cols].max(axis=1) - df[binary_cols].min(axis=1)) / df[binary_cols].sum(axis=1)


def _ordinal_imbalance(df: pd.DataFrame) -> pd.Series:
    """% of majority class - % of minority class"""
    ordinal_cols = ["0", "1", "2", "3"]
    return (df[ordinal_cols].max(axis=1) - df[ordinal_cols].min(axis=1)) / df[ordinal_cols].sum(axis=1)
   

def dataset_imbalance(label_distributions: pd.DataFrame) -> pd.DataFrame:
    grouped_df = label_distributions.groupby(["label", "offset"])[["0","1", "2", "3", "lower", "higher"]].sum()

    grouped_df["ordinal_imbalance"] = _ordinal_imbalance(grouped_df)
    grouped_df["binary_imbalance"] = _binary_imbalance(grouped_df)
    grouped_df = grouped_df.reset_index()
    return grouped_df


def _get_imbalance(row, gsplits_df):
    mask = (gsplits_df.offset == row.offset) & (gsplits_df.label == row.label)
    col = "binary_imbalance" if row.task == "binary_classification" else "ordinal_imbalance"
    return gsplits_df.loc[mask, col].values[0]


def _imbalance_plot(ax, filtered_df, labels, colors, labels_display) -> None:
    coefs = np.polyfit(filtered_df.imbalance*100, filtered_df.performance_result, 1)
    poly = np.poly1d(coefs)
    x_line = np.linspace(0, 90, 10)
    y_line = poly(x_line)

    ax.scatter(
        filtered_df.imbalance*100,
        filtered_df.performance_result,
        c=[colors[labels.index(l)] for l in filtered_df.label],
        label=[labels_display[l] for l in filtered_df.label],
        edgecolor="black",
    )
    ax.plot(x_line, y_line, linestyle="--", c="grey")

    ax.set_xlabel("Class imbalance (%)")
    ax.set_xticks(np.arange(0, 1, 0.2)*100) 


def _balanced_unbalanced_plot(ax, eval_df, bad_stats_df, labels, colors, labels_display) -> None:
    coefs = np.polyfit(bad_stats_df.performance_result, eval_df.performance_result, 1)
    poly = np.poly1d(coefs)
    x_line = np.linspace(min(bad_stats_df.performance_result), max(bad_stats_df.performance_result), 10)
    y_line = poly(x_line)

    ax.scatter(
        bad_stats_df.performance_result,
        eval_df.performance_result,
        c=[colors[labels.index(l)] for l in eval_df.label],
        label=[labels_display[l] for l in eval_df.label],
        edgecolor="black",
    )
    ax.plot(x_line, y_line, linestyle="--", c="grey")


def _format_sign(v: float):
    if v < 0:
        sign = "-"
    else:
        sign = ""

    value_no_lead = f"{np.abs(v):.2f}".lstrip("0")
    return f"{sign}{value_no_lead}"


def imbalance_performance_plot(
    eval_stats_df: pd.DataFrame,
    bad_stats_df: pd.DataFrame,
    dataset_imbalance: pd.DataFrame,
    plot_kwargs: dict,
    tasks: list[str] = TASKS,
    labels: list[str] = LABELS,
    labels_display: dict = LABELS_DISPLAY,
) -> None:
    eval_stats_df["imbalance"] = eval_stats_df.apply(lambda row: _get_imbalance(row, dataset_imbalance), axis=1)
    bad_stats_df["imbalance"] = bad_stats_df.apply(lambda row: _get_imbalance(row, dataset_imbalance), axis=1)

    cmap1 = matplotlib.colormaps["viridis"]
    cmap1 = matplotlib.colors.ListedColormap(cmap1(np.linspace(0.2, 0.9, 256)))
    cmap2 = matplotlib.colormaps["plasma_r"]
    colors1 = [cmap1(v) for v in np.linspace(0, 1, 5)]
    colors2 = [cmap2(v) for v in np.linspace(0, 1, 5)]
    colors = colors1 + colors2

    colors[5] = "#fcce25"  # overwrite Depressed
    colors[8] = "#7F2447"  # overwrite Stressed
    colors[9] = "#590278"  # overwrite Voices

    width_in_cm = 19.0 * 2
    height_in_cm = 8.0 * 2

    width_in_inches = width_in_cm / 2.54
    height_in_inches = height_in_cm / 2.54

    fig = plt.figure(figsize=(width_in_inches, height_in_inches))
    grid = plt.GridSpec(2, 3, figure=fig, wspace=0.4, hspace=0.4)

    for col in range(grid.ncols):
        for row in range(grid.nrows):
            ax = fig.add_subplot(grid[row, col])

            task = tasks[row]
            if col == 2:
                if row == 0:
                    xlabel = "MAE"
                    ylabel = "MAMAE"
                elif row == 1:
                    xlabel = "Acc"
                    ylabel = "BAcc"

                filtered_stats_df = eval_stats_df.loc[eval_stats_df.task == task]
                filtered_bad_stats_df = bad_stats_df.loc[bad_stats_df.task == task]
                _balanced_unbalanced_plot(
                    ax, filtered_stats_df, filtered_bad_stats_df,
                    labels, colors, labels_display
                )

                spearmanr, pvalue = stats.spearmanr(filtered_stats_df.performance_result, filtered_bad_stats_df.performance_result)
                ax.set_xlabel(xlabel)
                ax.set_ylabel(ylabel)
                ax.set_title(f"\u03C1={_format_sign(spearmanr)}")
                continue

            if col == 0:
                df = eval_stats_df
                ylabel = "MAMAE" if task == "ordinal_regression" else "BAcc"
            elif col == 1:
                df = bad_stats_df
                ylabel = "MAE" if task == "ordinal_regression" else "Acc"

            filtered_df = df.loc[df.task == task]
            spearmanr, pvalue = stats.spearmanr(filtered_df.imbalance, filtered_df.performance_result)
            _imbalance_plot(ax, filtered_df, labels, colors, labels_display)
            ax.set_ylabel(ylabel)
            ax.set_title(f"\u03C1={_format_sign(spearmanr)}")

    legend_handles = [matplotlib.patches.Patch(color=c, label=labels_display[l]) for c, l in zip(colors, labels)]
    ax.legend(handles=legend_handles, bbox_to_anchor=(1.05, 1), loc='upper left')

    fig.savefig(f"imbalance_performance_plot.{plot_kwargs.get('format', 'png')}", **plot_kwargs)
