from collections import defaultdict
import itertools
from typing import NamedTuple

import numpy as np
import pandas as pd
from sklearn import set_config
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import SplineTransformer, PowerTransformer

from hamilton.htypes import Parallelizable, Collect
from hamilton.function_modifiers import source, parameterize, extract_fields, config


class UserPartition(NamedTuple):
    user_id: str
    data: pd.DataFrame


class RecordIndex(NamedTuple):
    user_id: str
    timestamp: pd.DatetimeIndex


SEQUENCE_LENGTH = 3
MIN_N_EMA = 21
SPLIT_NAMES = ["train", "validation", "train_macro", "test"]
TIME_FEATURES = ["month", "day_of_month", "day_of_week", "is_weekend"]


def user_partition(clean_data: pd.DataFrame) -> Parallelizable[UserPartition]:
    user_ids = clean_data.user_id.unique().tolist()
    for user_id in user_ids:
        partition_df = clean_data.loc[clean_data.user_id == user_id]
        yield UserPartition(user_id, partition_df)


def user_sorted_dates_with_ema(user_partition: UserPartition, labels: list[str]) -> np.ndarray:
    df = user_partition.data
    return df.loc[~df[labels[0]].isna(), "timestamp"].sort_values(ascending=True).unique()


def user_input_sequence_dates(
    user_partition: UserPartition,
    user_sorted_dates_with_ema: np.ndarray,
    offset: int,
    minimum_n_ema: int = MIN_N_EMA,
    sequence_length: int = SEQUENCE_LENGTH,
) -> dict[RecordIndex, pd.DatetimeIndex]:
    user_id = user_partition.user_id
    df = user_partition.data
    input_sequence_dates = {}

    if len(user_sorted_dates_with_ema) < minimum_n_ema:
        return {}

    for date_with_ema in user_sorted_dates_with_ema:
        end_date = date_with_ema - pd.Timedelta(offset, unit="d")
        start_date = end_date - pd.Timedelta(sequence_length - 1, unit="d")
        input_sequence_date_range = pd.date_range(start_date, end_date)

        sequence = df.loc[df.timestamp.isin(input_sequence_date_range)]

        if len(sequence.timestamp.unique()) != sequence_length:
            continue

        key = RecordIndex(user_id, date_with_ema)
        input_sequence_dates[key] = input_sequence_date_range

    return input_sequence_dates


def user_splits_ema_dates(
    user_input_sequence_dates: dict[RecordIndex, pd.DatetimeIndex],
    minimum_n_ema: int = MIN_N_EMA,
) -> dict[str, list[RecordIndex]]:
    """NOTE. the keys of user_input_sequence_dates != user_sorted_dates_with_ema
    because certain dates rejected for invalid input sequences
    NOTE. we need to do another check for MINIMUM_N_EMA because of rejected input sequences
    """

    N_TEST_EMA = 7
    N_VALIDATION_EMA = 7

    indices: list[RecordIndex] = list(user_input_sequence_dates.keys())
    sorted_valid_ema_dates = sorted(indices, key=lambda i: i.timestamp)

    splits_dates = dict(train=[], validation=[], train_macro=[], test=[])

    if len(sorted_valid_ema_dates) < minimum_n_ema:  # keep data only if user has enough valid sequences
        return splits_dates

    for idx, label_date in enumerate(sorted_valid_ema_dates[::-1]):  # reverse iteration (descending)
        if idx < N_TEST_EMA:  # newest data is test
            splits_dates["test"].append(label_date)

        elif idx < N_TEST_EMA + N_VALIDATION_EMA:  # 'second newest' is validation
            splits_dates["validation"].append(label_date)
            splits_dates["train_macro"].append(label_date)

        else:  # the rest is train
            splits_dates["train"].append(label_date)
            splits_dates["train_macro"].append(label_date)

    return splits_dates


def _date_range_union(list_of_ranges: list[pd.DatetimeIndex]) -> pd.DatetimeIndex:
    full_range = pd.DatetimeIndex([])
    for range_ in list_of_ranges:
        full_range = full_range.union(range_)

    return full_range


@config.when(dataset="crosscheck")
def user_splits_df__crosscheck(
    user_partition: UserPartition,
    user_splits_ema_dates: dict[str, list[RecordIndex]],
    user_input_sequence_dates: dict[RecordIndex, pd.DatetimeIndex],
) -> dict:
    df = user_partition.data

    train_features_dates = _date_range_union([user_input_sequence_dates[idx] for idx in user_splits_ema_dates["train"]])
    validation_feature_dates = _date_range_union(
        [user_input_sequence_dates[idx] for idx in user_splits_ema_dates["validation"]]
    )
    train_macro_feature_dates = _date_range_union(
        [user_input_sequence_dates[idx] for idx in user_splits_ema_dates["train_macro"]]
    )
    test_feature_dates = _date_range_union([user_input_sequence_dates[idx] for idx in user_splits_ema_dates["test"]])

    train_df = df.loc[df.timestamp.isin(train_features_dates)]
    validation_df = df.loc[df.timestamp.isin(validation_feature_dates)]
    train_macro_df = df.loc[df.timestamp.isin(train_macro_feature_dates)]
    test_df = df.loc[df.timestamp.isin(test_feature_dates)]

    return dict(
        train=train_df,
        validation=validation_df,
        train_macro=train_macro_df,
        test=test_df,
    )


@config.when(dataset="friendsfamily")
def user_splits_df__friendsfamily(
    user_partition: UserPartition,
    user_splits_ema_dates: dict[str, list[RecordIndex]],
    user_input_sequence_dates: dict[RecordIndex, pd.DatetimeIndex],
    temp_dir_path: str,
) -> dict:
    df = user_partition.data

    train_features_dates = _date_range_union([user_input_sequence_dates[idx] for idx in user_splits_ema_dates["train"]])
    validation_feature_dates = _date_range_union(
        [user_input_sequence_dates[idx] for idx in user_splits_ema_dates["validation"]]
    )
    train_macro_feature_dates = _date_range_union(
        [user_input_sequence_dates[idx] for idx in user_splits_ema_dates["train_macro"]]
    )
    test_feature_dates = _date_range_union([user_input_sequence_dates[idx] for idx in user_splits_ema_dates["test"]])

    train_df = df.loc[df.timestamp.isin(train_features_dates)]
    validation_df = df.loc[df.timestamp.isin(validation_feature_dates)]
    train_macro_df = df.loc[df.timestamp.isin(train_macro_feature_dates)]
    test_df = df.loc[df.timestamp.isin(test_feature_dates)]

    output_paths = dict(
        train=f"{temp_dir_path}/train_{user_partition.user_id}.parquet",
        validation=f"{temp_dir_path}/validation_{user_partition.user_id}.parquet",
        train_macro=f"{temp_dir_path}/train_macro_{user_partition.user_id}.parquet",
        test=f"{temp_dir_path}/test_{user_partition.user_id}.parquet",
    )

    train_df.to_parquet(output_paths["train"])
    validation_df.to_parquet(output_paths["validation"])
    train_macro_df.to_parquet(output_paths["train_macro"])
    test_df.to_parquet(output_paths["test"])

    return output_paths


def _melt_feature(df: pd.DataFrame, indices: list[str], feature_name: str, feature_epochs: list[str]) -> pd.DataFrame:
    """Unravel a dataframe of shape (n_examples, 1 feature * n_time_periods)
    into (n_examples * n_time_periods, 1 feature);
    Works on one feature at a time to avoid memory issues
    """
    single_var_df = pd.melt(df[feature_epochs + indices], id_vars=indices)
    # extract the epoch info from feature column name
    single_var_df.insert(2, "epoch", single_var_df.variable.str.extract(r"([1234])"))
    single_var_df = single_var_df.sort_values(["timestamp", "epoch"])
    single_var_df = single_var_df.rename(columns={"value": feature_name})
    single_var_df = single_var_df.drop(columns=["variable"])
    return single_var_df


def _melt_dataframe(df: pd.DataFrame, indices: list[str] = ["user_id", "timestamp"]) -> pd.DataFrame:
    """Convert a dataframe of shape (n_examples, n_features * n_time_periods)
    into (n_examples * n_time_periods, n_features);
    First identify features with multiple epoch columns based on column name,
    then apply `melt_feature()` on each feature
    """
    merged_df = None

    # Identify features with multiple epoch columns
    feature_epochs = defaultdict(list)
    for col in df.columns:
        if ("_ep_" not in col) or ("0" in col):
            continue

        partitions = col.partition("_ep_")
        feature_name = partitions[0]
        feature_epochs[feature_name].append(col)

    # Melt feature epochs then merge with main dataframe one by one
    for feature_name, epochs in feature_epochs.items():
        melted_df = _melt_feature(df, indices, feature_name, epochs)

        if merged_df is None:
            merged_df = melted_df
        else:
            merged_df: pd.DataFrame = pd.merge(merged_df, melted_df, on=indices + ["epoch"])

    return merged_df


@config.when(dataset="crosscheck")
def user_melt_df__crosscheck(
    user_splits_df: dict,
    labels: list[str],
    time_features: list[str] = TIME_FEATURES,
) -> dict:
    INDICES = ["user_id", "timestamp"]
    FEATURES_DAILY_FILL = ["sleep_start", "sleep_end", "sleep_duration"] + time_features

    melted_dfs = dict()

    for split_name, split_df in user_splits_df.items():
        melted_df = _melt_dataframe(split_df, indices=INDICES)
        merged_df = pd.merge(split_df[INDICES + FEATURES_DAILY_FILL + labels], melted_df, on=INDICES)
        melted_dfs[split_name] = merged_df

    return melted_dfs


def _flatten_collected_splits_list(collected: list[dict], split_name: str) -> list:
    return list(itertools.chain(*[partition[split_name] for partition in collected]))


@extract_fields(
    dict(
        train_ema_dates=list[RecordIndex],
        validation_ema_dates=list[RecordIndex],
        train_macro_ema_dates=list[RecordIndex],
        test_ema_dates=list[RecordIndex],
    )
)
def global_splits_ema_dates(
    user_splits_ema_dates: Collect[dict],
) -> dict[str, list[RecordIndex]]:
    return dict(
        train_ema_dates=_flatten_collected_splits_list(user_splits_ema_dates, split_name="train"),
        validation_ema_dates=_flatten_collected_splits_list(user_splits_ema_dates, split_name="validation"),
        train_macro_ema_dates=_flatten_collected_splits_list(user_splits_ema_dates, split_name="train_macro"),
        test_ema_dates=_flatten_collected_splits_list(user_splits_ema_dates, split_name="test"),
    )


def global_input_sequence_dates(
    user_input_sequence_dates: Collect[dict[RecordIndex, pd.DatetimeIndex]]
) -> dict[RecordIndex, pd.DatetimeIndex]:
    return dict((k, v) for partition in user_input_sequence_dates for k, v in partition.items())


@config.when(dataset="crosscheck")
@extract_fields(
    dict(
        train_df=pd.DataFrame,
        validation_df=pd.DataFrame,
        train_macro_df=pd.DataFrame,
        test_df=pd.DataFrame,
    )
)
def global_splits_df__crosscheck(user_melt_df: Collect[dict]) -> dict[str, pd.DataFrame]:
    train_df = pd.concat([partition["train"] for partition in user_melt_df], axis=0, ignore_index=True)
    validation_df = pd.concat(
        [partition["validation"] for partition in user_melt_df],
        axis=0,
        ignore_index=True,
    )
    train_macro_df = pd.concat(
        [partition["train_macro"] for partition in user_melt_df],
        axis=0,
        ignore_index=True,
    )
    test_df = pd.concat([partition["test"] for partition in user_melt_df], axis=0, ignore_index=True)

    return dict(
        train_df=train_df,
        validation_df=validation_df,
        train_macro_df=train_macro_df,
        test_df=test_df,
    )


@config.when(dataset="friendsfamily")
@extract_fields(
    dict(
        train_df=pd.DataFrame,
        validation_df=pd.DataFrame,
        train_macro_df=pd.DataFrame,
        test_df=pd.DataFrame,
    )
)
def global_splits_df__friendsfamily(
    user_splits_df: Collect[dict],
    temp_dir_path: str
) -> dict[str, pd.DataFrame]:
    import duckdb

    train_df = duckdb.read_parquet(f"{temp_dir_path}/train_*.parquet").df()
    validation_df = duckdb.read_parquet(f"{temp_dir_path}/validation_*.parquet").df()
    train_macro_df = duckdb.read_parquet(f"{temp_dir_path}/train_macro*.parquet").df()
    test_df = duckdb.read_parquet(f"{temp_dir_path}/test_*.parquet").df()

    return dict(
        train_df=train_df,
        validation_df=validation_df,
        train_macro_df=train_macro_df,
        test_df=test_df,
    )


def _get_spline_transformer(period, n_splines=None, degree=3):
    if n_splines is None:
        n_splines = period

    n_knots = n_splines + 1  # periodic and include_bias is True
    return SplineTransformer(
        degree=degree,
        n_knots=n_knots,
        knots=np.linspace(0, period, n_knots).reshape(n_knots, 1),
        extrapolation="periodic",
        include_bias=True,
    )


def _get_time_pipeline():
    return Pipeline(
        [
            (
                "splines",
                ColumnTransformer(
                    [
                        (
                            "month_of_year_cycle",
                            _get_spline_transformer(12, n_splines=6),
                            ["month"],
                        ),
                        (
                            "day_of_month_cycle",
                            _get_spline_transformer(31, n_splines=10),
                            ["day_of_month"],
                        ),
                        (
                            "day_of_week_cycle",
                            _get_spline_transformer(7, n_splines=3),
                            ["day_of_week"],
                        ),
                    ],
                    remainder="passthrough",
                    verbose_feature_names_out=False,
                ),
            )
        ]
    )


def _get_continuous_pipeline():
    return Pipeline(
        [
            ("standardization", PowerTransformer(method="yeo-johnson")),
            ("imputer", SimpleImputer(strategy="constant", fill_value=0)),
        ]
    )


def _get_preprocessing_pipeline(continuous_features, time_features, labels):
    return Pipeline(
        [
            (
                "column_transformer",
                ColumnTransformer(
                    [
                        ("continuous", _get_continuous_pipeline(), continuous_features),
                        ("time", _get_time_pipeline(), time_features),
                        ("labels", "passthrough", labels),
                    ],
                    remainder="passthrough",
                    verbose_feature_names_out=False,
                ),
            ),
        ]
    )


@extract_fields(
    dict(
        train_scaled_df=pd.DataFrame,
        validation_scaled_df=pd.DataFrame,
        train_macro_scaled_df=pd.DataFrame,
        test_scaled_df=pd.DataFrame,
    )
)
def scaled_splits_df(
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    train_macro_df: pd.DataFrame,
    test_df: pd.DataFrame,
    labels: list[str],
    continuous_features: list[str],
    time_features: list[str] = TIME_FEATURES,
) -> dict[str, pd.DataFrame]:
    set_config(transform_output="pandas")

    pipeline = _get_preprocessing_pipeline(
        continuous_features=continuous_features,
        time_features=time_features,
        labels=labels,
    )

    train_scaled_df = pipeline.fit_transform(train_df)
    validation_scaled_df = pipeline.transform(validation_df)

    train_macro_scaled_df = pipeline.fit_transform(train_macro_df)
    test_scaled_df = pipeline.transform(test_df)

    return dict(
        train_scaled_df=train_scaled_df,
        validation_scaled_df=validation_scaled_df,
        train_macro_scaled_df=train_macro_scaled_df,
        test_scaled_df=test_scaled_df,
    )


SEQUENCE_SPLITS_SHARED_PARAMS = dict(
    clean_data=source("clean_data"),
    global_sequence_dates=source("global_input_sequence_dates"),
    labels=source("labels"),
)

@parameterize(
    train_sequence_df=dict(
        split_df=source("train_scaled_df"), ema_dates=source("train_ema_dates"), **SEQUENCE_SPLITS_SHARED_PARAMS
    ),
    validation_sequence_df=dict(
        split_df=source("validation_scaled_df"),
        ema_dates=source("validation_ema_dates"),
        **SEQUENCE_SPLITS_SHARED_PARAMS
    ),
    train_macro_sequence_df=dict(
        split_df=source("train_macro_scaled_df"),
        ema_dates=source("train_macro_ema_dates"),
        **SEQUENCE_SPLITS_SHARED_PARAMS
    ),
    test_sequence_df=dict(
        split_df=source("test_scaled_df"), ema_dates=source("test_ema_dates"), **SEQUENCE_SPLITS_SHARED_PARAMS
    ),
)
def sequence_splits_df(
    split_df: pd.DataFrame,
    clean_data: pd.DataFrame,
    ema_dates: list[RecordIndex],
    labels: list[str],
    global_sequence_dates: dict[RecordIndex, pd.DatetimeIndex],
) -> pd.DataFrame:
    all_sequences = []

    sequence_idx = 0
    for record_index in ema_dates:
        # get the consecutive input 3 dates associated with the ema date
        sequence_dates = global_sequence_dates[record_index]

        # select the features columns for the input dates
        feature_mask = (split_df.user_id == record_index.user_id) & (split_df.timestamp.isin(sequence_dates))
        single_sequence = split_df.loc[feature_mask].copy()

        # select all labels at ema date; set previous values as nan because irrelevant
        label_mask = (clean_data.user_id == record_index.user_id) & (clean_data.timestamp == record_index.timestamp)
        single_sequence.loc[:, labels] = np.nan
        values = clean_data.loc[label_mask, labels].tail(1).values[0]
        for k, v in zip(labels, values):
            single_sequence.loc[single_sequence.index[-1], k] = v

        # give a unique global index to the sequence
        single_sequence["seq_idx"] = sequence_idx
        all_sequences.append(single_sequence)
        sequence_idx += 1

    sequence_df = pd.concat(all_sequences, axis=0, ignore_index=True)
    sequence_df["intra_seq_idx"] = sequence_df.groupby("seq_idx").cumcount()

    return sequence_df
