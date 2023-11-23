from typing import NamedTuple

import pandas as pd

from hamilton.function_modifiers import (
    load_from,
    source,
    extract_columns,
)


DATASET_NAME = "crosscheck"
SEQUENCE_LENGTH = 3
MIN_N_EMA = 21
SPLIT_NAMES = ["train", "validation", "train_macro", "test"]
TIME_FEATURES = ["month", "day_of_month", "day_of_week", "is_weekend"]


class UserPartition(NamedTuple):
    user_id: str
    data: pd.DataFrame


class RecordIndex(NamedTuple):
    user_id: str
    timestamp: pd.DatetimeIndex


@extract_columns("timestamp")
@load_from.csv(path=source("raw_data_path"))
def raw_data(data: pd.DataFrame) -> pd.DataFrame:
    data = data.rename(columns={"day": "timestamp", "eureka_id": "user_id"})
    data = data.loc[data.timestamp != 19691231]
    data.timestamp = pd.to_datetime(data.timestamp, format="%Y%m%d").dt.normalize()
    return data


def labels() -> list[str]:
    return [
        "ema_CALM",
        "ema_HOPEFUL",
        "ema_SLEEPING",
        "ema_SOCIAL",
        "ema_THINK",
        "ema_DEPRESSED",
        "ema_HARM",
        "ema_SEEING_THINGS",
        "ema_STRESSED",
        "ema_VOICES",
    ]


def continuous_features() -> list[str]:
    return [
        "sleep_start",
        "sleep_end",
        "sleep_duration",
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
        "sms_in_num",
        "sms_out_num",
        "unlock_duration",
        "unlock_num",
    ]


def month(timestamp: pd.Series) -> pd.Series:
    return timestamp.dt.month


def day_of_month(timestamp: pd.Series) -> pd.Series:
    return timestamp.dt.day


def day_of_week(timestamp: pd.Series) -> pd.Series:
    return timestamp.dt.day_of_week


def is_weekend(day_of_week: pd.Series) -> pd.Series:
    return day_of_week.replace({0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 1, 6: 1})


def normalized_6h_features(raw_data: pd.DataFrame) -> pd.DataFrame:
    FEATURES_WITH_6H_PERIOD = [
        "act_in_vehicle_ep_1",
        "act_in_vehicle_ep_2",
        "act_in_vehicle_ep_3",
        "act_in_vehicle_ep_4",
        "act_on_bike_ep_1",
        "act_on_bike_ep_2",
        "act_on_bike_ep_3",
        "act_on_bike_ep_4",
        "act_on_foot_ep_1",
        "act_on_foot_ep_2",
        "act_on_foot_ep_3",
        "act_on_foot_ep_4",
        "act_running_ep_1",
        "act_running_ep_2",
        "act_running_ep_3",
        "act_running_ep_4",
        "act_still_ep_1",
        "act_still_ep_2",
        "act_still_ep_3",
        "act_still_ep_4",
        "act_tilting_ep_1",
        "act_tilting_ep_2",
        "act_tilting_ep_3",
        "act_tilting_ep_4",
        "act_unknown_ep_1",
        "act_unknown_ep_2",
        "act_unknown_ep_3",
        "act_unknown_ep_4",
        "act_walking_ep_1",
        "act_walking_ep_2",
        "act_walking_ep_3",
        "act_walking_ep_4",
    ]

    df = raw_data.copy()
    df[FEATURES_WITH_6H_PERIOD] /= 60 * 60 * 6
    return df[FEATURES_WITH_6H_PERIOD]


def normalized_24h_features(raw_data: pd.DataFrame) -> pd.DataFrame:
    FEATURES_WITH_24H_PERIOD = [
        "act_in_vehicle_ep_0",
        "act_on_bike_ep_0",
        "act_on_foot_ep_0",
        "act_running_ep_0",
        "act_still_ep_0",
        "act_tilting_ep_0",
        "act_unknown_ep_0",
        "act_walking_ep_0",
    ]

    df = raw_data.copy()
    df[FEATURES_WITH_24H_PERIOD] /= 60 * 60 * 24
    return df[FEATURES_WITH_24H_PERIOD]


def forward_fill_features(raw_data: pd.DataFrame) -> pd.DataFrame:
    FEATURES_TO_FORWARD_FILL = ["sleep_start", "sleep_end", "sleep_duration"]

    df = raw_data.copy()
    df[FEATURES_TO_FORWARD_FILL] = df[FEATURES_TO_FORWARD_FILL].shift(1)
    return df[FEATURES_TO_FORWARD_FILL]


def clean_data(
    raw_data: pd.DataFrame,
    month: pd.Series,
    day_of_month: pd.Series,
    day_of_week: pd.Series,
    is_weekend: pd.Series,
    normalized_24h_features: pd.DataFrame,
    normalized_6h_features: pd.DataFrame,
    forward_fill_features: pd.DataFrame,
) -> pd.DataFrame:
    df = raw_data.copy()

    df["month"] = month
    df["day_of_month"] = day_of_month
    df["day_of_week"] = day_of_week
    df["is_weekend"] = is_weekend
    df[normalized_24h_features.columns] = normalized_24h_features
    df[normalized_6h_features.columns] = normalized_6h_features
    df[forward_fill_features.columns] = forward_fill_features

    return df
