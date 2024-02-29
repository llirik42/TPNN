from dateutil import tz
from datetime import datetime

import pandas as pd


def preprocess_features(df: pd.DataFrame) -> None:
    preprocess_formatted_date(df)
    preprocess_summary_columns(df)
    preprocess_recip_type(df)
    preprocess_loud_cover(df)


def preprocess_formatted_date(df: pd.DataFrame) -> None:
    to_zone: tz.tzlocal = tz.tzlocal()
    dates: pd.Series = df['Formatted Date'].apply(lambda x: datetime.strptime(x, "%Y-%m-%d %H:%M:%S.%f %z").astimezone(tz=to_zone))
    df.drop(['Formatted Date'], axis=1, inplace=True)
    df['Year'] = dates.dt.year
    df['Month'] = dates.dt.month
    df['Day'] = dates.dt.day
    df['Weekday'] = dates.dt.weekday
    df['Hour'] = dates.dt.hour


def preprocess_summary_columns(df: pd.DataFrame) -> None:
    df.drop('Summary', axis=1, inplace=True)
    df.drop('Daily Summary', axis=1, inplace=True)


def preprocess_recip_type(df: pd.DataFrame) -> None:
    df['Rain'] = df['Precip Type'].apply(lambda x: float(x == 'rain'))
    df['Snow'] = df['Precip Type'].apply(lambda x: float(x == 'snow'))
    df.drop(['Precip Type'], axis=1, inplace=True)


def preprocess_loud_cover(df: pd.DataFrame) -> None:
    df.drop('Loud Cover', axis=1, inplace=True)


def handle_remissions(df: pd.DataFrame) -> None:
    to_median: pd.DataFrame = df['Pressure (millibars)'].replace(value=df['Pressure (millibars)'].median(), to_replace=0.0)
    df['Pressure (millibars)'] = to_median
