from datetime import datetime

import pandas as pd


def d(date:str):
    return datetime.strptime(date, '%Y-%m-%d').date()


def show_full_pd_df():
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)

def assert_pd_equal_ignore_order(actual, expected):
    show_full_pd_df()
    return pd.testing.assert_frame_equal(
        expected.sort_values(by=list(expected.columns)).reset_index(drop=True),
        actual.sort_values(by=list(actual.columns)).reset_index(drop=True)
    )