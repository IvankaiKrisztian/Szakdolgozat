import pandas as pd

from utilities.utils import parse_date, get_split_date


def test_get_split_date():
    df = pd.DataFrame([
        {'date': parse_date('2026-01-01')},
        {'date': parse_date('2026-01-02')},
        {'date': parse_date('2026-01-04')},
        {'date': parse_date('2026-01-06')},
    ])
    split_percentage = 0.75

    actual = get_split_date(df, split_percentage)

    assert actual == parse_date('2026-01-04')