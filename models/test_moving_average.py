import pandas as pd

from models.moving_average import forecast_moving_average
from utilities.utils import parse_date


def test_forecast_moving_average():
    demand_df = pd.DataFrame([
        {'date': parse_date('2025-12-31'), 'demand': 4},
        {'date': parse_date('2026-01-01'), 'demand': 4},
        {'date': parse_date('2026-01-02'), 'demand': 3},
        {'date': parse_date('2026-01-03'), 'demand': 7},
    ])
    lags = 3

    expected = 14 / 3

    actual,explanation = forecast_moving_average(demand_df, lags)

    assert actual == expected
