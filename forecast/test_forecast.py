import pytest
import pandas as pd

from forecast.forecast import forecast_over_horizon
from models.fuzzy import fuzzy_forecast_pipeline
from models.moving_average import forecast_moving_average
from utilities.utils import parse_date, assert_pd_equal_ignore_order

VERY_HIGH_DEMAND = 'VeryHighDemand'
HIGH_DEMAND = 'HighDemand'
LOW_DEMAND = 'LowDemand'


@pytest.fixture
def demand_df():
    return pd.DataFrame([
        {'date': parse_date('2026-12-20'), 'demand': 4},
        {'date': parse_date('2026-12-21'), 'demand': 6},
        {'date': parse_date('2026-12-22'), 'demand': 3},
        {'date': parse_date('2026-12-23'), 'demand': 4},
        {'date': parse_date('2026-12-24'), 'demand': 8},
        {'date': parse_date('2026-12-25'), 'demand': 10},
        {'date': parse_date('2026-12-26'), 'demand': 12},
        {'date': parse_date('2026-12-27'), 'demand': 6},
        {'date': parse_date('2026-12-28'), 'demand': 3},
        {'date': parse_date('2026-12-29'), 'demand': 2},
        {'date': parse_date('2026-12-30'), 'demand': 1},
        {'date': parse_date('2026-12-31'), 'demand': 0},
        {'date': parse_date('2027-01-01'), 'demand': 6},
        {'date': parse_date('2027-01-02'), 'demand': 4},
        {'date': parse_date('2027-01-03'), 'demand': 12},
        {'date': parse_date('2027-01-04'), 'demand': 3},
        {'date': parse_date('2027-01-05'), 'demand': 7},
    ])


def test_forecast_over_horizon_with_average(demand_df):
    forecast_horizon = 7
    start_date = parse_date('2026-12-30')

    expected_df = pd.DataFrame([
        {'date': parse_date('2026-12-30'), 'prediction': 45 / 7},
        {'date': parse_date('2026-12-31'), 'prediction': 42 / 7},
        {'date': parse_date('2027-01-01'), 'prediction': 34 / 7},
        {'date': parse_date('2027-01-02'), 'prediction': 30 / 7},
        {'date': parse_date('2027-01-03'), 'prediction': 22 / 7},
        {'date': parse_date('2027-01-04'), 'prediction': 28 / 7},
        {'date': parse_date('2027-01-05'), 'prediction': 28 / 7},
    ])

    actual_df = forecast_over_horizon(demand_df, start_date, forecast_horizon, forecast_moving_average, [7])

    assert_pd_equal_ignore_order(actual_df, expected_df)


def test_forecast_over_horizon_with_fuzzy(demand_df):
    forecast_horizon = 7
    start_date = parse_date('2026-12-30')

    expected_df = pd.DataFrame([
        {'date': parse_date('2026-12-30'), 'prediction': 4.0},
        {'date': parse_date('2026-12-31'), 'prediction': 4.0},
        {'date': parse_date('2027-01-01'), 'prediction': 0.0},
        {'date': parse_date('2027-01-02'), 'prediction': 0.0},
        {'date': parse_date('2027-01-03'), 'prediction': 0.0},
        {'date': parse_date('2027-01-04'), 'prediction': 0.0},
        {'date': parse_date('2027-01-05'), 'prediction': 8.0},
    ])

    fuzzy_list = [
        {
            "name": LOW_DEMAND,
            "type": "shoulder",
            "a": 0,
            "b": 4,
            "direction": "left",
        },
        {
            "name": HIGH_DEMAND,
            "type": "triangular",
            "a": 1,
            "b": 3,
            "c": 7,
        },
        {
            "name": VERY_HIGH_DEMAND,
            "type": "shoulder",
            "a": 6,
            "b": 12,
            "direction": "right",
        },
    ]

    rule_base = pd.DataFrame([
        {'lag_1_fuzzy_set': LOW_DEMAND, 'lag_2_fuzzy_set': HIGH_DEMAND, 'demand': 4},
        {'lag_1_fuzzy_set': HIGH_DEMAND, 'lag_2_fuzzy_set': VERY_HIGH_DEMAND, 'demand': 8},
    ])

    actual_df = forecast_over_horizon(demand_df, start_date, forecast_horizon, fuzzy_forecast_pipeline, [fuzzy_list, rule_base])

    assert_pd_equal_ignore_order(actual_df, expected_df)
