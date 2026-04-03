

import pandas as pd

from evaluate.forecast import forecast_over_horizon
from models.fuzzy import fuzzy_forecast_pipeline
from models.moving_average import forecast_moving_average
from utilities.utils import d, assert_pd_equal_ignore_order

L_D = 'LowDemand'

def test_forecast_over_horizon_with_average():
    h = 7
    start_date = d('2026-12-30')

    demand_df = pd.DataFrame([
        {'date':d('2026-12-20'), 'demand': 4},
        {'date':d('2026-12-21'), 'demand': 6},
        {'date':d('2026-12-22'), 'demand': 3},
        {'date':d('2026-12-23'), 'demand': 4},
        {'date':d('2026-12-24'), 'demand': 8},
        {'date':d('2026-12-25'), 'demand': 10},
        {'date':d('2026-12-26'), 'demand': 12},
        {'date':d('2026-12-27'), 'demand': 6},
        {'date':d('2026-12-28'), 'demand': 3},
        {'date':d('2026-12-29'), 'demand': 2},
        {'date':d('2026-12-30'), 'demand': 1},
        {'date':d('2026-12-31'), 'demand': 0},
        {'date':d('2027-01-01'), 'demand': 6},
        {'date':d('2027-01-02'), 'demand': 4},
        {'date':d('2027-01-03'), 'demand': 12},
        {'date':d('2027-01-04'), 'demand': 3},
        {'date':d('2027-01-05'), 'demand': 7},
    ])

    expected_df = pd.DataFrame([
        {'date': d('2026-12-30'), 'prediction': 45 / 7},
        {'date': d('2026-12-31'), 'prediction': 42 / 7},
        {'date': d('2027-01-01'), 'prediction': 34 / 7},
        {'date': d('2027-01-02'), 'prediction': 30 / 7},
        {'date': d('2027-01-03'), 'prediction': 22 / 7},
        {'date': d('2027-01-04'), 'prediction': 28 / 7},
        {'date': d('2027-01-05'), 'prediction': 28 / 7},
    ])

    args = [7]
    forecast_method = forecast_moving_average

    actual_df = forecast_over_horizon(demand_df,start_date,h,forecast_method,args)

    assert_pd_equal_ignore_order(actual_df,expected_df)


def test_forecast_over_horizon_with_fuzzy():
    h = 7
    start_date = d('2026-12-30')

    demand_df = pd.DataFrame([
        {'date':d('2026-12-20'), 'demand': 4},
        {'date':d('2026-12-21'), 'demand': 6},
        {'date':d('2026-12-22'), 'demand': 3},
        {'date':d('2026-12-23'), 'demand': 4},
        {'date':d('2026-12-24'), 'demand': 8},
        {'date':d('2026-12-25'), 'demand': 10},
        {'date':d('2026-12-26'), 'demand': 12},
        {'date':d('2026-12-27'), 'demand': 6},
        {'date':d('2026-12-28'), 'demand': 3},
        {'date':d('2026-12-29'), 'demand': 2},
        {'date':d('2026-12-30'), 'demand': 1},
        {'date':d('2026-12-31'), 'demand': 0},
        {'date':d('2027-01-01'), 'demand': 6},
        {'date':d('2027-01-02'), 'demand': 4},
        {'date':d('2027-01-03'), 'demand': 12},
        {'date':d('2027-01-04'), 'demand': 3},
        {'date':d('2027-01-05'), 'demand': 7},
    ])

    expected_df = pd.DataFrame([
        {'date': d('2026-12-30'), 'prediction': 5.142857},
        {'date': d('2026-12-31'), 'prediction': 4.8},
        {'date': d('2027-01-01'), 'prediction': 4.8},
        {'date': d('2027-01-02'), 'prediction': 4.8},
        {'date': d('2027-01-03'), 'prediction': 4.8},
        {'date': d('2027-01-04'), 'prediction': 4.8},
        {'date': d('2027-01-05'), 'prediction': 6.222222},
    ])

    fuzzy_list = [
        {
            "name": L_D,
            "type": "shoulder",
            "a": 0,
            "b": 4,
            "direction": "left",
        },
        {
            "name": 'HighDemand',
            "type": "triangular",
            "a": 1,
            "b": 3,
            "c": 7,
        },
        {
            "name": 'VeryHighDemand',
            "type": "shoulder",
            "a": 6,
            "b": 12,
            "direction": "right",
        },
    ]

    rule_base = pd.DataFrame([
        {'lag_1_fuzzy_set': L_D, 'lag_2_fuzzy_set': 'HighDemand', 'demand': 4},
        {'lag_1_fuzzy_set': 'HighDemand', 'lag_2_fuzzy_set': 'VeryHighDemand', 'demand': 8},
    ])


    args = [fuzzy_list,2,rule_base]
    forecast_method = fuzzy_forecast_pipeline

    actual_df = forecast_over_horizon(demand_df,start_date,h,forecast_method,args)

    assert_pd_equal_ignore_order(actual_df,expected_df)

