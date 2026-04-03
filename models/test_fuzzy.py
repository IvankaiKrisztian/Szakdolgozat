import pandas as pd

from models.fuzzy import TriangularFuzzySet, ShoulderFuzzySet, fuzzify, fuzzy_forecast, \
    prepare_fuzzified_forecast_demand_df
from utilities.utils import d, assert_pd_equal_ignore_order

VH_D = 'VeryHighDemand'
H_D = 'HighDemand'
L_D = 'LowDemand'

def test_triangular_fuzzy_set():
    f_set = TriangularFuzzySet(1, 4, 10)
    below_a = f_set.get_membership_value(0)
    above_a = f_set.get_membership_value(2)
    above_b = f_set.get_membership_value(5)
    full_membership_value = f_set.get_membership_value(4)
    above_c = f_set.get_membership_value(11)

    assert below_a == 0
    assert full_membership_value == 1
    assert above_c == 0
    assert above_a == 1 / 3
    assert above_b == 5 / 6


def test_left_shoulder_fuzzy_set():
    f_set = ShoulderFuzzySet(1, 4, "left")
    full_membership = f_set.get_membership_value(0)
    above_a = f_set.get_membership_value(2)
    above_b = f_set.get_membership_value(5)

    assert above_b == 0
    assert above_a == 2 / 3
    assert full_membership == 1


def test_right_shoulder_fuzzy_set():
    f_set = ShoulderFuzzySet(1, 4, "right")
    below_a = f_set.get_membership_value(0)
    above_a = f_set.get_membership_value(2)
    full_membership = f_set.get_membership_value(5)

    assert below_a == 0
    assert above_a == 1 / 3
    assert full_membership == 1


def test_fuzzify():
    fuzzy_list = [
        {
            "name": "VeryLowDemand",
            "type": "shoulder",
            "a": 1,
            "b": 4,
            "direction": "left",
        },
        {
            "name": "LowDemand",
            "type": "triangular",
            "a": 1,
            "b": 3,
            "c": 7,
        }
    ]
    x = 2
    assert fuzzify(fuzzy_list, x) == "VeryLowDemand"



def test_fuzzy_forecast_two_rule_firing():
    rule_base = pd.DataFrame([
        {'lag_1_fuzzy_set': L_D, 'lag_2_fuzzy_set': H_D, 'demand': 4},
        {'lag_1_fuzzy_set': H_D, 'lag_2_fuzzy_set': VH_D, 'demand': 8},
    ])

    fuzzified_demand_df = pd.DataFrame([
        {'lag_1_fuzzy_set': L_D, 'lag_2_fuzzy_set': L_D, 'lag_1_membership': 0.3, 'lag_2_membership': 0.5},
        {'lag_1_fuzzy_set': L_D, 'lag_2_fuzzy_set': VH_D, 'lag_1_membership': 0.3, 'lag_2_membership': 0.7},
        {'lag_1_fuzzy_set': L_D, 'lag_2_fuzzy_set': H_D, 'lag_1_membership': 0.3, 'lag_2_membership': 0.4},# matched rule
        {'lag_1_fuzzy_set': H_D, 'lag_2_fuzzy_set': VH_D, 'lag_1_membership': 0.2, 'lag_2_membership': 0.7},# matched rule
        {'lag_1_fuzzy_set': H_D, 'lag_2_fuzzy_set': H_D, 'lag_1_membership': 0.2, 'lag_2_membership': 0.4},
        {'lag_1_fuzzy_set': H_D, 'lag_2_fuzzy_set': L_D, 'lag_1_membership': 0.2, 'lag_2_membership': 0.5},
        {'lag_1_fuzzy_set': VH_D, 'lag_2_fuzzy_set': L_D, 'lag_1_membership': 0.2, 'lag_2_membership': 0.5},
        {'lag_1_fuzzy_set': VH_D, 'lag_2_fuzzy_set': H_D, 'lag_1_membership': 0.2, 'lag_2_membership': 0.4},
        {'lag_1_fuzzy_set': VH_D, 'lag_2_fuzzy_set': VH_D, 'lag_1_membership': 0.2, 'lag_2_membership': 0.7},
    ])

    firing_strengths = (.3 * .4) + (.2 * .7)
    first_rule = (.3 * .4) * 4
    second_rule = (.2 * .7) * 8

    expected_forecast = (
            (first_rule + second_rule)
            /
            firing_strengths
    )

    actual_forecast = fuzzy_forecast(rule_base, fuzzified_demand_df)

    assert actual_forecast == expected_forecast


def test_fuzzy_forecast_no_rule_firing():


    rule_base = pd.DataFrame([
        {'lag_1_fuzzy_set': L_D, 'lag_2_fuzzy_set': H_D, 'demand': 4},
        {'lag_1_fuzzy_set': H_D, 'lag_2_fuzzy_set': VH_D, 'demand': 8},
    ])

    fuzzified_demand_df = pd.DataFrame([
        {'lag_1_fuzzy_set': L_D, 'lag_2_fuzzy_set': L_D, 'lag_1_membership': 0, 'lag_2_membership': 0.5},
        {'lag_1_fuzzy_set': L_D, 'lag_2_fuzzy_set': VH_D, 'lag_1_membership': 0, 'lag_2_membership': 0.7},
        {'lag_1_fuzzy_set': L_D, 'lag_2_fuzzy_set': H_D, 'lag_1_membership': 0, 'lag_2_membership': 0.4},
        {'lag_1_fuzzy_set': H_D, 'lag_2_fuzzy_set': VH_D, 'lag_1_membership': 0, 'lag_2_membership': 0.7},
        {'lag_1_fuzzy_set': H_D, 'lag_2_fuzzy_set': H_D, 'lag_1_membership': 0, 'lag_2_membership': 0.4},
        {'lag_1_fuzzy_set': H_D, 'lag_2_fuzzy_set': L_D, 'lag_1_membership': 0, 'lag_2_membership': 0.5},
        {'lag_1_fuzzy_set': VH_D, 'lag_2_fuzzy_set': L_D, 'lag_1_membership': 0.2, 'lag_2_membership': 0.5},
        {'lag_1_fuzzy_set': VH_D, 'lag_2_fuzzy_set': H_D, 'lag_1_membership': 0.2, 'lag_2_membership': 0.4},
        {'lag_1_fuzzy_set': VH_D, 'lag_2_fuzzy_set': VH_D, 'lag_1_membership': 0.2, 'lag_2_membership': 0.7},
    ])

    actual_forecast = fuzzy_forecast(rule_base, fuzzified_demand_df)

    assert actual_forecast == 0


def test_prepare_fuzzified_forecast_demand_df():
    demand_df = pd.DataFrame([
        {'date': d('2026-02-03'), 'demand': 2},
        {'date': d('2026-02-04'), 'demand': 4},

    ])

    rule_base = pd.DataFrame([
        {'lag_1_fuzzy_set': L_D, 'lag_2_fuzzy_set': H_D, 'demand': 4},
        {'lag_1_fuzzy_set': H_D, 'lag_2_fuzzy_set': VH_D, 'demand': 8},
    ])

    fuzzy_list = [
        {
            "name": L_D,
            "type": "shoulder",
            "a": 1,
            "b": 6,
            "direction": "left",
        },
        {
            "name": H_D,
            "type": "triangular",
            "a": 2,
            "b": 5,
            "c": 7,
        },
        {
            "name": VH_D,
            "type": "triangular",
            "a": 4,
            "b": 7,
            "c": 10,
        }
    ]

    expected_demand_df = pd.DataFrame([
        {'lag_1_fuzzy_set': L_D, 'lag_2_fuzzy_set': H_D, 'lag_1_membership': 0.4, 'lag_2_membership': 0},
        {'lag_1_fuzzy_set': H_D, 'lag_2_fuzzy_set': VH_D, 'lag_1_membership': 2/3, 'lag_2_membership': 0},
    ])

    actual_demand_df = prepare_fuzzified_forecast_demand_df(demand_df, fuzzy_list, rule_base)

    assert_pd_equal_ignore_order(actual_demand_df, expected_demand_df)
