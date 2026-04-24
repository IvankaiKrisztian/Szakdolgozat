import pandas as pd

from models.fuzzy import TriangularFuzzySet, ShoulderFuzzySet, fuzzy_forecast, \
    prepare_fuzzified_forecast_demand_df, fuzzify_by_set_name, get_linguistic_form
from utilities.utils import parse_date, assert_pd_equal_ignore_order

VERY_HIGH_DEMAND = 'VeryHighDemand'
HIGH_DEMAND = 'HighDemand'
LOW_DEMAND = 'LowDemand'


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
            "name": LOW_DEMAND,
            "type": "triangular",
            "a": 1,
            "b": 3,
            "c": 7,
        }
    ]
    demand_value = 2
    assert fuzzify_by_set_name(fuzzy_list, demand_value) == "VeryLowDemand"


def test_fuzzy_forecast_two_rule_firing():
    rule_base = pd.DataFrame([
        {'lag_1_fuzzy_set': LOW_DEMAND, 'lag_2_fuzzy_set': HIGH_DEMAND, 'demand': 4},
        {'lag_1_fuzzy_set': HIGH_DEMAND, 'lag_2_fuzzy_set': VERY_HIGH_DEMAND, 'demand': 8},
    ])

    fuzzified_demand_df = pd.DataFrame([
        {'lag_1_fuzzy_set': LOW_DEMAND, 'lag_2_fuzzy_set': LOW_DEMAND, 'lag_1_membership': 0.3, 'lag_2_membership': 0.5},
        {'lag_1_fuzzy_set': LOW_DEMAND, 'lag_2_fuzzy_set': VERY_HIGH_DEMAND, 'lag_1_membership': 0.3, 'lag_2_membership': 0.7},
        {'lag_1_fuzzy_set': LOW_DEMAND, 'lag_2_fuzzy_set': HIGH_DEMAND, 'lag_1_membership': 0.3, 'lag_2_membership': 0.4},  # matched rule
        {'lag_1_fuzzy_set': HIGH_DEMAND, 'lag_2_fuzzy_set': VERY_HIGH_DEMAND, 'lag_1_membership': 0.2, 'lag_2_membership': 0.7},  # matched rule
        {'lag_1_fuzzy_set': HIGH_DEMAND, 'lag_2_fuzzy_set': HIGH_DEMAND, 'lag_1_membership': 0.2, 'lag_2_membership': 0.4},
        {'lag_1_fuzzy_set': HIGH_DEMAND, 'lag_2_fuzzy_set': LOW_DEMAND, 'lag_1_membership': 0.2, 'lag_2_membership': 0.5},
        {'lag_1_fuzzy_set': VERY_HIGH_DEMAND, 'lag_2_fuzzy_set': LOW_DEMAND, 'lag_1_membership': 0.2, 'lag_2_membership': 0.5},
        {'lag_1_fuzzy_set': VERY_HIGH_DEMAND, 'lag_2_fuzzy_set': HIGH_DEMAND, 'lag_1_membership': 0.2, 'lag_2_membership': 0.4},
        {'lag_1_fuzzy_set': VERY_HIGH_DEMAND, 'lag_2_fuzzy_set': VERY_HIGH_DEMAND, 'lag_1_membership': 0.2, 'lag_2_membership': 0.7},
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
        {'lag_1_fuzzy_set': LOW_DEMAND, 'lag_2_fuzzy_set': HIGH_DEMAND, 'demand': 4},
        {'lag_1_fuzzy_set': HIGH_DEMAND, 'lag_2_fuzzy_set': VERY_HIGH_DEMAND, 'demand': 8},
    ])

    fuzzified_demand_df = pd.DataFrame([
        {'lag_1_fuzzy_set': LOW_DEMAND, 'lag_2_fuzzy_set': LOW_DEMAND, 'lag_1_membership': 0, 'lag_2_membership': 0.5},
        {'lag_1_fuzzy_set': LOW_DEMAND, 'lag_2_fuzzy_set': VERY_HIGH_DEMAND, 'lag_1_membership': 0, 'lag_2_membership': 0.7},
        {'lag_1_fuzzy_set': LOW_DEMAND, 'lag_2_fuzzy_set': HIGH_DEMAND, 'lag_1_membership': 0, 'lag_2_membership': 0.4},
        {'lag_1_fuzzy_set': HIGH_DEMAND, 'lag_2_fuzzy_set': VERY_HIGH_DEMAND, 'lag_1_membership': 0, 'lag_2_membership': 0.7},
        {'lag_1_fuzzy_set': HIGH_DEMAND, 'lag_2_fuzzy_set': HIGH_DEMAND, 'lag_1_membership': 0, 'lag_2_membership': 0.4},
        {'lag_1_fuzzy_set': HIGH_DEMAND, 'lag_2_fuzzy_set': LOW_DEMAND, 'lag_1_membership': 0, 'lag_2_membership': 0.5},
        {'lag_1_fuzzy_set': VERY_HIGH_DEMAND, 'lag_2_fuzzy_set': LOW_DEMAND, 'lag_1_membership': 0.2, 'lag_2_membership': 0.5},
        {'lag_1_fuzzy_set': VERY_HIGH_DEMAND, 'lag_2_fuzzy_set': HIGH_DEMAND, 'lag_1_membership': 0.2, 'lag_2_membership': 0.4},
        {'lag_1_fuzzy_set': VERY_HIGH_DEMAND, 'lag_2_fuzzy_set': VERY_HIGH_DEMAND, 'lag_1_membership': 0.2, 'lag_2_membership': 0.7},
    ])

    actual_forecast = fuzzy_forecast(rule_base, fuzzified_demand_df)

    assert actual_forecast == 6


def test_prepare_fuzzified_forecast_demand_df():
    demand_df = pd.DataFrame([
        {'date': parse_date('2026-02-03'), 'demand': 2},
        {'date': parse_date('2026-02-04'), 'demand': 4},
    ])

    rule_base = pd.DataFrame([
        {'lag_1_fuzzy_set': LOW_DEMAND, 'lag_2_fuzzy_set': HIGH_DEMAND, 'demand': 4},
        {'lag_1_fuzzy_set': HIGH_DEMAND, 'lag_2_fuzzy_set': VERY_HIGH_DEMAND, 'demand': 8},
    ])

    fuzzy_list = [
        {
            "name": LOW_DEMAND,
            "type": "shoulder",
            "a": 1,
            "b": 6,
            "direction": "left",
        },
        {
            "name": HIGH_DEMAND,
            "type": "triangular",
            "a": 2,
            "b": 5,
            "c": 7,
        },
        {
            "name": VERY_HIGH_DEMAND,
            "type": "triangular",
            "a": 4,
            "b": 7,
            "c": 10,
        }
    ]

    expected_demand_df = pd.DataFrame([
        {'lag_1_fuzzy_set': LOW_DEMAND, 'lag_2_fuzzy_set': HIGH_DEMAND, 'lag_1_membership': 0.4, 'lag_2_membership': 0},
        {'lag_1_fuzzy_set': HIGH_DEMAND, 'lag_2_fuzzy_set': VERY_HIGH_DEMAND, 'lag_1_membership': 2/3, 'lag_2_membership': 0},
    ])

    actual_demand_df = prepare_fuzzified_forecast_demand_df(demand_df, fuzzy_list, rule_base)

    assert_pd_equal_ignore_order(actual_demand_df, expected_demand_df)


def test_get_linguistic_form():
    rule_base = pd.DataFrame([
        {'lag_1_fuzzy_set': LOW_DEMAND, 'lag_2_fuzzy_set': HIGH_DEMAND, 'demand': 4},
        {'lag_1_fuzzy_set': HIGH_DEMAND, 'lag_2_fuzzy_set': VERY_HIGH_DEMAND, 'demand': 8},
    ])

    lingustic_rules = get_linguistic_form(rule_base)

    expected_lingustic_rules = pd.DataFrame([
        {'linguistic_form': 'IF LAG 1 demand IS LowDemand AND LAG 2 demand IS HighDemand THEN 4'},
        {'linguistic_form': 'IF LAG 1 demand IS HighDemand AND LAG 2 demand IS VeryHighDemand THEN 8'},
    ])

    assert_pd_equal_ignore_order(lingustic_rules, expected_lingustic_rules)