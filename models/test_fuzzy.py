import pandas as pd

from models.fuzzy import TriangularFuzzySet, ShoulderFuzzySet, fuzzy_forecast, \
    prepare_fuzzified_forecast_demand_df, get_highest_membership_fuzzy_set_name, get_linguistic_form, \
    create_rule_base_df, \
    get_one_lag_all_fuzzification, fallback_to_midpoints
from utilities.utils import parse_date, assert_pd_equal_ignore_order, show_full_pd_df

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
    assert get_highest_membership_fuzzy_set_name(fuzzy_list, demand_value) == "VeryLowDemand"


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
    fuzzy_sets = [
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

    firing_strengths = (.3 * .4) + (.2 * .7)
    first_rule = (.3 * .4) * 4
    second_rule = (.2 * .7) * 8

    expected_forecast = (
            (first_rule + second_rule)
            /
            firing_strengths
    )

    actual_forecast,explanation = fuzzy_forecast(rule_base, fuzzified_demand_df,fuzzy_sets)
    explanation.to_csv('explain.csv')

    assert actual_forecast == expected_forecast


def test_fuzzy_forecast_no_rule_firing():
    rule_base = pd.DataFrame([
        {'lag_1_fuzzy_set': LOW_DEMAND, 'lag_2_fuzzy_set': HIGH_DEMAND, 'demand': 4},
        {'lag_1_fuzzy_set': HIGH_DEMAND, 'lag_2_fuzzy_set': VERY_HIGH_DEMAND, 'demand': 8},
    ])
    fuzzy_sets = [
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

    actual_forecast,explanation = fuzzy_forecast(rule_base, fuzzified_demand_df,fuzzy_sets)
    assert actual_forecast == 5.166666666666667




def test_prepare_fuzzified_forecast_demand_df():
    demand_df = pd.DataFrame([
        {'date': parse_date('2026-02-03'), 'demand': 2},
        {'date': parse_date('2026-02-04'), 'demand': 4},
    ])

    rule_base = pd.DataFrame([
        {'lag_1_fuzzy_set': LOW_DEMAND, 'lag_2_fuzzy_set': HIGH_DEMAND, 'demand': 4},
        {'lag_1_fuzzy_set': HIGH_DEMAND, 'lag_2_fuzzy_set': VERY_HIGH_DEMAND, 'demand': 8},
    ])

    fuzzy_sets = [
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

    actual_demand_df = prepare_fuzzified_forecast_demand_df(demand_df, fuzzy_sets, rule_base)

    assert_pd_equal_ignore_order(actual_demand_df, expected_demand_df)


def test_get_linguistic_form():
    rule_base = pd.DataFrame([
        {'lag_1_fuzzy_set': LOW_DEMAND, 'lag_2_fuzzy_set': HIGH_DEMAND, 'demand': 4},
        {'lag_1_fuzzy_set': HIGH_DEMAND, 'lag_2_fuzzy_set': VERY_HIGH_DEMAND, 'demand': 8},
    ])

    lingustic_rules = get_linguistic_form(rule_base,2)

    expected_lingustic_rules = pd.DataFrame([
        {'linguistic_form': 'IF LAG 1 demand IS LowDemand AND LAG 2 demand IS HighDemand THEN 4'},
        {'linguistic_form': 'IF LAG 1 demand IS HighDemand AND LAG 2 demand IS VeryHighDemand THEN 8'},
    ])

    assert_pd_equal_ignore_order(lingustic_rules, expected_lingustic_rules)


def test_create_all_rule_base_df():
    demand_df = pd.DataFrame([
        {'date': parse_date('2026-02-03'), 'demand': 5},
        {'date': parse_date('2026-02-04'), 'demand': 4},
    ])
    lag_days = 1

    fuzzy_sets = [
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
    fuzzification_type = 'all'

    actual_df = create_rule_base_df(demand_df, lag_days, fuzzy_sets,fuzzification_type)


    expected_df = pd.DataFrame([
        {'lag_1_fuzzy_set': 'HighDemand'    ,'demand': 4.0,},
        {'lag_1_fuzzy_set': 'LowDemand'     ,'demand': 4.0,},
        {'lag_1_fuzzy_set': 'VeryHighDemand','demand': 4.0,},
    ])
    show_full_pd_df()
    print(actual_df.head(10))

    assert_pd_equal_ignore_order(actual_df, expected_df)





def test_get_one_lag_fuzzification():
    demand_df = pd.DataFrame([
        {'date': parse_date('2026-02-04'), 'demand': 4, 'lag_1': 3},
    ])

    fuzzy_sets = [
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
    col_name = "lag_1"

    actual_df = get_one_lag_all_fuzzification(demand_df, fuzzy_sets, col_name)

    expected_df = pd.DataFrame([
        {'date': parse_date('2026-02-04'), 'demand': 4,'lag_1': 3,'lag_1_fuzzy_set': 'HighDemand'},
        {'date': parse_date('2026-02-04'), 'demand': 4,'lag_1': 3,'lag_1_fuzzy_set': 'LowDemand'},
    ])

    print(actual_df.head(10))

    assert_pd_equal_ignore_order(actual_df, expected_df)





def test_fallback_to_midpoints():
    rule_base = pd.DataFrame([
        {'lag_1_fuzzy_set': LOW_DEMAND, 'lag_2_fuzzy_set': HIGH_DEMAND, 'maci':'laci','gecko':'hecko'},
        {'lag_1_fuzzy_set': VERY_HIGH_DEMAND, 'lag_2_fuzzy_set': HIGH_DEMAND, 'maci': 'laci', 'gecko': 'hecko'},
    ])

    fuzzy_sets = [
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

    actual = fallback_to_midpoints(rule_base, fuzzy_sets)
    print(actual)
    shoulder_midpoint = (1+6)/2
    assert actual == (shoulder_midpoint+5+7+5)/4
