import operator

import pandas as pd


class TriangularFuzzySet:
    def __init__(self, a, b, c):
        self.a = a
        self.b = b
        self.c = c

    def get_membership_value(self, value):
        if value <= self.a or value >= self.c:
            return 0
        elif self.a < value <= self.b:
            return (
                    (value - self.a)
                    /
                    (self.b - self.a)
            )
        elif self.b < value < self.c:
            return (
                    (self.c - value)
                    /
                    (self.c - self.b)
            )


class ShoulderFuzzySet:
    def __init__(self, a, b, direction):
        self.a = a
        self.b = b
        self.direction = direction

    def get_membership_value(self, value):
        if self.direction == "right":
            if self.b <= value:
                return 1
            if value < self.a:
                return 0
            elif self.a <= value < self.b:
                return (
                        (value - self.a)
                        /
                        (self.b - self.a)
                )
        elif self.direction == "left":
            if self.b < value:
                return 0
            if value <= self.a:
                return 1
            elif self.a < value <= self.b:
                return (
                        (self.b - value)
                        /
                        (self.b - self.a)
                )


def get_all_fuzzy_membership_values(fuzzy_list, value):
    membership_values = {}
    for fuzzy_set in fuzzy_list:
        if fuzzy_set["type"] == "triangular":
            f_set = TriangularFuzzySet(fuzzy_set["a"], fuzzy_set["b"], fuzzy_set["c"])
        elif fuzzy_set["type"] == "shoulder":
            f_set = ShoulderFuzzySet(fuzzy_set["a"], fuzzy_set["b"], fuzzy_set["direction"])
        else:
            raise ValueError(f"Unknown fuzzy set type: {fuzzy_set['type']}")
        membership_values[fuzzy_set["name"]] = f_set.get_membership_value(value)
    return membership_values


def fuzzify_by_set_name(fuzzy_list, value):
    if pd.isna(value):
        return None
    membership_values = get_all_fuzzy_membership_values(fuzzy_list, value)
    return max(membership_values.items(), key=operator.itemgetter(1))[0]

def fuzzify_by_value(fuzzy_list, value):
    if pd.isna(value):
        return None
    membership_values = get_all_fuzzy_membership_values(fuzzy_list, value)
    return max(membership_values.items(), key=operator.itemgetter(1))[1]


def create_fuzzified_lag_df(df, lag_days, fuzzy_sets):
    df = df.copy()
    for i in range(1, lag_days + 1):
        df[f'lag_{i}_demand'] = df['demand'].shift(i)
        df[f'lag_{i}_fuzzy_set'] = df[f'lag_{i}_demand'].apply(lambda value: fuzzify_by_set_name(fuzzy_sets, value))
    return df


def fuzzy_forecast_pipeline(demand_df, fuzzy_list, rule_base):
    prepared_df = prepare_fuzzified_forecast_demand_df(demand_df, fuzzy_list, rule_base)
    return fuzzy_forecast(rule_base, prepared_df)


def fuzzy_forecast(rule_base, fuzzified_demand_df):
    rule_base_cols = list(set(rule_base.columns) - {'demand'})
    pred_df = rule_base.merge(fuzzified_demand_df, on=rule_base_cols, how='inner')
    membership_cols = list(set(pred_df.columns) - set(rule_base_cols) - {'demand'})
    pred_df['firing_strength'] = pred_df[membership_cols].prod(axis=1)
    pred_df['rule_prediction'] = pred_df['firing_strength'] * pred_df['demand']
    total_strengths = pred_df['firing_strength'].sum()
    if total_strengths <= 0:
        return 0
    return pred_df['rule_prediction'].sum() / total_strengths


def prepare_fuzzified_forecast_demand_df(demand_df, fuzzy_list, rule_base):
    lags = len(rule_base.columns) - 1
    lagged_df = demand_df.copy()
    lagged_df['lag_1'] = lagged_df['demand']
    for i in range(2, lags + 1):
        lagged_df[f"lag_{i}"] = lagged_df['demand'].shift(i - 1)
    lagged_df.dropna(inplace=True)
    lagged_df = lagged_df.tail(1)
    lagged_with_sets = lagged_df.drop(columns=['demand', 'date']).merge(rule_base, how='cross')
    return get_lags_membership_values(lagged_with_sets, lags, fuzzy_list).drop(columns=['demand'])


def get_lags_membership_values(lagged_with_sets, lags, fuzzy_list):
    for i in range(1, lags + 1):
        def get_membership(row, lag_index=i):
            matching_sets = [fs for fs in fuzzy_list if fs['name'] == row[f"lag_{lag_index}_fuzzy_set"]]
            return fuzzify_by_value(matching_sets, row[f"lag_{lag_index}"])

        lagged_with_sets[f"lag_{i}_membership"] = lagged_with_sets.apply(get_membership, axis=1)
        lagged_with_sets = lagged_with_sets.drop(columns=[f"lag_{i}"])
    return lagged_with_sets
