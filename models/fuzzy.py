import logging
import operator

import numpy as np
import pandas as pd
global no_rules_fired


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


def get_highest_membership_fuzzy_set_name(fuzzy_list, value):
    if pd.isna(value):
        return None
    membership_values = get_all_fuzzy_membership_values(fuzzy_list, value)
    return max(membership_values.items(), key=operator.itemgetter(1))[0]

def get_highest_membership_fuzzy_set_membership_value(fuzzy_list, value):
    if pd.isna(value):
        return None
    membership_values = get_all_fuzzy_membership_values(fuzzy_list, value)
    return max(membership_values.items(), key=operator.itemgetter(1))[1]


def get_fuzzification_method(fuzzification_type):
    if fuzzification_type == 'highest':
        return get_one_lag_highest_fuzzification
    elif fuzzification_type == 'all':
        return get_one_lag_all_fuzzification
    else:
        raise ValueError(f"Unknown fuzzification method: {fuzzification_type}")


def create_rule_base_df(df, lag_days, fuzzy_sets,fuzzification_type='highest'):
    fuzzification = get_fuzzification_method(fuzzification_type)
    for i in range(1, lag_days + 1):
        df[f'lag_{i}'] = df['demand'].shift(i)
        df = fuzzification(df, fuzzy_sets, f'lag_{i}')
    fuzz_cols = [f'lag_{i}_fuzzy_set' for i in range(1, lag_days+1)]
    return  (
        df
        .dropna()
        .groupby(fuzz_cols)
        ['demand'].mean()
        .reset_index()
    )

def get_one_lag_highest_fuzzification(demand_df, fuzzy_sets, col_name):
    demand_df[f'{col_name}_fuzzy_set'] = demand_df[col_name].apply(lambda value: get_highest_membership_fuzzy_set_name(fuzzy_sets, value))
    return demand_df

def get_one_lag_all_fuzzification(demand_df, fuzzy_sets, col_name):
    all_fuzzy_sets = []
    for fuzzy_set in fuzzy_sets:
        input_f_list = [fuzzy_set]
        def get_membership(row, fset=input_f_list):
            return get_highest_membership_fuzzy_set_membership_value(fset, row[col_name])
        new_df = demand_df.copy()
        new_df[f'{col_name}_fuzzy_set'] = new_df[col_name].apply(lambda value: get_highest_membership_fuzzy_set_name(input_f_list, value))
        new_df[f"{col_name}_membership"] = new_df.apply(get_membership, axis=1)
        new_df = new_df[0 < new_df[f"{col_name}_membership"]]
        new_df.drop(columns=[f"{col_name}_membership"],inplace=True)
        all_fuzzy_sets.append(new_df)
    return pd.concat(all_fuzzy_sets)

def fuzzy_forecast_pipeline(demand_df, fuzzy_sets, rule_base):
    prepared_df = prepare_fuzzified_forecast_demand_df(demand_df, fuzzy_sets, rule_base)
    return fuzzy_forecast(rule_base, prepared_df,fuzzy_sets)


def fuzzy_forecast(rule_base, fuzzified_demand_df,fuzzy_sets):
    rule_base_cols = list(set(rule_base.columns) - {'demand'})
    pred_df = rule_base.merge(fuzzified_demand_df, on=rule_base_cols, how='inner')
    membership_cols = list(set(pred_df.columns) - set(rule_base_cols) - {'demand'})
    pred_df['firing_strength'] = pred_df[membership_cols].prod(axis=1)
    pred_df['rule_prediction'] = pred_df['firing_strength'] * pred_df['demand']
    explanation = get_linguistic_form(pred_df[0<pred_df['firing_strength']],len(rule_base_cols),'prediction')
    mean_demand = rule_base['demand'].mean()
    total_strengths = pred_df['firing_strength'].sum()
    if total_strengths <= 0:
        logging.debug(f"Rule not fired!")
        return mean_demand, f"Rule not fired!" #fallback_to_midpoints(fuzzified_demand_df,fuzzy_sets), mean_demand
    return pred_df['rule_prediction'].sum() / total_strengths, explanation


def prepare_fuzzified_forecast_demand_df(demand_df, fuzzy_list, rule_base):
    lags = len(rule_base.columns) - 1
    lagged_df = demand_df.copy()
    lagged_df['lag_1'] = lagged_df['demand']
    for i in range(2, lags + 1):
        lagged_df[f"lag_{i}"] = lagged_df['demand'].shift(i - 1)
    last_date = lagged_df['date'].max()
    lagged_df = lagged_df[lagged_df['date'] == last_date]
    lagged_with_sets = lagged_df.drop(columns=['demand', 'date']).merge(rule_base, how='cross')
    return get_lags_membership_values(lagged_with_sets, lags, fuzzy_list).drop(columns=['demand'])


def get_lags_membership_values(lagged_with_sets, lags, fuzzy_list):
    for i in range(1, lags + 1):
        def get_membership(row, lag_index=i):
            matching_sets = [fs for fs in fuzzy_list if fs['name'] == row[f"lag_{lag_index}_fuzzy_set"]]
            return get_highest_membership_fuzzy_set_membership_value(matching_sets, row[f"lag_{lag_index}"])

        lagged_with_sets[f"lag_{i}_membership"] = lagged_with_sets.apply(get_membership, axis=1)
        lagged_with_sets = lagged_with_sets.drop(columns=[f"lag_{i}"])
    return lagged_with_sets


def get_linguistic_form(rule_base,lags=7,form='rule_base'):
    lingustic_form = rule_base.copy()
    for i in range(lags):
        lag = i+1
        if lag == 1:
            lingustic_form[f"linguistic_form"] = "IF LAG 1 demand IS " + lingustic_form[f"lag_1_fuzzy_set"]
        else:
            lingustic_form[f"linguistic_form"] = lingustic_form[f"linguistic_form"] + f" AND LAG {lag} demand IS " + lingustic_form[f"lag_{lag}_fuzzy_set"]
    lingustic_form["demand_as_string"] = lingustic_form["demand"].astype("str")
    lingustic_form[f"linguistic_form"] = lingustic_form[f"linguistic_form"] + " THEN " + lingustic_form[f"demand_as_string"]
    if form == 'rule_base':
        return lingustic_form[[f"linguistic_form"]]
    else:
        return lingustic_form


def fallback_to_midpoints(rule_base, fuzzy_sets):
    f_sets_df = pd.DataFrame(fuzzy_sets)
    result = rule_base.melt(var_name="lag_name",value_name="fuzzy_set_name").merge(f_sets_df, how='left', left_on='fuzzy_set_name', right_on='name')
    result['midpoint'] = np.where(result['type'] =='shoulder', (result['a'] + result['b'])/2, result['b'])
    return result['midpoint'].mean()