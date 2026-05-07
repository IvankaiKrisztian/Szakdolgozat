"""
Fuzzy time series demand forecasting model.

Architecture
------------
Training  : Fuzzify each historical demand value to its dominant set (argmax μ),
            construct L-lagged antecedent-consequent pairs, then group identical
            antecedent patterns and average their crisp consequents into a rule base.

Inference : For each input vector, compute full membership degrees across all sets,
            compute per-rule firing strength as the product of antecedent memberships
            (Mamdani conjunction), and return the firing-strength-weighted average of
            consequent values (centre-of-gravity defuzzification over discrete rules).
            If no rule fires, fall back to the global mean of all rule consequents.
"""

import logging
import operator

import numpy as np
import pandas as pd

global no_rules_fired


# ---------------------------------------------------------------------------
# Membership function classes
# ---------------------------------------------------------------------------

class TriangularFuzzySet:
    """
    Triangular membership function defined by support endpoints (a, c) and
    modal value b.

        μ(x) = 0                        if x ≤ a or x ≥ c
        μ(x) = (x - a) / (b - a)       if a < x ≤ b
        μ(x) = (c - x) / (c - b)       if b < x < c

    Parameters
    ----------
    a : float  Left foot (μ = 0).
    b : float  Apex (μ = 1).
    c : float  Right foot (μ = 0).
    """

    def __init__(self, a, b, c):
        self.a = a
        self.b = b
        self.c = c

    def get_membership_value(self, value):
        if value <= self.a or value >= self.c:
            return 0
        elif self.a < value <= self.b:
            return (value - self.a) / (self.b - self.a)
        elif self.b < value < self.c:
            return (self.c - value) / (self.c - self.b)


class ShoulderFuzzySet:
    """
    Shoulder (open) membership function for boundary sets that must saturate
    at 1 beyond a threshold — avoids truncating extreme demand values.

    Right shoulder (μ = 1 for x ≥ b):
        μ(x) = 0                        if x < a
        μ(x) = (x - a) / (b - a)       if a ≤ x < b
        μ(x) = 1                        if x ≥ b

    Left shoulder (μ = 1 for x ≤ a):
        μ(x) = 1                        if x ≤ a
        μ(x) = (b - x) / (b - a)       if a < x ≤ b
        μ(x) = 0                        if x > b

    Parameters
    ----------
    a : float  Transition start.
    b : float  Transition end (saturation boundary).
    direction : str  "left" or "right".
    """

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
                return (value - self.a) / (self.b - self.a)
        elif self.direction == "left":
            if self.b < value:
                return 0
            if value <= self.a:
                return 1
            elif self.a < value <= self.b:
                return (self.b - value) / (self.b - self.a)


# ---------------------------------------------------------------------------
# Membership computation utilities
# ---------------------------------------------------------------------------

def get_all_fuzzy_membership_values(fuzzy_list, value):
    """
    Compute μ_i(value) for every set A_i in the partition.

    Parameters
    ----------
    fuzzy_list : list[dict]  Set definitions with keys: name, type, a, b,
                             and c (triangular) or direction (shoulder).
    value : float

    Returns
    -------
    dict  {set_name: μ_i(value)}
    """
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
    """
    Fuzzify a crisp value by argmax: return the name of the set with the
    highest membership degree.  Ties are broken by list order.
    Returns None for NaN inputs (missing demand is not assigned a set).
    """
    if pd.isna(value):
        return None
    membership_values = get_all_fuzzy_membership_values(fuzzy_list, value)
    return max(membership_values.items(), key=operator.itemgetter(1))[0]


def get_highest_membership_fuzzy_set_membership_value(fuzzy_list, value):
    """
    Return max_i μ_i(value) — the degree of the dominant set membership.
    Returns None for NaN inputs.
    """
    if pd.isna(value):
        return None
    membership_values = get_all_fuzzy_membership_values(fuzzy_list, value)
    return max(membership_values.items(), key=operator.itemgetter(1))[1]


# ---------------------------------------------------------------------------
# Fuzzification strategies
# ---------------------------------------------------------------------------

def get_fuzzification_method(fuzzification_type):
    """
    Select the fuzzification strategy used during rule base construction.

    "highest"
        Each lag value maps to exactly one set (argmax).  The rule antecedent
        is a single crisp label per lag.  Simpler rule space, fewer rules.

    "all"
        Each lag value maps to every set for which μ > 0, duplicating the row
        once per non-zero membership.  Enriches the antecedent space at the
        cost of a larger (and more redundant) rule base.
    """
    if fuzzification_type == 'highest':
        return get_one_lag_highest_fuzzification
    elif fuzzification_type == 'all':
        return get_one_lag_all_fuzzification
    else:
        raise ValueError(f"Unknown fuzzification method: {fuzzification_type}")


def get_one_lag_highest_fuzzification(demand_df, fuzzy_sets, col_name):
    """
    Assign each value in col_name to its dominant fuzzy set (argmax μ).
    Appends a '{col_name}_fuzzy_set' column.
    """
    demand_df[f'{col_name}_fuzzy_set'] = demand_df[col_name].apply(
        lambda value: get_highest_membership_fuzzy_set_name(fuzzy_sets, value)
    )
    return demand_df


def get_one_lag_all_fuzzification(demand_df, fuzzy_sets, col_name):
    """
    NOT_USED
    For each set A_i with μ_i(x) > 0, retain a copy of the row labelled A_i.
    Rows with zero membership in a set are discarded for that set.
    The result is the union of per-set filtered DataFrames, so one input row
    can produce up to |fuzzy_sets| output rows.
    """
    all_fuzzy_sets = []
    for fuzzy_set in fuzzy_sets:
        input_f_list = [fuzzy_set]

        def get_membership(row, fset=input_f_list):
            return get_highest_membership_fuzzy_set_membership_value(fset, row[col_name])

        new_df = demand_df.copy()
        new_df[f'{col_name}_fuzzy_set'] = new_df[col_name].apply(
            lambda value: get_highest_membership_fuzzy_set_name(input_f_list, value)
        )
        new_df[f"{col_name}_membership"] = new_df.apply(get_membership, axis=1)
        new_df = new_df[0 < new_df[f"{col_name}_membership"]]
        new_df.drop(columns=[f"{col_name}_membership"], inplace=True)
        all_fuzzy_sets.append(new_df)
    return pd.concat(all_fuzzy_sets)


# ---------------------------------------------------------------------------
# Rule base construction (training)
# ---------------------------------------------------------------------------

def create_rule_base_df(df, lag_days, fuzzy_sets, fuzzification_type='highest'):
    """
    Build the fuzzy rule base from historical demand.

    Algorithm
    ---------
    1. Shift 'demand' to create L lagged columns: lag_1 (t-1) … lag_L (t-L).
    2. Fuzzify each lag column using the selected strategy.
    3. Group all rows that share the same antecedent pattern
       (lag_1_fuzzy_set, …, lag_L_fuzzy_set) and average their crisp
       consequent values — this is the Sung-Chiang rule aggregation method.

    The resulting rule base has one row per unique L-tuple of fuzzy set names,
    with 'demand' holding the averaged crisp consequent.

    Parameters
    ----------
    df : pd.DataFrame         Training data with a 'demand' column.
    lag_days : int            Number of lags L.
    fuzzy_sets : list[dict]   Fuzzy partition definition.
    fuzzification_type : str  "highest" (default) or "all".

    Returns
    -------
    pd.DataFrame  Columns: lag_1_fuzzy_set … lag_L_fuzzy_set, demand.
    """
    fuzzification = get_fuzzification_method(fuzzification_type)
    for i in range(1, lag_days + 1):
        df[f'lag_{i}'] = df['demand'].shift(i)
        df = fuzzification(df, fuzzy_sets, f'lag_{i}')

    fuzz_cols = [f'lag_{i}_fuzzy_set' for i in range(1, lag_days + 1)]
    return (
        df
        .dropna()
        .groupby(fuzz_cols)['demand'].mean()
        .reset_index()
    )


# ---------------------------------------------------------------------------
# Inference pipeline
# ---------------------------------------------------------------------------

def fuzzy_forecast_pipeline(demand_df, fuzzy_sets, rule_base):
    """
    Full single-step inference: fuzzify the most recent L observations,
    match against the rule base, and defuzzify.

    Parameters
    ----------
    demand_df : pd.DataFrame  Last L rows of real demand ('date', 'demand').
    fuzzy_sets : list[dict]   Fuzzy partition.
    rule_base : pd.DataFrame  Output of create_rule_base_df.

    Returns
    -------
    tuple  (prediction: float, explanation: str | pd.DataFrame)
    """
    prepared_df = prepare_fuzzified_forecast_demand_df(demand_df, fuzzy_sets, rule_base)
    return fuzzy_forecast(rule_base, prepared_df, fuzzy_sets)


def fuzzy_forecast(rule_base, fuzzified_demand_df, fuzzy_sets):
    """
    Mamdani inference with centre-of-gravity defuzzification over a discrete
    rule base.

    Firing strength for rule r:
        w_r = prod_{i=1}^{L} mu_{A_i^r}(x_i)

    where A_i^r is the fuzzy set assigned to lag i in rule r, and x_i is the
    i-th lag of the current input.  The product implements fuzzy conjunction.

    Defuzzified output:
        y_hat = sum_r (w_r * y_r) / sum_r w_r

    where y_r is the crisp consequent (averaged training output) of rule r.

    Fallback: if sum_r w_r = 0 (no rule fires), return the global mean of all
    rule consequents — equivalent to a zero-order Takagi-Sugeno model with a
    single constant rule.

    Parameters
    ----------
    rule_base : pd.DataFrame
    fuzzified_demand_df : pd.DataFrame  Output of prepare_fuzzified_forecast_demand_df.
    fuzzy_sets : list[dict]

    Returns
    -------
    tuple  (prediction: float, explanation: str | pd.DataFrame)
    """
    rule_base_cols = list(set(rule_base.columns) - {'demand'})
    pred_df = rule_base.merge(fuzzified_demand_df, on=rule_base_cols, how='inner')

    membership_cols = list(set(pred_df.columns) - set(rule_base_cols) - {'demand'})
    # Product conjunction across all antecedent memberships
    pred_df['firing_strength'] = pred_df[membership_cols].prod(axis=1)
    pred_df['rule_prediction'] = pred_df['firing_strength'] * pred_df['demand']

    explanation = get_linguistic_form(
        pred_df[pred_df['firing_strength'] > 0], len(rule_base_cols), 'prediction'
    )

    mean_demand = rule_base['demand'].mean()
    total_strengths = pred_df['firing_strength'].sum()

    if total_strengths <= 0:
        logging.debug(f"Rule not fired!")
        return mean_demand, "Rule not fired!"

    return pred_df['rule_prediction'].sum() / total_strengths, explanation


def prepare_fuzzified_forecast_demand_df(demand_df, fuzzy_list, rule_base):
    """
    Construct the fuzzified input vector for inference.

    Takes the most recent observation (last row of demand_df) and builds the
    L-lagged input with full membership degrees, cross-joined against the rule
    base so that each rule can be evaluated simultaneously.

    lag_1 corresponds to the most recent demand value (t),
    lag_i corresponds to t-(i-1) for i > 1.

    Parameters
    ----------
    demand_df : pd.DataFrame  Last L rows of real demand.
    fuzzy_list : list[dict]
    rule_base : pd.DataFrame  Used only to infer L from column count.

    Returns
    -------
    pd.DataFrame  Rows: one per (rule, input combination). Columns: lag antecedent
                  set names and per-lag membership values.
    """
    lags = len(rule_base.columns) - 1
    lagged_df = demand_df.copy()
    lagged_df['lag_1'] = lagged_df['demand']
    for i in range(2, lags + 1):
        lagged_df[f"lag_{i}"] = lagged_df['demand'].shift(i - 1)

    last_date = lagged_df['date'].max()
    lagged_df = lagged_df[lagged_df['date'] == last_date]

    # Cross join: pair the single input row with every rule in the base
    lagged_with_sets = lagged_df.drop(columns=['demand', 'date']).merge(rule_base, how='cross')
    return get_lags_membership_values(lagged_with_sets, lags, fuzzy_list).drop(columns=['demand'])


def get_lags_membership_values(lagged_with_sets, lags, fuzzy_list):
    """
    For each lag i, evaluate mu_{A_i^r}(x_i) — the membership of the i-th input
    value in the set prescribed by rule r's antecedent.

    This is what makes the inference "all-membership": instead of the crisp
    argmax used during training, each input value is evaluated against its
    rule-assigned set to obtain a continuous firing degree.

    Drops the raw lag value columns after computing degrees.
    """
    for i in range(1, lags + 1):
        def get_membership(row, lag_index=i):
            matching_sets = [
                fs for fs in fuzzy_list if fs['name'] == row[f"lag_{lag_index}_fuzzy_set"]
            ]
            return get_highest_membership_fuzzy_set_membership_value(
                matching_sets, row[f"lag_{lag_index}"]
            )

        lagged_with_sets[f"lag_{i}_membership"] = lagged_with_sets.apply(get_membership, axis=1)
        lagged_with_sets = lagged_with_sets.drop(columns=[f"lag_{i}"])
    return lagged_with_sets


# ---------------------------------------------------------------------------
# Explainability
# ---------------------------------------------------------------------------

def get_linguistic_form(rule_base, lags=7, form='rule_base'):
    """
    Serialise rules as readable IF-THEN strings for inspection and reporting.

    Example output:
        "IF LAG 1 demand IS MediumDemand AND LAG 2 demand IS HighDemand
         AND ... THEN 72.5"

    Parameters
    ----------
    rule_base : pd.DataFrame
    lags : int
    form : str  "rule_base" -> return only the linguistic_form column;
                otherwise -> return the full DataFrame with all columns appended.

    Returns
    -------
    pd.DataFrame
    """
    lingustic_form = rule_base.copy()
    for i in range(lags):
        lag = i + 1
        if lag == 1:
            lingustic_form["linguistic_form"] = (
                "IF LAG 1 demand IS " + lingustic_form["lag_1_fuzzy_set"]
            )
        else:
            lingustic_form["linguistic_form"] = (
                lingustic_form["linguistic_form"]
                + f" AND LAG {lag} demand IS "
                + lingustic_form[f"lag_{lag}_fuzzy_set"]
            )

    lingustic_form["demand_as_string"] = lingustic_form["demand"].astype("str")
    lingustic_form["linguistic_form"] = (
        lingustic_form["linguistic_form"] + " THEN " + lingustic_form["demand_as_string"]
    )

    if form == 'rule_base':
        return lingustic_form[["linguistic_form"]]
    else:
        return lingustic_form


def fallback_to_midpoints(rule_base, fuzzy_sets):
    """
    Alternative fallback (not currently active — superseded by global mean).

    Computes the mean of the modal values (b for triangular, midpoint of
    transition interval for shoulder sets) across all antecedent sets present
    in the input vector.  Included for reference.
    """
    f_sets_df = pd.DataFrame(fuzzy_sets)
    result = (
        rule_base
        .melt(var_name="lag_name", value_name="fuzzy_set_name")
        .merge(f_sets_df, how='left', left_on='fuzzy_set_name', right_on='name')
    )
    result['midpoint'] = np.where(
        result['type'] == 'shoulder',
        (result['a'] + result['b']) / 2,
        result['b']
    )
    return result['midpoint'].mean()
