"""
Walk-forward (expanding-window) forecasting pipeline.

All evaluation is performed in a strictly causal manner: at each step t, the
model receives only observations strictly prior to t.  No future data leaks
into any prediction.
"""

import logging
from datetime import timedelta

import pandas as pd

from evaluate.evaluate_model import get_model_score
from models.fuzzy import create_rule_base_df, fuzzy_forecast_pipeline
from models.moving_average import forecast_moving_average
from utilities.utils import get_holidays_df


def forecast_over_horizon(demand_df, start_date, forecast_horizon, lags, forecast_method, method_args):
    """
    Generate one-step-ahead predictions over a contiguous horizon using
    walk-forward validation.

    For each day t in [start_date, start_date + forecast_horizon):
        1. Restrict the available history to observations with date < t.
        2. Take the last 'lags' rows as the input window.
        3. Call forecast_method(window, *method_args) to obtain (value, explanation).
        4. Store the result.

    The method signature (demand_df, *args) -> (float, any) is satisfied by
    both fuzzy_forecast_pipeline and forecast_moving_average, enabling
    model-agnostic use of this loop.

    Parameters
    ----------
    demand_df : pd.DataFrame  Full available history ('date', 'demand').
    start_date : datetime     First day to predict.
    forecast_horizon : int    Number of days to predict.
    lags : int                Input window length L.
    forecast_method : callable
    method_args : list        Additional arguments forwarded to forecast_method.

    Returns
    -------
    pd.DataFrame  Columns: date, prediction, explanation.
    """
    forecast_df = pd.DataFrame()

    for horizon in range(forecast_horizon):
        filter_end_date = start_date + timedelta(days=horizon)

        # Strictly causal: only data before the prediction date is used
        new_df = demand_df[demand_df['date'] < filter_end_date]
        new_df = new_df.sort_values(by=['date'])
        new_df = new_df.tail(lags)

        forecast_value, explanation = forecast_method(new_df, *method_args)

        new_forecast = pd.DataFrame([{
            'date': filter_end_date,
            'prediction': forecast_value,
            'explanation': explanation
        }])
        forecast_df = pd.concat([forecast_df, new_forecast])
        logging.debug(f'Forecasted for: {filter_end_date}')

    return forecast_df


def run_experiment_fuzzy_and_average_model(demand, split_date, lags_to_use, fuzzy_sets):
    """
    Execute the full experiment: train, forecast, and evaluate both models.

    Split design
    ------------
    Training set  : date < split_date
    Test set      : date >= split_date - L  (the L days before split_date are
                    included so the first test prediction has a full input window)
    Evaluation set: date >= split_date + L  (first L test days serve as input only)

    Hungarian public holidays are removed from the test set before evaluation
    because holiday demand is systematically distorted and would inflate error
    metrics independently of model quality.

    Parameters
    ----------
    demand : pd.DataFrame     Full demand series ('date', 'demand').
    split_date : datetime     Train/test boundary.
    lags_to_use : int         Input window length L.
    fuzzy_sets : list[dict]   Fuzzy partition definition.

    Returns
    -------
    tuple  (fuzzy_score: pd.DataFrame, moving_average_score: pd.DataFrame)
           Each DataFrame contains mae, mae_pct, bias, bias_pct, score.
    """
    train_data = demand[demand['date'] < split_date].reset_index(drop=True)

    # Prepend L days before split_date so the first forecast has a full window
    test_data = demand[
        demand['date'] >= (split_date - timedelta(days=lags_to_use))
    ].reset_index(drop=True)[['date', 'demand']]

    rule_base = create_rule_base_df(train_data, lags_to_use, fuzzy_sets)

    holidays_df = get_holidays_df()
    test_data = test_data.merge(
        holidays_df, how='left_anti', left_on='date', right_on='date'
    ).drop(columns=['holiday'])

    validation_start_date = test_data['date'].min() + timedelta(days=lags_to_use)
    forecast_horizon = (test_data['date'].max() - validation_start_date).days + 1

    moving_average_forecast = forecast_over_horizon(
        test_data, validation_start_date, forecast_horizon, lags_to_use,
        forecast_moving_average, [lags_to_use]
    )
    fuzzy_model_forecast = forecast_over_horizon(
        test_data, validation_start_date, forecast_horizon, lags_to_use,
        fuzzy_forecast_pipeline, [fuzzy_sets, rule_base]
    )

    # Trim to the evaluation window (exclude the L warm-up days)
    test_data = test_data[test_data['date'] >= validation_start_date]

    fuzzy_score = get_model_score(test_data, fuzzy_model_forecast)
    moving_average_score = get_model_score(test_data, moving_average_forecast)

    return fuzzy_score, moving_average_score


def explain_forecast(pred_df, date):
    """
    Retrieve the linguistic explanation for a specific forecast date.

    Returns the fired-rule table (IF-THEN strings with firing strengths and
    rule contributions) stored during inference for the given date.

    Parameters
    ----------
    pred_df : pd.DataFrame  Fuzzy forecast output with an 'explanation' column.
    date : str              Date string matching the 'date' column format.

    Returns
    -------
    str | pd.DataFrame  Explanation stored at inference time for that date.
    """
    filtered = pred_df[pred_df["date"] == date]
    return filtered['explanation'].head(1)[0]
