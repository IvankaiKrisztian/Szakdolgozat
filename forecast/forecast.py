import logging
from datetime import timedelta

import pandas as pd

from evaluate.evaluate_model import get_model_score
from models.fuzzy import create_rule_base_df, fuzzy_forecast_pipeline
from models.moving_average import forecast_moving_average
from utilities.utils import get_holidays_df


def forecast_over_horizon(demand_df, start_date, forecast_horizon,lags, forecast_method, method_args):
    forecast_df = pd.DataFrame()
    for horizon in range(forecast_horizon):
        filter_end_date = start_date + timedelta(days=horizon)
        new_df = demand_df[demand_df['date'] < filter_end_date]
        new_df = new_df.sort_values(by=['date'])
        new_df = new_df.tail(lags)
        forecast_value, explanation = forecast_method(new_df, *method_args)
        new_forecast = pd.DataFrame([{'date': filter_end_date, 'prediction': forecast_value, 'explanation': explanation}])
        forecast_df = pd.concat([forecast_df, new_forecast])
        logging.debug(f'Forecasted for: {filter_end_date}')
    return forecast_df


def run_experiment_fuzzy_and_average_model(demand,split_date,lags_to_use,fuzzy_sets):
    train_data = demand[demand['date'] < split_date].reset_index(drop=True)
    test_data = demand[demand['date'] >= (split_date - timedelta(days=lags_to_use))].reset_index(drop=True)[
        ['date', 'demand']]
    rule_base = create_rule_base_df(train_data, lags_to_use, fuzzy_sets)
    holidays_df = get_holidays_df()
    test_data = test_data.merge(holidays_df, how='left_anti', left_on='date', right_on='date').drop(columns=['holiday'])
    validation_start_date = test_data['date'].min() + timedelta(days=lags_to_use)
    forecast_horizon = (test_data['date'].max() - validation_start_date).days + 1
    moving_average_forecast = forecast_over_horizon(test_data, validation_start_date, forecast_horizon, lags_to_use,
                                                    forecast_moving_average, [lags_to_use])
    fuzzy_model_forecast = forecast_over_horizon(test_data, validation_start_date, forecast_horizon, lags_to_use,
                                                 fuzzy_forecast_pipeline, [fuzzy_sets, rule_base])
    test_data = test_data[test_data['date'] >= validation_start_date]
    fuzzy_score = get_model_score(test_data, fuzzy_model_forecast)
    moving_average_score = get_model_score(test_data, moving_average_forecast)
    return fuzzy_score, moving_average_score
