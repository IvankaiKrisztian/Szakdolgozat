from datetime import timedelta

import pandas as pd

CONTEXT_WINDOW_DAYS = 10


def forecast_over_horizon(demand_df, start_date, forecast_horizon, forecast_method, method_args):
    forecast_df = pd.DataFrame()
    for horizon in range(forecast_horizon):

        filter_end_date = start_date + timedelta(days=horizon)
        filter_start_date = filter_end_date - timedelta(days=CONTEXT_WINDOW_DAYS)
        new_df = demand_df[demand_df['date'] < filter_end_date]
        new_df = new_df[new_df['date'] >= filter_start_date]
        forecast_value = forecast_method(new_df, *method_args)
        new_forecast = pd.DataFrame([{'date': filter_end_date, 'prediction': forecast_value}])
        print('Forecast for date' + str(filter_end_date))
        forecast_df = pd.concat([forecast_df, new_forecast])
    return forecast_df
