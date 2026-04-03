from datetime import timedelta

import pandas as pd


def forecast_over_horizon(demand_df, start_date, h, forecast_method,args):
    forecast_df = pd.DataFrame()
    for horizon in range(h):
        filter_start_date = start_date - timedelta(days=8)
        filter_end_date = start_date + timedelta(days=horizon)
        new_df = demand_df[demand_df['date'] < filter_end_date]
        new_df = new_df[new_df['date'] >= filter_start_date]
        forecast_value = forecast_method(new_df,*args)
        print(f"Preidction finished for:{filter_end_date}")
        new_forecast = pd.DataFrame([{'date': filter_end_date, 'prediction': forecast_value}])
        forecast_df = pd.concat([forecast_df,new_forecast])
    return forecast_df