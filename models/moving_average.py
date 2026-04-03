def forecast_moving_average(demand_df,lags):
    demand_df = demand_df[-lags:]
    return demand_df['demand'].mean()
