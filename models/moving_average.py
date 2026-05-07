"""
7-day simple moving average — benchmark model.

Used as the naive baseline against which the fuzzy model is evaluated.
The t+1 prediction is the unweighted mean of the L most recent real demand values.
"""


def forecast_moving_average(demand_df, lags):
    """
    Predict demand at t+1 as the arithmetic mean of the previous L observations.

    Parameters
    ----------
    demand_df : pd.DataFrame  Historical demand; must have at least 'lags' rows
                              and a 'demand' column.
    lags : int                Window length L (typically 7).

    Returns
    -------
    tuple  (mean_demand: float, window_df: pd.DataFrame)
           The second element is the L-row window used, returned for symmetry
           with the fuzzy pipeline signature.
    """
    demand_df = demand_df[-lags:]
    return demand_df['demand'].mean(), demand_df
