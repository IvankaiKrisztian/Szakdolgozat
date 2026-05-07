"""
Model evaluation — Vandeput combined score.

Reference: N. Vandeput, "Inventory Optimization", 2020.
"""

import pandas as pd


def get_model_score(actual_demand: pd.DataFrame, forecasted_demand: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the Vandeput combined forecast accuracy score.

    Error convention (Vandeput): e_t = forecast_t - actual_t
    Positive error = over-forecast; negative error = under-forecast.

    Metrics
    -------
    MAE        = (1/N) * sum |e_t|
    Bias       = (1/N) * sum e_t          (signed; reveals systematic skew)
    MAE%       = MAE / mean(actual)        (scale-free accuracy)
    Bias%      = Bias / mean(actual)       (scale-free bias)
    Score      = MAE% + |Bias%|

    The score jointly penalises dispersion (MAE%) and systematic directional
    error (|Bias%|).  A model that is accurate on average but consistently
    skewed in one direction will be penalised more than one with the same MAE
    and no bias.

    Parameters
    ----------
    actual_demand : pd.DataFrame     Columns: date, demand.
    forecasted_demand : pd.DataFrame Columns: date, prediction.

    Returns
    -------
    pd.DataFrame  Single-row frame with columns: mae, mae_pct, bias, bias_pct, score.
    """
    merged = actual_demand.merge(forecasted_demand, on="date")
    errors = merged["prediction"] - merged["demand"]
    mean_demand = merged["demand"].mean()

    mae = errors.abs().mean()
    bias = errors.mean()
    mae_pct = mae / mean_demand
    bias_pct = bias / mean_demand

    return pd.DataFrame({
        "mae":       [mae],
        "mae_pct":   [mae_pct],
        "bias":      [bias],
        "bias_pct":  [bias_pct],
        "score":     [mae_pct + abs(bias_pct)],
    })
