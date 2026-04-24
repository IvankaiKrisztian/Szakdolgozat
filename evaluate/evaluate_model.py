import pandas as pd


def get_model_score(actual_demand: pd.DataFrame, forecasted_demand: pd.DataFrame) -> pd.DataFrame:
    merged = actual_demand.merge(forecasted_demand, on="date")
    # Error defined as forecast - demand (positive = over-forecast), per Vandeput
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
