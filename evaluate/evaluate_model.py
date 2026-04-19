import pandas as pd


def get_model_score(actual_demand: pd.DataFrame, forecasted_demand: pd.DataFrame) -> pd.DataFrame:
    merged = actual_demand.merge(forecasted_demand, on="date")
    errors = merged["demand"] - merged["forecast"]
    mae = errors.abs().mean()
    bias = errors.mean()
    return pd.DataFrame({
        "mae":   [mae],
        "bias":  [bias],
        "score": [mae + abs(bias)],
    })
