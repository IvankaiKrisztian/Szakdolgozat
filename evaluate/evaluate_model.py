import pandas as pd


def get_model_score(actual_demand: pd.DataFrame, forecasted_demand: pd.DataFrame) -> pd.DataFrame:
    merged = actual_demand.merge(forecasted_demand, on="date")
    mae = (merged["demand"] - merged["forecast"]).abs().mean()
    bias = (merged["demand"] - merged["forecast"]).sum()
    return pd.DataFrame({
        "mae":   [mae],
        "bias":  [bias],
        "score": [mae + abs(bias)],
    })
