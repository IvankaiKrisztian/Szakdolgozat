import pandas as pd

from evaluate.evaluate_model import get_model_score


def test_evaluate_model():
    actual_demand = pd.DataFrame({
        "date":   ["2025-01-01", "2025-01-02", "2025-01-03", "2025-01-04", "2025-01-05"],
        "demand": [1, 0, 7, 2, 4],
    })

    forecasted_demand = pd.DataFrame({
        "date":     ["2025-01-01", "2025-01-02", "2025-01-03", "2025-01-04", "2025-01-05"],
        "prediction": [0, 1, 0, 7, 2],  # errors (forecast-demand): -1, 1, -7, 5, -2
    })

    result = get_model_score(actual_demand, forecasted_demand)

    # mean_demand = 2.8, mae = 3.2, bias = -0.8
    expected = pd.DataFrame({
        "mae":      [3.2],
        "mae_pct":  [3.2 / 2.8],
        "bias":     [-0.8],
        "bias_pct": [-0.8 / 2.8],
        "score":    [3.2 / 2.8 + 0.8 / 2.8],
    })

    pd.testing.assert_frame_equal(result, expected, check_dtype=False)
