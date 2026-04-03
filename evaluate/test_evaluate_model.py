import pandas as pd

from evaluate.evaluate_model import get_model_score


def test_evaluate_model():
    actual_demand = pd.DataFrame({
        "date":   ["2025-01-01", "2025-01-02", "2025-01-03", "2025-01-04", "2025-01-05"],
        "demand": [1, 0, 7, 2, 4],
    })

    forecasted_demand = pd.DataFrame({
        "date":     ["2025-01-01", "2025-01-02", "2025-01-03", "2025-01-04", "2025-01-05"],
        "forecast": [0, 1, 0, 7, 2],  # errors: 1, -1, 7, -5, 2
    })

    result = get_model_score(actual_demand, forecasted_demand)

    expected = pd.DataFrame({
        "mae":   [3.2],
        "bias":  [4.0],
        "score": [7.2],
    })

    pd.testing.assert_frame_equal(result, expected, check_dtype=False)
