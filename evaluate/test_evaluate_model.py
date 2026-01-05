import polars as pl

from evaluate.evaluate_model import get_model_score


def test_evaluate_model():
    actual_demand = pl.DataFrame(
        [
            ("2025-01-01", 1),
            ("2025-01-02", 0),
            ("2025-01-03", 7),
            ("2025-01-04", 2),
            ("2025-01-05", 4),
        ],
        ["date","demand"]
    )

    forecasted_demand = pl.DataFrame(
        [
            ("2025-01-01", 0), # 1
            ("2025-01-02", 1), # -1
            ("2025-01-03", 0), # 7
            ("2025-01-04", 7), # -5
            ("2025-01-05", 2), # 2
        ],
        ["date","forecast"]
    )

    result = get_model_score(actual_demand, forecasted_demand)

    expected_result = pl.DataFrame(
        [
            (3.2, 4, 7.2)
        ],
        ["mae","bias","score"]
    )

    assert result.equals(expected_result)

