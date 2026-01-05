import polars as pl

def get_model_score(actual_demand: pl.DataFrame, forecasted_demand: pl.DataFrame)-> pl.DataFrame:
    return (
        actual_demand
        .join(forecasted_demand, on="date")
        .with_columns((pl.col("demand") - pl.col("forecast")).abs().alias("absolute_error"),
                      (pl.col("demand") - pl.col("forecast")).alias("bias"),
                      pl.lit(1).alias("group_by_col"))
        .group_by("group_by_col").agg([pl.mean("absolute_error").alias("mae"), pl.sum("bias").alias("bias")])
        .select("mae", "bias",(pl.col("mae") + pl.col("bias")).alias("score"))
    )
