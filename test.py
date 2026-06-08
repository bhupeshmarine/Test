from pyspark.sql import functions as F

if saved_feature_mode == 2:
    for g in saved_max_groups:
        new_df = new_df.withColumn(
            g["output_col"],
            F.greatest(*[F.col(c).cast("double") for c in g["input_cols"]])
        )
