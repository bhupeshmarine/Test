import pandas as pd
from pyspark.sql import functions as F

pred_df = spark.createDataFrame(
    pd.DataFrame({
        "row_id": list(range(len(probs))),
        "prob_match": probs.round(2),
        "predicted_label_binary": preds
    })
)

new_output = new_output.join(pred_df, on="row_id", how="left")

new_output = (
    new_output
    .withColumn(
        "predicted_label",
        F.when(F.col("predicted_label_binary") == 1, F.lit("Y"))
         .otherwise(F.lit("N"))
    )
    .withColumn(
        "bucket",
        F.when(F.col("prob_match") >= EXACT_THRESHOLD, F.lit("Exact"))
         .when(F.col("prob_match") >= HIGH_THRESHOLD, F.lit("High"))
         .when(F.col("prob_match") >= MEDIUM_THRESHOLD, F.lit("Medium"))
         .otherwise(F.lit("Low"))
    )
    .withColumn("model_used", F.lit(saved_display_name))
)
