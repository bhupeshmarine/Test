from pyspark.sql import functions as F

new_output = (
    new_output
    .withColumn("prob_match", F.round(F.col("prob_match"), 2))
    .withColumn("predicted_label_binary", F.when(F.col("prob_match") >= 0.5, F.lit(1)).otherwise(F.lit(0)))
    .withColumn("predicted_label", F.when(F.col("predicted_label_binary") == 1, F.lit("Y")).otherwise(F.lit("N")))
    .withColumn(
        "bucket",
        F.when(F.col("prob_match") >= EXACT_THRESHOLD, F.lit("Exact"))
         .when(F.col("prob_match") >= HIGH_THRESHOLD, F.lit("High"))
         .when(F.col("prob_match") >= MEDIUM_THRESHOLD, F.lit("Medium"))
         .otherwise(F.lit("Low"))
    )
    .withColumn("model_used", F.lit(saved_display_name))
)
