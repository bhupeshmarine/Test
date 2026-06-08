import pandas as pd
from pyspark.sql import functions as F
from pyspark.sql.window import Window

# 1. Add row id to your Spark dataframe
w = Window.orderBy("master_party_smun_identifier", "entity_bloomberg_id")

new_output = new_output.withColumn(
    "row_id",
    F.row_number().over(w) - 1
)

# 2. Convert model output arrays into a Spark dataframe
pred_df = spark.createDataFrame(
    pd.DataFrame({
        "row_id": range(len(probs)),
        "prob_match": probs.round(2),
        "predicted_label_binary": preds
    })
)

# 3. Join side-by-side using row_id
new_output = (
    new_output
    .join(pred_df, on="row_id", how="left")
    .drop("row_id")
)






new_output = (
    new_output
    .withColumn("predicted_label", F.when(F.col("predicted_label_binary") == 1, "Y").otherwise("N"))
    .withColumn(
        "bucket",
        F.when(F.col("prob_match") >= EXACT_THRESHOLD, "Exact")
         .when(F.col("prob_match") >= HIGH_THRESHOLD, "High")
         .when(F.col("prob_match") >= MEDIUM_THRESHOLD, "Medium")
         .otherwise("Low")
    )
    .withColumn("model_used", F.lit(saved_display_name))
)
