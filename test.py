from typing import Tuple, List
from pyspark.sql import DataFrame, SparkSession, Window
from pyspark.sql import functions as F


def _require_columns(df: DataFrame, required_cols: List[str], df_name: str) -> None:
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"{df_name} is missing required columns: {missing}")


def _normalize_col(col):
    return F.regexp_replace(F.lower(F.trim(col)), r"\s+", " ")


# ---------------------------------------------------------------------
# FUNCTION 1:
# Creates DF2_1 and DF4_FULL
# ---------------------------------------------------------------------
def create_step2_outputs(
    spark: SparkSession,
    validation_ml_output_table: str,
    bbg_table: str,
    prob_col: str = "prob_match",
    prdm_id_col: str = "master_party_smun_identifier",
    party_name_col: str = "party_legal_name",
    master_party_name_col: str = "master_party_legal_name",
    bbg_id_col: str = "entity_bloomberg_id",
    bbg_legal_name_col: str = "entity_legal_name",
    bbg_lei_name_col: str = "entity_lei_name",
    bbg_short_name_col: str = "entity_short_name",
    top_n: int = 5,
    exact_prob: float = 1.0,
    med_low: float = 0.8,
) -> Tuple[DataFrame, DataFrame]:

    df2 = spark.table(validation_ml_output_table).cache()
    bbg_raw = spark.table(bbg_table)

    _require_columns(
        df2,
        [prob_col, prdm_id_col, party_name_col, master_party_name_col],
        "validation_ml_output_table",
    )

    _require_columns(
        bbg_raw,
        [bbg_id_col, bbg_legal_name_col, bbg_lei_name_col, bbg_short_name_col],
        "bbg_table",
    )

    # 100% matches
    DF2_1 = df2.filter(F.col(prob_col) >= F.lit(exact_prob))

    # Less than 100%, above threshold
    DF3 = df2.filter(
        (F.col(prob_col) >= F.lit(med_low)) &
        (F.col(prob_col) < F.lit(exact_prob))
    )

    DF3_NAMES = (
        DF3
        .select(
            F.col(prdm_id_col),
            F.trim(F.col(party_name_col)).alias(party_name_col),
            F.trim(F.col(master_party_name_col)).alias(master_party_name_col),
        )
        .dropDuplicates()
    )

    BBG = (
        bbg_raw
        .select(
            F.col(bbg_id_col),
            F.trim(F.col(bbg_legal_name_col)).alias(bbg_legal_name_col),
            F.trim(F.col(bbg_lei_name_col)).alias(bbg_lei_name_col),
            F.trim(F.col(bbg_short_name_col)).alias(bbg_short_name_col),
        )
        .dropDuplicates()
        .withColumn("entity_legal_norm", _normalize_col(F.col(bbg_legal_name_col)))
        .withColumn("entity_lei_norm", _normalize_col(F.col(bbg_lei_name_col)))
        .withColumn("entity_short_norm", _normalize_col(F.col(bbg_short_name_col)))
    )

    DF3_NAMES_N = (
        DF3_NAMES
        .withColumn("party_legal_norm", _normalize_col(F.col(party_name_col)))
        .withColumn("master_party_norm", _normalize_col(F.col(master_party_name_col)))
        .withColumn("party_key3", F.substring(F.col("party_legal_norm"), 1, 3))
        .withColumn("master_key3", F.substring(F.col("master_party_norm"), 1, 3))
    )

    BBG_N = (
        BBG
        .withColumn("bbg_legal_key3", F.substring(F.col("entity_legal_norm"), 1, 3))
        .withColumn("bbg_lei_key3", F.substring(F.col("entity_lei_norm"), 1, 3))
        .withColumn("bbg_short_key3", F.substring(F.col("entity_short_norm"), 1, 3))
    )

    join_condition = (
        (F.col("p.party_key3") == F.col("b.bbg_legal_key3")) |
        (F.col("p.party_key3") == F.col("b.bbg_lei_key3")) |
        (F.col("p.party_key3") == F.col("b.bbg_short_key3")) |
        (F.col("p.master_key3") == F.col("b.bbg_legal_key3")) |
        (F.col("p.master_key3") == F.col("b.bbg_lei_key3")) |
        (F.col("p.master_key3") == F.col("b.bbg_short_key3"))
    )

    DF4_MATCH_CANDIDATES = (
        DF3_NAMES_N.alias("p")
        .join(BBG_N.alias("b"), join_condition, "left")
        .withColumn(
            "dist_party_to_entity_legal",
            F.levenshtein(F.col("party_legal_norm"), F.col("entity_legal_norm")),
        )
        .withColumn(
            "dist_party_to_entity_lei",
            F.levenshtein(F.col("party_legal_norm"), F.col("entity_lei_norm")),
        )
        .withColumn(
            "dist_party_to_entity_short",
            F.levenshtein(F.col("party_legal_norm"), F.col("entity_short_norm")),
        )
        .withColumn(
            "dist_master_to_entity_legal",
            F.levenshtein(F.col("master_party_norm"), F.col("entity_legal_norm")),
        )
        .withColumn(
            "dist_master_to_entity_lei",
            F.levenshtein(F.col("master_party_norm"), F.col("entity_lei_norm")),
        )
        .withColumn(
            "dist_master_to_entity_short",
            F.levenshtein(F.col("master_party_norm"), F.col("entity_short_norm")),
        )
        .withColumn(
            "closest_distance",
            F.least(
                F.col("dist_party_to_entity_legal"),
                F.col("dist_party_to_entity_lei"),
                F.col("dist_party_to_entity_short"),
                F.col("dist_master_to_entity_legal"),
                F.col("dist_master_to_entity_lei"),
                F.col("dist_master_to_entity_short"),
            ),
        )
        .withColumn(
            "name_distance",
            F.levenshtein(F.col("party_legal_norm"), F.col("entity_legal_norm")),
        )
        .withColumn(
            "name_sim",
            F.lit(1) - (
                F.col("name_distance") /
                F.greatest(
                    F.length(F.col("party_legal_norm")),
                    F.length(F.col("entity_legal_norm")),
                )
            ),
        )
    )

    w = (
        Window
        .partitionBy(prdm_id_col, party_name_col, master_party_name_col)
        .orderBy(
            F.col("closest_distance").asc(),
            F.col("name_sim").desc(),
            F.col(bbg_legal_name_col).asc(),
        )
    )

    DF4 = (
        DF4_MATCH_CANDIDATES
        .withColumn("match_rank", F.row_number().over(w))
        .filter(F.col("match_rank") <= F.lit(top_n))
    )

    DF3_DEDUP = DF3.dropDuplicates([prdm_id_col])

    DF4_FULL = (
        DF4.alias("m")
        .join(
            DF3_DEDUP.alias("d"),
            F.col(f"m.{prdm_id_col}") == F.col(f"d.{prdm_id_col}"),
            "inner",
        )
        .select(
            F.col("d.*"),
            F.col(f"m.{bbg_id_col}").alias("candidate_bbg_id"),
            F.col(f"m.{bbg_legal_name_col}").alias("candidate_entity_legal_name"),
            F.col(f"m.{bbg_lei_name_col}").alias("candidate_entity_lei_name"),
            F.col(f"m.{bbg_short_name_col}").alias("candidate_entity_short_name"),
            F.col("m.closest_distance"),
            F.col("m.match_rank"),
            F.col("m.name_sim"),
        )
    )

    return DF2_1, DF4_FULL


# ---------------------------------------------------------------------
# FUNCTION 2:
# Creates final bucketed output
# ---------------------------------------------------------------------
def create_final_bucket_output(
    df_val: DataFrame,
    df_mandar: DataFrame,
    id_col: str = "master_party_smun_identifier",
) -> DataFrame:

    _require_columns(
        df_val,
        [id_col, "prob_match", "name_max"],
        "df_val",
    )

    _require_columns(
        df_mandar,
        [id_col],
        "df_mandar",
    )

    w = Window.partitionBy(id_col).orderBy(
        F.desc("prob_match"),
        F.desc("name_max"),
    )

    df_val_unique = (
        df_val
        .withColumn("rn", F.row_number().over(w))
        .filter(F.col("rn") == 1)
        .drop("rn")
    )

    mandar_cols = [c for c in df_mandar.columns if c != id_col]

    df_mandar_renamed = df_mandar.select(
        F.col(id_col),
        *[F.col(c).alias(f"{c}_mandar") for c in mandar_cols],
    )

    df_joined = df_val_unique.join(
        df_mandar_renamed,
        on=id_col,
        how="left",
    )

    df_joined = df_joined.drop(
        *[c for c in df_joined.columns if "row_id" in c.lower()]
    )

    if "entity_bloomberg_id_mandar" in df_joined.columns:
        df_joined = df_joined.withColumn(
            "entity_bloomberg_id_mandar",
            F.when(
                F.col("entity_bloomberg_id_mandar") == "null",
                None,
            ).otherwise(F.col("entity_bloomberg_id_mandar")),
        )

    df_joined = df_joined.withColumn(
        "AI_TP_Flag",
        F.when(F.col("prob_match").cast("double") >= 0.75, "Yes").otherwise("No"),
    )

    df_joined = df_joined.withColumn(
        "final_bucket",
        F.when(
            (F.col("AI_TP_Flag") == "Yes") &
            (F.col("name_max") == 100) &
            (F.col("city_max") >= 80) &
            (F.col("street_max") >= 55) &
            F.col("entity_bloomberg_id").isNotNull() &
            F.col("entity_address").isNotNull() &
            (F.col("entity_bloomberg_id") == F.col("entity_bloomberg_id_mandar")),
            "Confirmed",
        )
        .when(
            (F.col("AI_TP_Flag") == "Yes") &
            (F.col("name_max") >= 85) &
            (F.col("city_max") >= 80) &
            (F.col("street_max") < 55) &
            F.col("entity_bloomberg_id").isNotNull() &
            F.col("entity_address").isNotNull() &
            (F.col("entity_bloomberg_id") == F.col("entity_bloomberg_id_mandar")),
            "Confirmed(Should be part of Maybe)",
        )
        .when(
            (F.col("AI_TP_Flag") == "Yes") &
            (F.col("name_max") == 100) &
            (F.col("street_max") >= 70) &
            F.col("entity_bloomberg_id").isNotNull() &
            F.col("entity_bloomberg_id_mandar").isNotNull() &
            F.col("entity_address").isNotNull() &
            (F.col("entity_bloomberg_id") != F.col("entity_bloomberg_id_mandar")),
            "Disputed",
        )
        .when(
            (F.col("AI_TP_Flag") == "Yes") &
            (F.col("name_max") == 100) &
            (F.col("street_max") >= 70) &
            F.col("entity_bloomberg_id_mandar").isNull() &
            F.col("entity_address").isNotNull() &
            (F.col("entity_bloomberg_id") != "Not Found"),
            "Additional",
        )
        .otherwise("Not Found"),
    )

    df_not_found = df_joined.filter(F.col("final_bucket") == "Not Found")
    df_other = df_joined.filter(F.col("final_bucket") != "Not Found")

    df_not_found_base = df_not_found.select(
        id_col,
        "final_bucket",
        "AI_TP_Flag",
    )

    df_val_clean = df_val.drop(
        *[c for c in df_val.columns if "row_id" in c.lower()]
    )

    df_not_found_enriched = df_not_found_base.join(
        df_val_clean,
        on=id_col,
        how="left",
    )

    w2 = Window.partitionBy(id_col).orderBy(F.desc("name_max"))

    df_not_found_top = (
        df_not_found_enriched
        .withColumn("rn", F.row_number().over(w2))
        .filter(F.col("rn") == 1)
        .drop("rn")
    )

    df_not_found_top_final = df_not_found_top.join(
        df_mandar_renamed,
        on=id_col,
        how="left",
    )

    df_final = df_other.unionByName(df_not_found_top_final, allowMissingColumns=True)

    def extract_suffix(col):
        return F.regexp_extract(
            F.lower(col),
            r"(llc|inc|co ltd|ltd|kk|corp|corporation|company|lp)[\s\.,]*$",
            1,
        )

    suffix_cols = [
        ("entity_legal_name", "suffix_entity_legal_name"),
        ("entity_lei_name", "suffix_entity_lei_name"),
        ("entity_short_name", "suffix_entity_short_name"),
        ("master_party_legal_name", "suffix_master_party_legal_name"),
        ("party_legal_name", "suffix_party_legal_name"),
    ]

    for src, tgt in suffix_cols:
        if src in df_final.columns:
            df_final = df_final.withColumn(tgt, extract_suffix(F.col(src)))
        else:
            df_final = df_final.withColumn(tgt, F.lit(""))

    for _, tgt in suffix_cols:
        df_final = df_final.withColumn(
            tgt,
            F.when(F.col(tgt) == "corp", "corporation").otherwise(F.col(tgt)),
        )

    df_final = df_final.withColumn(
        "Legal_suffix_mismatch",
        F.when(
            (
                (F.col("suffix_entity_legal_name") == F.col("suffix_master_party_legal_name")) |
                (F.col("suffix_entity_legal_name") == F.col("suffix_party_legal_name")) |
                (F.col("suffix_entity_lei_name") == F.col("suffix_master_party_legal_name")) |
                (F.col("suffix_entity_lei_name") == F.col("suffix_party_legal_name")) |
                (F.col("suffix_entity_short_name") == F.col("suffix_master_party_legal_name")) |
                (F.col("suffix_entity_short_name") == F.col("suffix_party_legal_name"))
            ),
            F.lit(False),
        ).otherwise(F.lit(True)),
    )

    df_final = df_final.withColumn(
        "bucket_updated",
        F.when(
            (F.col("final_bucket") == "Not Found") &
            (F.col("name_max") == 100) &
            F.col("entity_address").isNotNull(),
            "Maybe",
        ).otherwise(F.col("final_bucket")),
    )

    df_final = df_final.withColumn(
        "bucket_updated",
        F.when(
            F.col("bucket_updated") == "Confirmed(Should be part of Maybe)",
            "Maybe",
        ).otherwise(F.col("bucket_updated")),
    )

    if "bucket" in df_final.columns:
        df_final = df_final.withColumnRenamed("bucket", "prob_bucket")

    df_final = df_final.withColumnRenamed("bucket_updated", "match_bucket_final")

    desired_columns = [
        id_col,
        "entity_bloomberg_id",
        "entity_bloomberg_id_mandar",
        "match_bucket_final",
        "prob_match",
        "prob_bucket",
        "name_max",
        "street_max",
        "city_max",
        "state_max",
        "postal_code_max",
        "predicted_label_binary",
        "predicted_label",
        "model_used",
        "Composite_bbg_legal_name",
        "entity_legal_name",
        "entity_lei_name",
        "entity_short_name",
        "master_party_legal_name",
        "party_legal_name",
        "Composite_prdm_legal_name",
        "entity_address",
        "bbg_legal_State",
        "bbg_legal_ZipCode",
        "bbg_legal_city",
        "bbg_legal_country",
        "bbg_legal_street_address",
        "party_head_office_address",
        "party_principal_address",
        "party_legal_address",
        "master_party_legal_name_mandar",
        "party_smun_identifier_mandar",
        "bloomberg_cleaned_legal_name_mandar",
        "AI_TP_Flag",
        "Legal_suffix_mismatch",
    ]

    final_columns = [c for c in desired_columns if c in df_final.columns]

    return df_final.select(*final_columns)


from entity_resolution_postprocessing import (
    create_step2_outputs,
    create_final_bucket_output,
)

DF2_1, DF4_FULL = create_step2_outputs(
    spark=spark,
    validation_ml_output_table=validation_ML_output_table,
    bbg_table=BBG_INPUT_TABLE,
)

df_mandar = spark.table(existing_matches_mandar_table)

df_final_selected = create_final_bucket_output(
    df_val=DF4_FULL,
    df_mandar=df_mandar,
)
