from typing import Tuple

from pyspark.sql import DataFrame, SparkSession, Window
from pyspark.sql import functions as F


def _require_columns(df: DataFrame, required_cols: list[str], df_name: str) -> None:
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"{df_name} is missing required columns: {missing_cols}")


def _normalize_col(column):
    return F.regexp_replace(
        F.lower(F.trim(column)),
        r"\s+",
        " "
    )


def create_step2_outputs(
    spark: SparkSession,
    validation_ml_output_table: str,
    bbg_table: str,
    top_n: int = 5,
    exact_prob: float = 1.0,
    med_low: float = 0.8,
) -> Tuple[DataFrame, DataFrame]:
    """
    Creates Step-2 outputs after validation ML model.

    Inputs:
        1. validation_ml_output_table
        2. bbg_table

    Outputs:
        1. DF2_1    -> exact 100% ML matches
        2. DF4_FULL -> expanded BBG candidates for less-than-100% matches
    """

    prob_col = "prob_match"

    prdm_id_col = "master_party_smun_identifier"
    party_name_col = "party_legal_name"
    master_party_name_col = "master_party_legal_name"

    bbg_id_col = "entity_bloomberg_id"
    bbg_legal_name_col = "entity_legal_name"
    bbg_lei_name_col = "entity_lei_name"
    bbg_short_name_col = "entity_short_name"

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

    # ------------------------------------------------------------
    # 1. DF2_1: exact 100% matches
    # ------------------------------------------------------------
    DF2_1 = df2.filter(F.col(prob_col) >= F.lit(exact_prob))

    # ------------------------------------------------------------
    # 2. DF3: records between 0.8 and 1.0
    # ------------------------------------------------------------
    DF3 = df2.filter(
        (F.col(prob_col) >= F.lit(med_low))
        & (F.col(prob_col) < F.lit(exact_prob))
    )

    # ------------------------------------------------------------
    # 3. Get unique names from DF3
    # ------------------------------------------------------------
    DF3_NAMES = (
        DF3
        .select(
            F.col(prdm_id_col),
            F.trim(F.col(party_name_col)).alias(party_name_col),
            F.trim(F.col(master_party_name_col)).alias(master_party_name_col),
        )
        .dropDuplicates()
    )

    # ------------------------------------------------------------
    # 4. Normalize BBG names
    # ------------------------------------------------------------
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

    # ------------------------------------------------------------
    # 5. Normalize DF3 names
    # ------------------------------------------------------------
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

    # ------------------------------------------------------------
    # 6. Blocking join against all BBG name columns
    # ------------------------------------------------------------
    join_condition = (
        (F.col("p.party_key3") == F.col("b.bbg_legal_key3"))
        | (F.col("p.party_key3") == F.col("b.bbg_lei_key3"))
        | (F.col("p.party_key3") == F.col("b.bbg_short_key3"))
        | (F.col("p.master_key3") == F.col("b.bbg_legal_key3"))
        | (F.col("p.master_key3") == F.col("b.bbg_lei_key3"))
        | (F.col("p.master_key3") == F.col("b.bbg_short_key3"))
    )

    DF4_MATCH_CANDIDATES = (
        DF3_NAMES_N.alias("p")
        .join(BBG_N.alias("b"), join_condition, "left")
    )

    # ------------------------------------------------------------
    # 7. Calculate six Levenshtein distances
    # ------------------------------------------------------------
    DF4_MATCH_CANDIDATES = (
        DF4_MATCH_CANDIDATES
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
            F.lit(1)
            - (
                F.col("name_distance")
                / F.greatest(
                    F.length(F.col("party_legal_norm")),
                    F.length(F.col("entity_legal_norm")),
                )
            ),
        )
    )

    # ------------------------------------------------------------
    # 8. Rank and keep top N BBG candidates per input name
    # ------------------------------------------------------------
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

    # ------------------------------------------------------------
    # 9. Dedupe DF3, then join back to keep every ML-output column
    # ------------------------------------------------------------
    DF3_DEDUP = DF3.dropDuplicates(
        [prdm_id_col, party_name_col, master_party_name_col]
    )

    DF4_FULL = (
        DF4.alias("m")
        .join(
            DF3_DEDUP.alias("d"),
            (
                (F.col(f"m.{prdm_id_col}") == F.col(f"d.{prdm_id_col}"))
                & (F.col(f"m.{party_name_col}") == F.col(f"d.{party_name_col}"))
                & (
                    F.col(f"m.{master_party_name_col}")
                    == F.col(f"d.{master_party_name_col}")
                )
            ),
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
            F.col("m.dist_party_to_entity_legal"),
            F.col("m.dist_party_to_entity_lei"),
            F.col("m.dist_party_to_entity_short"),
            F.col("m.dist_master_to_entity_legal"),
            F.col("m.dist_master_to_entity_lei"),
            F.col("m.dist_master_to_entity_short"),
            F.col("m.name_sim"),
        )
    )

    return DF2_1, DF4_FULL

from increase_search_space import create_step2_outputs

DF2_1, DF4_FULL = create_step2_outputs(
    spark=spark,
    validation_ml_output_table=validation_ML_output_table,
    bbg_table=BBG_INPUT_TABLE,
)
DF2_1, DF4_FULL = create_step2_outputs(
    spark=spark,
    validation_ml_output_table=validation_ML_output_table,
    bbg_table=BBG_INPUT_TABLE,
    prob_col="prob_match",
    prdm_id_col="master_party_smun_identifier",
    party_name_col="party_legal_name",
    master_party_name_col="master_party_legal_name",
    bbg_id_col="entity_bloomberg_id",
    bbg_legal_name_col="entity_legal_name",
    bbg_lei_name_col="entity_lei_name",
    bbg_short_name_col="entity_short_name",
    top_n=5,
    exact_prob=1.0,
    med_low=0.8
)

def create_step2_outputs(
    spark,
    validation_ml_output_table,
    bbg_table,
    prob_col="prob_match",
    prdm_id_col="master_party_smun_identifier",
    party_name_col="party_legal_name",
    master_party_name_col="master_party_legal_name",
    bbg_id_col="entity_bloomberg_id",
    bbg_legal_name_col="entity_legal_name",
    bbg_lei_name_col="entity_lei_name",
    bbg_short_name_col="entity_short_name",
    top_n=5,
    exact_prob=1.0,
    med_low=0.8,
):
