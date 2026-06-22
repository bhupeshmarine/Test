from typing import Tuple, Optional
from pyspark.sql import DataFrame, SparkSession, Window
from pyspark.sql import functions as F


def _check_columns(df: DataFrame, required_cols: list[str], df_name: str) -> None:
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"{df_name} is missing columns: {missing}")


def _normalize_name(col: str):
    return F.lower(
        F.regexp_replace(
            F.trim(F.col(col)),
            r"[^a-zA-Z0-9 ]",
            ""
        )
    )


def create_step2_tables(
    spark: SparkSession,
    validation_ml_output_table: str,
    bbg_input_table: str,
    step2_all_100_table: Optional[str] = None,
    step2_less_100_table: Optional[str] = None,
    prob_col: str = "prob_match",
    prdm_id_col: str = "master_party_smun_identifier",
    prdm_name_col: str = "party_legal_name",
    bbg_id_col: str = "entity_bloomberg_id",
    bbg_name_col: str = "entity_name",
    threshold: float = 0.80,
    top_n: int = 5,
    save_tables: bool = True,
) -> Tuple[DataFrame, DataFrame]:
    """
    Creates two Step-2 output tables:

    1. step2_all_100_df:
       Records where ML model probability is exactly 1.0.

    2. step2_less_100_df:
       Records where ML model probability is >= threshold and < 1.0,
       expanded with additional BBG candidate records using name similarity.

    Parameters
    ----------
    spark : SparkSession
        Active Spark session.

    validation_ml_output_table : str
        Table containing ML output with probability column.

    bbg_input_table : str
        BBG source/master table used to increase search space.

    step2_all_100_table : str, optional
        Output table name for 100% matches.

    step2_less_100_table : str, optional
        Output table name for expanded candidates.

    Returns
    -------
    Tuple[DataFrame, DataFrame]
        step2_all_100_df, step2_less_100_df
    """

    ml_df = spark.table(validation_ml_output_table)
    bbg_df = spark.table(bbg_input_table)

    _check_columns(
        ml_df,
        [prob_col, prdm_id_col, prdm_name_col],
        "validation_ml_output_table"
    )

    _check_columns(
        bbg_df,
        [bbg_id_col, bbg_name_col],
        "bbg_input_table"
    )

    # 1. Exact 100% match table
    step2_all_100_df = ml_df.filter(F.col(prob_col) == F.lit(1.0))

    # 2. Less than 100%, but still high-confidence records
    less_100_df = ml_df.filter(
        (F.col(prob_col) >= F.lit(threshold)) &
        (F.col(prob_col) < F.lit(1.0))
    )

    # Normalize PRDM names
    less_100_clean = (
        less_100_df
        .withColumn("prdm_clean_name", _normalize_name(prdm_name_col))
        .withColumn("blocking_key", F.substring(F.col("prdm_clean_name"), 1, 5))
    )

    # Normalize BBG names
    bbg_clean = (
        bbg_df
        .withColumn("bbg_clean_name", _normalize_name(bbg_name_col))
        .withColumn("blocking_key", F.substring(F.col("bbg_clean_name"), 1, 5))
    )

    # Join on blocking key to avoid full cross join
    candidate_df = (
        less_100_clean.alias("p")
        .join(
            bbg_clean.alias("b"),
            on="blocking_key",
            how="left"
        )
        .withColumn(
            "name_distance",
            F.levenshtein(
                F.col("p.prdm_clean_name"),
                F.col("b.bbg_clean_name")
            )
        )
    )

    window_spec = Window.partitionBy(
        F.col(f"p.{prdm_id_col}")
    ).orderBy(
        F.col("name_distance").asc(),
        F.col(f"p.{prob_col}").desc()
    )

    step2_less_100_df = (
        candidate_df
        .withColumn("candidate_rank", F.row_number().over(window_spec))
        .filter(F.col("candidate_rank") <= top_n)
        .drop("prdm_clean_name", "bbg_clean_name", "blocking_key")
    )

    if save_tables:
        if not step2_all_100_table or not step2_less_100_table:
            raise ValueError(
                "Output table names are required when save_tables=True."
            )

        (
            step2_all_100_df
            .write
            .mode("overwrite")
            .option("overwriteSchema", "true")
            .saveAsTable(step2_all_100_table)
        )

        (
            step2_less_100_df
            .write
            .mode("overwrite")
            .option("overwriteSchema", "true")
            .saveAsTable(step2_less_100_table)
        )

    return step2_all_100_df, step2_less_100_df



from increase_search_space import create_step2_tables

step2_all_100_df, step2_less_100_df = create_step2_tables(
    spark=spark,
    validation_ml_output_table=validation_ML_output_table,
    bbg_input_table=BBG_INPUT_TABLE,
    step2_all_100_table=step2_all_100_table,
    step2_less_100_table=step2_less_100_table,
    prob_col="prob_match",
    prdm_id_col="master_party_smun_identifier",
    prdm_name_col="party_legal_name",
    bbg_id_col="entity_bloomberg_id",
    bbg_name_col="entity_name",
    threshold=0.80,
    top_n=5,
    save_tables=True
)
