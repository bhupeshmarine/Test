from typing import List, Optional
from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as F


def create_final_bucket_output(
    df_val: DataFrame,
    df_mandar: DataFrame,
    id_col: str = "master_party_smun_identifier",
    prob_col: str = "prob_match",
    name_score_col: str = "name_max",
    city_score_col: str = "city_max",
    street_score_col: str = "street_max",
    ai_tp_threshold: float = 0.75,
    confirmed_name_threshold: int = 100,
    confirmed_city_threshold: int = 80,
    confirmed_street_threshold: int = 55,
    disputed_street_threshold: int = 70,
    maybe_name_threshold: int = 85,
    bbg_id_col: str = "entity_bloomberg_id",
    entity_address_col: str = "entity_address",
    final_bucket_col: str = "final_bucket",
    output_bucket_col: str = "match_bucket_final",
    final_columns: Optional[List[str]] = None,
) -> DataFrame:

    w = Window.partitionBy(id_col).orderBy(
        F.desc(prob_col),
        F.desc(name_score_col)
    )

    df_val_unique = (
        df_val
        .withColumn("rn", F.row_number().over(w))
        .filter(F.col("rn") == 1)
        .drop("rn")
    )

    df_mandar_renamed = df_mandar.select(
        F.col(id_col),
        *[
            F.col(c).alias(f"{c}_mandar")
            for c in df_mandar.columns
            if c != id_col
        ]
    )

    bbg_id_mandar_col = f"{bbg_id_col}_mandar"

    df_joined = df_val_unique.join(
        df_mandar_renamed,
        on=id_col,
        how="left"
    )

    df_joined = df_joined.withColumn(
        "AI_TP_Flag",
        F.when(F.col(prob_col) >= F.lit(ai_tp_threshold), "Yes").otherwise("No")
    )

    df_joined = df_joined.withColumn(
        final_bucket_col,
        F.when(
            (F.col("AI_TP_Flag") == "Yes") &
            (F.col(name_score_col) == confirmed_name_threshold) &
            (F.col(city_score_col) >= confirmed_city_threshold) &
            (F.col(street_score_col) >= confirmed_street_threshold) &
            F.col(bbg_id_col).isNotNull() &
            F.col(entity_address_col).isNotNull() &
            (F.col(bbg_id_col) == F.col(bbg_id_mandar_col)),
            "Confirmed"
        )
        .when(
            (F.col("AI_TP_Flag") == "Yes") &
            (F.col(name_score_col) >= maybe_name_threshold) &
            (F.col(city_score_col) >= confirmed_city_threshold) &
            (F.col(street_score_col) < confirmed_street_threshold) &
            F.col(bbg_id_col).isNotNull() &
            F.col(entity_address_col).isNotNull() &
            (F.col(bbg_id_col) == F.col(bbg_id_mandar_col)),
            "Confirmed(Should be part of Maybe)"
        )
        .when(
            (F.col("AI_TP_Flag") == "Yes") &
            (F.col(name_score_col) == confirmed_name_threshold) &
            (F.col(street_score_col) >= disputed_street_threshold) &
            F.col(bbg_id_col).isNotNull() &
            F.col(bbg_id_mandar_col).isNotNull() &
            F.col(entity_address_col).isNotNull() &
            (F.col(bbg_id_col) != F.col(bbg_id_mandar_col)),
            "Disputed"
        )
        .when(
            (F.col("AI_TP_Flag") == "Yes") &
            (F.col(name_score_col) == confirmed_name_threshold) &
            (F.col(street_score_col) >= disputed_street_threshold) &
            F.col(bbg_id_mandar_col).isNull() &
            F.col(entity_address_col).isNotNull(),
            "Additional"
        )
        .otherwise("Not Found")
    )

    df_final = df_joined.withColumn(
        output_bucket_col,
        F.when(
            (F.col(final_bucket_col) == "Not Found") &
            (F.col(name_score_col) == confirmed_name_threshold) &
            F.col(entity_address_col).isNotNull(),
            "Maybe"
        )
        .when(
            F.col(final_bucket_col) == "Confirmed(Should be part of Maybe)",
            "Maybe"
        )
        .otherwise(F.col(final_bucket_col))
    )

    if final_columns is None:
        final_columns = [
            id_col,
            bbg_id_col,
            bbg_id_mandar_col,
            output_bucket_col,
            prob_col,
            name_score_col,
            street_score_col,
            city_score_col,
            "state_max",
            "postal_code_max",
            "predicted_label_binary",
            "predicted_label",
            "model_used",
            "entity_legal_name",
            "entity_lei_name",
            "entity_short_name",
            "master_party_legal_name",
            "party_legal_name",
            entity_address_col,
            "party_head_office_address",
            "party_principal_address",
            "party_legal_address",
            "AI_TP_Flag",
            final_bucket_col,
        ]

    final_columns = [c for c in final_columns if c in df_final.columns]

    return df_final.select(*final_columns)


df_final_selected = create_final_bucket_output(
    df_val=df_val,
    df_mandar=df_mandar,
    id_col="master_party_smun_identifier",
    prob_col="prob_match",
    name_score_col="name_max",
    city_score_col="city_max",
    street_score_col="street_max",
    ai_tp_threshold=0.75,
    confirmed_name_threshold=100,
    confirmed_city_threshold=80,
    confirmed_street_threshold=55,
    disputed_street_threshold=70,
    maybe_name_threshold=85,
    bbg_id_col="entity_bloomberg_id",
    entity_address_col="entity_address"
)
