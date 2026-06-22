DF4_FULL = (
    DF4.alias("m")
    .join(
        DF3_DEDUP.alias("d"),
        F.col(f"m.{prdm_id_col}") == F.col(f"d.{prdm_id_col}"),
        "inner",
    )
