import pandas as pd

# Example: df is your dataset
name_col = "entity_legal_name"

# 1. Create name length column
df["name_length"] = df[name_col].astype(str).str.len()

# 2. Calculate summary statistics
name_length_stats = {
    "min": df["name_length"].min(),
    "p25": df["name_length"].quantile(0.25),
    "p50_median": df["name_length"].quantile(0.50),
    "p75": df["name_length"].quantile(0.75),
    "max": df["name_length"].max(),
    "average": df["name_length"].mean()
}

name_length_stats


name_col = "entity_legal_name"

s = df[name_col].fillna("")

df["has_non_latin"] = s.str.contains(r"[^\x00-\x7F]", regex=True)
df["is_numeric_only"] = s.str.strip().str.fullmatch(r"\d+")
df["has_control_char"] = s.str.contains(r"[\n\r\t]", regex=True)

df["special_char_flag"] = (
    df["has_non_latin"] |
    df["is_numeric_only"] |
    df["has_control_char"]
)

special_character_rate = df["special_char_flag"].mean()

special_character_rate

import pandas as pd

def profile_name_columns(df: pd.DataFrame, name_columns: list) -> pd.DataFrame:
    """
    Cleans selected name columns and returns profiling summary:
    null count, non-null count, blank count, unique count, duplicate count.
    """

    results = []

    for col in name_columns:
        if col not in df.columns:
            results.append({
                "column_name": col,
                "status": "column not found",
                "total_records": len(df),
                "null_count": None,
                "blank_count": None,
                "valid_name_count": None,
                "unique_name_count": None,
                "duplicate_name_count": None,
                "unique_name_rate": None
            })
            continue

        # Original column
        original = df[col]

        # Cleaned column: strip, uppercase, replace multiple spaces with single space
        cleaned = (
            original
            .astype("string")
            .str.strip()
            .str.upper()
            .str.replace(r"\s+", " ", regex=True)
        )

        total_records = len(df)
        null_count = original.isna().sum()
        blank_count = (cleaned == "").sum()

        # Valid names = not null and not blank
        valid_names = cleaned[cleaned.notna() & (cleaned != "")]

        valid_name_count = len(valid_names)
        unique_name_count = valid_names.nunique()
        duplicate_name_count = valid_name_count - unique_name_count
        unique_name_rate = unique_name_count / valid_name_count if valid_name_count > 0 else 0

        results.append({
            "column_name": col,
            "status": "ok",
            "total_records": total_records,
            "null_count": null_count,
            "blank_count": blank_count,
            "valid_name_count": valid_name_count,
            "unique_name_count": unique_name_count,
            "duplicate_name_count": duplicate_name_count,
            "unique_name_rate": round(unique_name_rate, 4)
        })

    return pd.DataFrame(results)

country_col = "country_code"

s = df[country_col].astype("string").str.strip().str.upper()

valid_country_pattern = s.str.fullmatch(r"[A-Z]{2}|[A-Z]{3}")

total_records = len(df)
valid_country_count = valid_country_pattern.fillna(False).sum()
invalid_country_count = total_records - valid_country_count

invalid_country_rate = invalid_country_count / total_records
