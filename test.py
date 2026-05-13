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




import pandas as pd

def profile_country_code_format(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    results = []

    for col in columns:
        s = df[col].astype("string").str.strip().str.upper()

        total_records = len(df)
        null_count = df[col].isna().sum()
        blank_count = (s == "").sum()

        valid_values = s[s.notna() & (s != "")]
        unique_count = valid_values.nunique()

        valid_pattern = valid_values.str.fullmatch(r"[A-Z]{2}|[A-Z]{3}")

        valid_pattern_count = valid_pattern.fillna(False).sum()
        invalid_pattern_count = len(valid_values) - valid_pattern_count

        invalid_pattern_rate = (
            invalid_pattern_count / len(valid_values)
            if len(valid_values) > 0
            else 0
        )

        results.append({
            "column_name": col,
            "total_records": total_records,
            "null_count": null_count,
            "blank_count": blank_count,
            "valid_value_count": len(valid_values),
            "unique_count": unique_count,
            "valid_pattern_count": valid_pattern_count,
            "invalid_pattern_count": invalid_pattern_count,
            "invalid_pattern_rate": round(invalid_pattern_rate, 4)
        })

    return pd.DataFrame(results)



import pandas as pd

def profile_country_consistency(df: pd.DataFrame, country_columns: list) -> pd.DataFrame:
    """
    Checks whether multiple country fields are same within each record.
    Example: principal_country, head_office_country, legal_country
    """

    # Clean country columns
    cleaned = (
        df[country_columns]
        .astype("string")
        .apply(lambda col: col.str.strip().str.upper().str.replace(r"\s+", " ", regex=True))
    )

    # Replace blank strings with NA
    cleaned = cleaned.replace("", pd.NA)

    # Number of available country values per row
    available_country_count = cleaned.notna().sum(axis=1)

    # Number of unique country values per row
    unique_country_count = cleaned.nunique(axis=1, dropna=True)

    # All same = at least 2 country values available and only 1 unique country
    all_same_flag = (available_country_count >= 2) & (unique_country_count == 1)

    # Not all same = at least 2 country values available and more than 1 unique country
    not_all_same_flag = (available_country_count >= 2) & (unique_country_count > 1)

    comparable_records = (available_country_count >= 2).sum()

    result = {
        "total_records": len(df),
        "comparable_records": comparable_records,
        "all_same_count": all_same_flag.sum(),
        "not_all_same_count": not_all_same_flag.sum(),
        "all_same_rate": round(all_same_flag.sum() / comparable_records, 4) if comparable_records > 0 else 0,
        "not_all_same_rate": round(not_all_same_flag.sum() / comparable_records, 4) if comparable_records > 0 else 0,
        "records_with_less_than_2_country_values": (available_country_count < 2).sum()
    }

    return pd.DataFrame([result])



import pandas as pd

def profile_industry_code_patterns(df: pd.DataFrame, code_columns: list) -> pd.DataFrame:
    """
    Profiles industry code columns for nulls, unique values, and expected pattern validity.
    """

    pattern_map = {
        "party_sic_code": r"^SIC-\d{4}$",
        "party_smicx_code": r"^SMICS-\d{8}$",
        "naics_code": r"^NAICSUS-\d{6}$",
        "nace_code": r"^NACE-\d{2}\.\d{2}$"
    }

    results = []

    for col in code_columns:
        s = df[col].astype("string").str.strip().str.upper()

        total_records = len(df)
        null_count = df[col].isna().sum()
        blank_count = (s == "").sum()

        valid_values = s[s.notna() & (s != "")]
        valid_value_count = len(valid_values)
        unique_count = valid_values.nunique()

        pattern = pattern_map[col]

        valid_pattern = valid_values.str.fullmatch(pattern)

        valid_pattern_count = valid_pattern.fillna(False).sum()
        invalid_pattern_count = valid_value_count - valid_pattern_count

        invalid_pattern_rate = (
            invalid_pattern_count / valid_value_count
            if valid_value_count > 0
            else 0
        )

        results.append({
            "column_name": col,
            "expected_pattern": pattern,
            "total_records": total_records,
            "null_count": null_count,
            "blank_count": blank_count,
            "valid_value_count": valid_value_count,
            "unique_count": unique_count,
            "valid_pattern_count": valid_pattern_count,
            "invalid_pattern_count": invalid_pattern_count,
            "invalid_pattern_rate": round(invalid_pattern_rate, 4)
        })

    return pd.DataFrame(results)
