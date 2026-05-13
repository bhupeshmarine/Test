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
