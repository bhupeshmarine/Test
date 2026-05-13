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
