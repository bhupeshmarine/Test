# Databricks notebook source
# MAGIC %md
# MAGIC # Entity Resolution Agentic AI Framework
# MAGIC This file converts the existing PRDM-BBG entity-resolution notebook into an agent-style pipeline.
# MAGIC It preserves the same high-level logic: read data -> standardize -> create candidates -> create fuzzy features -> validation output -> write table.

# COMMAND ----------

# =========================
# 0. Imports and Config
# =========================

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any
import re
import numpy as np
import pandas as pd

try:
    from rapidfuzz import fuzz, process
except Exception:
    from fuzzywuzzy import fuzz, process  # fallback if rapidfuzz is not installed

try:
    from pyspark.sql import SparkSession
    spark  # noqa: F821
except Exception:
    spark = None


@dataclass
class ERConfig:
    input_type: str = "raw"  # raw or combined
    zip_match: str = "no"    # yes or no
    read_from_table: str = "no"
    write_to_table: str = "yes"

    output_table: str = "mrd_ered_dev.ered_bronze.ML_input_features_test_484121"

    bbg_path: str = "/Workspace/Shared/Global_Entity_Resolution/EDA Notebook/Composite_EDA_BBGvsPRD/data/BBG_USA_484121.csv"
    prd_path: str = "/Workspace/Shared/Global_Entity_Resolution/EDA Notebook/Composite_EDA_BBGvsPRD/data/PRDM_US_484121.csv"
    combined_path: str = "/Workspace/Shared/Global_Entity_Resolution/ML Approach/Training data/Labelled_Data/Labelled_data_1195_input.csv"
    combined_excel_path: str = "/Workspace/Shared/Global_Entity_Resolution/ML Approach/Codes/Processing_code/training_data_before_features.xlsx"
    mna_table: str = "mrd_ered_dev.ered_bronze.mna_prdm_bbg"

    bbg_pk: str = "entity_bloomberg_id"
    prdm_pk: str = "master_party_smun_identifier"

    chunk_size: int = 100000


# COMMAND ----------

# =========================
# 1. Tool Layer
# =========================

class DataTools:
    @staticmethod
    def read_csv(path: str, **kwargs) -> pd.DataFrame:
        return pd.read_csv(path, **kwargs)

    @staticmethod
    def read_excel(path: str, sheet_name: str) -> pd.DataFrame:
        return pd.read_excel(path, sheet_name=sheet_name)

    @staticmethod
    def read_spark_table(table_name: str) -> pd.DataFrame:
        if spark is None:
            raise RuntimeError("Spark session not available. Run this in Databricks or pass CSV/Excel paths.")
        return spark.table(table_name).toPandas()

    @staticmethod
    def write_delta_in_batches(df: pd.DataFrame, table_name: str, chunk_size: int = 100000):
        if spark is None:
            raise RuntimeError("Spark session not available. Cannot write Delta table outside Databricks.")
        spark.conf.set("spark.sql.execution.arrow.pyspark.enabled", "true")
        for i in range(0, len(df), chunk_size):
            pdf_chunk = df.iloc[i:i + chunk_size]
            sdf = spark.createDataFrame(pdf_chunk)
            mode = "overwrite" if i == 0 else "append"
            sdf.write.mode(mode).saveAsTable(table_name)
            print(f"Wrote rows {i} to {i + len(pdf_chunk)} into {table_name}")


class StandardizationTools:
    LEGAL_SUFFIXES = [
        "inc", "inc.", "corporation", "corp", "corp.", "company", "co", "co.",
        "llc", "l.l.c", "ltd", "ltd.", "limited", "plc", "pvt", "private",
        "sa", "spa", "ag", "gmbh", "bv", "nv", "lp", "llp"
    ]

    ADDRESS_REPL_MAP = {
        "united states": "us", "west": "w", "south": "s", "north": "n", "east": "e",
        "building": "bldg", "country road": "cr", "road": "rd", "lane": "ln",
        "highway": "hwy", "avenue": "av", "street": "st", "boulevard": "blvd",
        "drive": "dr", "court": "ct", "place": "pl", "square": "sq", "circle": "cir",
        "terrace": "ter", "way": "wy", "parkway": "pkwy", "expressway": "expy",
        "walk": "walk", "crescent": "cres", "close": "cl", "mews": "mews",
        "row": "row", "rise": "rise", "hill": "hill", "gardens": "gdns", "garden": "gdn",
        "parade": "parade", "quay": "quay", "apartment": "apt", "flat": "flat",
        "unit": "unit", "suite": "ste", "floor": "fl", "room": "rm", "department": "dept",
        "office": "ofc", "northeast": "ne", "northwest": "nw", "southeast": "se",
        "southwest": "sw", "park": "pk", "heights": "hts", "valley": "vly",
        "mount": "mt", "lake": "lk", "island": "is", "bay": "bay", "bridge": "brg",
        "harbour": "hbr", "point": "pt", "fork": "frk", "ridge": "rdge", "centre": "ctr",
        "district": "dist", "borough": "boro", "township": "twp", "county": "cnty",
        "number": "no", "post office box": "po box", "zip code": "zip", "postal code": "postcode"
    }

    @staticmethod
    def safe_str_series(s: pd.Series) -> pd.Series:
        return s.fillna("").astype(str).replace("nan", "", regex=False)

    @staticmethod
    def string_standardize(df: pd.DataFrame, col_list: List[str]) -> pd.DataFrame:
        out = df.copy()
        for col in col_list:
            if col not in out.columns:
                out[col] = ""
            out[col] = StandardizationTools.safe_str_series(out[col]).str.strip().str.lower()
            out[col] = out[col].str.replace(r"[^\w\s]", "", regex=True)
            out[col] = out[col].str.replace(r"\s+", " ", regex=True).str.strip()
            out[col] = out[col].str.replace(".", "", regex=False).str.replace(",", "", regex=False).str.replace("'", "", regex=False)
        return out

    @staticmethod
    def name_standardize(df: pd.DataFrame, col_list: List[str], mna_dict: Optional[Dict[str, str]] = None) -> pd.DataFrame:
        out = df.copy()
        for col in col_list:
            if col not in out.columns:
                out[col] = ""
            out[col] = StandardizationTools.safe_str_series(out[col]).str.lower()
            if mna_dict:
                for pattern, repl in mna_dict.items():
                    try:
                        out[col] = out[col].str.replace(pattern.lower(), str(repl).lower(), regex=True)
                    except Exception:
                        pass
            out[col] = out[col].str.replace(r"\s*\(RE:[^\)]*\)", "", regex=True)
            for suffix in StandardizationTools.LEGAL_SUFFIXES:
                out[col] = out[col].str.replace(rf"\b{re.escape(suffix)}\b", "", regex=True)
            out[col] = out[col].str.replace(r"[^\w\s]", "", regex=True)
            out[col] = out[col].str.replace(r"\s+", " ", regex=True).str.strip()
        return out

    @staticmethod
    def address_standardize(df: pd.DataFrame, col_list: List[str]) -> pd.DataFrame:
        out = df.copy()
        for col in col_list:
            if col not in out.columns:
                out[col] = ""
            out[col] = StandardizationTools.safe_str_series(out[col]).str.lower()
            for full, abbr in StandardizationTools.ADDRESS_REPL_MAP.items():
                out[col] = out[col].str.replace(rf"\b{re.escape(full)}\b", abbr, regex=True)
            out[col] = out[col].str.replace(r"[^\w\s\-]", "", regex=True)
            out[col] = out[col].str.replace(r"\s+", " ", regex=True).str.strip()
        return out

    @staticmethod
    def country_2_to_3(df: pd.DataFrame, map_df: pd.DataFrame, country_col_list: List[str]) -> pd.DataFrame:
        out = df.copy()
        if map_df is None or not {"Alpha_2", "Alpha_3"}.issubset(set(map_df.columns)):
            return out
        m = map_df[["Alpha_2", "Alpha_3"]].drop_duplicates(subset=["Alpha_2"])
        for col in country_col_list:
            if col not in out.columns:
                continue
            original = "original_" + col
            out[col] = StandardizationTools.safe_str_series(out[col]).str.upper()
            out = pd.merge(out, m, left_on=col, right_on="Alpha_2", how="left")
            out.rename(columns={col: original, "Alpha_3": col}, inplace=True)
            out.drop(["Alpha_2"], axis=1, inplace=True, errors="ignore")
            out[col] = out[col].fillna(out[original]).astype(str).str.lower()
        return out

    @staticmethod
    def extract_us_postal_code(df: pd.DataFrame, address_column: str, output_col: str = "postal_code") -> pd.DataFrame:
        out = df.copy()
        if address_column not in out.columns:
            out[output_col] = ""
            return out
        zip_pattern = r"\b\d{5}(?:-\d{4})?\b"
        out[output_col] = StandardizationTools.safe_str_series(out[address_column]).str.extract(f"({zip_pattern})", expand=False)
        return out

    @staticmethod
    def split_bbg_address(df: pd.DataFrame, split_col: str) -> pd.DataFrame:
        out = df.copy()
        if split_col not in out.columns:
            for c in ["bbg_legal_street_address", "bbg_legal_city", "bbg_legal_State", "bbg_legal_ZipCode", "bbg_legal_country"]:
                out[c] = ""
            return out
        parts = StandardizationTools.safe_str_series(out[split_col]).str.split("\n", expand=True)
        for i in range(6):
            if i not in parts.columns:
                parts[i] = None
        change_flag = np.where(parts[5].isnull(), "N", "Y")
        split_df = pd.DataFrame({"change_flag": change_flag})
        split_df["bbg_legal_street_address"] = np.where(split_df["change_flag"] == "Y", parts[0].fillna("") + " " + parts[1].fillna(""), parts[0])
        split_df["bbg_legal_city"] = np.where(split_df["change_flag"] == "Y", parts[2], parts[1])
        split_df["bbg_legal_State"] = np.where(split_df["change_flag"] == "Y", parts[3], parts[2])
        split_df["bbg_legal_ZipCode"] = np.where(split_df["change_flag"] == "Y", parts[4], parts[3])
        split_df["bbg_legal_country"] = np.where(split_df["change_flag"] == "Y", parts[5], parts[4])
        country_null = split_df["bbg_legal_country"].isnull()
        split_df.loc[country_null, "bbg_legal_country"] = split_df.loc[country_null, "bbg_legal_ZipCode"]
        split_df.loc[country_null, "bbg_legal_ZipCode"] = split_df.loc[country_null, "bbg_legal_State"]
        split_df.loc[country_null, "bbg_legal_State"] = split_df.loc[country_null, "bbg_legal_city"]
        return pd.concat([out.reset_index(drop=True), split_df.reset_index(drop=True)], axis=1)


class FeatureTools:
    @staticmethod
    def get_string_length(df: pd.DataFrame, col: str) -> pd.DataFrame:
        out = df.copy()
        out[col] = StandardizationTools.safe_str_series(out[col])
        out[col + "_str_length"] = out[col].str.len()
        return out[[col, col + "_str_length"]]

    @staticmethod
    def row_wise_fuzzy(df: pd.DataFrame, category: str, col1: str, col2: str) -> pd.DataFrame:
        out = df.copy()
        if col1 not in out.columns:
            out[col1] = ""
        if col2 not in out.columns:
            out[col2] = ""
        score_col = "fuzzy_score_" + category
        out[score_col] = [fuzz.ratio(str(x), str(y)) for x, y in zip(out[col1].fillna(""), out[col2].fillna(""))]
        return out

    @staticmethod
    def fuzzy_match_entities(main_df: pd.DataFrame, map_list: List[str], match_column: str, threshold: int) -> pd.DataFrame:
        out = main_df.copy()
        meta = [(x, x) for x in map_list]

        def match_one(x):
            if not x:
                return (pd.NA, pd.NA, pd.NA)
            result = process.extractOne(x, map_list, scorer=fuzz.token_set_ratio)
            if result is None:
                return (pd.NA, pd.NA, pd.NA)
            best_text, score, idx = result
            if score < threshold:
                return ("NO MATCH", score, x)
            canonical_col, orig = meta[idx]
            return (best_text, score, x)

        matched = out[match_column].apply(match_one)
        out["standardized_name"] = matched.apply(lambda t: t[0])
        out["match_score"] = matched.apply(lambda t: t[1])
        out["match_source"] = matched.apply(lambda t: t[2])
        return out


# COMMAND ----------

# =========================
# 2. Agent Layer
# =========================

class DataIntakeAgent:
    def __init__(self, config: ERConfig):
        self.config = config

    def run(self) -> Dict[str, pd.DataFrame]:
        cfg = self.config
        if cfg.input_type == "raw":
            print("Data Intake Agent: reading raw BBG and PRDM data")
            df_bbg = DataTools.read_csv(cfg.bbg_path)
            df_prd = DataTools.read_csv(cfg.prd_path)
            return {"df_bbg": df_bbg, "df_prd": df_prd, "df_combined": pd.DataFrame()}

        if cfg.input_type == "combined":
            print("Data Intake Agent: reading combined/labeled data")
            if cfg.read_from_table == "yes":
                df_combined = DataTools.read_spark_table(cfg.combined_path)
            else:
                try:
                    xls = pd.ExcelFile(cfg.combined_excel_path)
                    frames = []
                    for sheet in ["400_lei", "4K_cross"]:
                        if sheet in xls.sheet_names:
                            frames.append(pd.read_excel(xls, sheet_name=sheet))
                    df_combined = pd.concat(frames, axis=0, ignore_index=True) if frames else DataTools.read_csv(cfg.combined_path, encoding="latin1")
                except Exception:
                    df_combined = DataTools.read_csv(cfg.combined_path, encoding="latin1")
            return {"df_bbg": pd.DataFrame(), "df_prd": pd.DataFrame(), "df_combined": df_combined}

        raise ValueError("input_type must be 'raw' or 'combined'")


class ColumnSelectionAgent:
    bbg_final_cols = [
        "country_domicile_3_char_iso_code", "naics_national_industry_code", "entity_address",
        "entity_short_name", "entity_lei_name", "entity_legal_name", "entity_lei_code", "entity_bloomberg_id"
    ]

    prdm_final_cols = [
        "master_party_smun_identifier", "party_us_naics_code", "party_head_office_address_country_code",
        "party_head_office_address_postal_extension_code", "party_head_office_address_postal_code",
        "party_head_office_address_county_name", "party_head_office_address_state_code",
        "party_head_office_address_city_name", "party_head_office_address_line_3_text",
        "party_head_office_address_line_2_text", "party_head_office_address_line_1_text",
        "party_principal_address_country_code", "party_principal_address_postal_extension_code",
        "party_principal_address_postal_code", "party_principal_address_state_code",
        "party_principal_address_county_name", "party_principal_address_city_name",
        "party_principal_address_line_3_text", "party_principal_address_line_2_text",
        "party_principal_address_line_1_text", "party_legal_address_country_code",
        "party_legal_address_postal_extension_code", "party_legal_address_postal_code",
        "party_legal_address_county_name", "party_legal_address_state_code", "party_legal_address_city_name",
        "party_legal_address_line_3_text", "party_legal_address_line_2_text", "party_legal_address_line_1_text",
        "party_legal_entity_identifier", "party_incorporation_country_code", "party_legal_name", "master_party_legal_name"
    ]

    def run(self, data: Dict[str, pd.DataFrame], input_type: str) -> Dict[str, pd.DataFrame]:
        if input_type == "raw":
            df_bbg = data["df_bbg"].copy()
            df_prd = data["df_prd"].copy()
            df_bbg = df_bbg[[c for c in self.bbg_final_cols if c in df_bbg.columns]]
            df_prd = df_prd[[c for c in self.prdm_final_cols if c in df_prd.columns]]
            data["df_bbg"] = df_bbg
            data["df_prd"] = df_prd
        return data


class StandardizationAgent:
    def __init__(self, config: ERConfig):
        self.config = config
        self.mna_dict = self._load_mna_dict()

    def _load_mna_dict(self) -> Dict[str, str]:
        try:
            df_mna = DataTools.read_spark_table(self.config.mna_table)
            key_col = "PRDM Legal Name (Alternative Name)"
            value_col = "BBG Legal Name"
            if {key_col, value_col}.issubset(df_mna.columns):
                return df_mna[[key_col, value_col]].dropna().set_index(key_col)[value_col].to_dict()
        except Exception:
            print("M&A dictionary not loaded; continuing without it.")
        return {}

    def _standardize_bbg(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out["Standardized_composite_bbg_legal_name"] = np.where(
            out.get("entity_legal_name", pd.Series(index=out.index)).isnull(),
            np.where(out.get("entity_lei_name", pd.Series(index=out.index)).isnull(), out.get("entity_short_name", ""), out.get("entity_lei_name", "")),
            out.get("entity_legal_name", "")
        )
        out["Composite_bbg_legal_name"] = out["Standardized_composite_bbg_legal_name"]

        bbg_name_cols = ["entity_legal_name", "entity_lei_name", "entity_short_name"]
        for col in bbg_name_cols:
            if col in out.columns:
                out["Standardized_" + col] = out[col]
        standard_bbg_name_cols = ["Standardized_" + c for c in bbg_name_cols if c in out.columns] + ["Standardized_composite_bbg_legal_name"]
        out = StandardizationTools.name_standardize(out, standard_bbg_name_cols, self.mna_dict)

        if "entity_address" in out.columns:
            out = StandardizationTools.split_bbg_address(out, "entity_address")
            out["Composite_bbg_legal_address"] = out["entity_address"]
            out["Standardized_composite_bbg_legal_address"] = out["Composite_bbg_legal_address"]
            address_components = ["bbg_legal_street_address", "bbg_legal_city", "bbg_legal_State", "bbg_legal_ZipCode", "bbg_legal_country"]
            for col in address_components:
                out["Standardized_" + col] = out[col] if col in out.columns else ""
            out = StandardizationTools.address_standardize(out, ["Standardized_" + c for c in address_components] + ["Standardized_composite_bbg_legal_address"])
            out = StandardizationTools.name_standardize(out, ["Standardized_composite_bbg_legal_address"])
        return out

    def _standardize_prdm(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out["Standardized_composite_prdm_legal_name"] = np.where(
            out.get("party_legal_name", pd.Series(index=out.index)).isnull(),
            out.get("master_party_legal_name", ""),
            out.get("party_legal_name", "")
        )
        out["Composite_prdm_legal_name"] = out["Standardized_composite_prdm_legal_name"]

        prd_name_cols = ["party_legal_name", "master_party_legal_name"]
        for col in prd_name_cols:
            if col in out.columns:
                out["Standardized_" + col] = out[col]
        standard_prdm_name_cols = ["Standardized_" + c for c in prd_name_cols if c in out.columns] + ["Standardized_composite_prdm_legal_name"]
        out = StandardizationTools.name_standardize(out, standard_prdm_name_cols, self.mna_dict)

        granular_address_cols = [c for c in out.columns if any(x in c for x in ["address_line", "address_city", "address_state", "address_postal", "address_country", "county_name"])]
        for col in granular_address_cols:
            out[col] = StandardizationTools.safe_str_series(out[col]).str.replace("nan", "", regex=False)

        # Remove country prefix from state codes, e.g. US-TX -> TX
        for col in [c for c in granular_address_cols if "state_code" in c]:
            out[col] = StandardizationTools.safe_str_series(out[col]).str.split("-").str[-1]

        def join_cols(cols: List[str]) -> pd.Series:
            existing = [c for c in cols if c in out.columns]
            if not existing:
                return pd.Series([""] * len(out), index=out.index)
            s = StandardizationTools.safe_str_series(out[existing[0]])
            for c in existing[1:]:
                s = s + " " + StandardizationTools.safe_str_series(out[c])
            return s.str.replace(r"\s+", " ", regex=True).str.strip()

        out["party_legal_address"] = join_cols([
            "party_legal_address_line_1_text", "party_legal_address_line_2_text", "party_legal_address_line_3_text",
            "party_legal_address_city_name", "party_legal_address_state_code", "party_legal_address_postal_code", "party_legal_address_country_code"
        ])
        out["party_principal_address"] = join_cols([
            "party_principal_address_line_1_text", "party_principal_address_line_2_text", "party_principal_address_line_3_text",
            "party_principal_address_city_name", "party_principal_address_state_code", "party_principal_address_postal_code", "party_principal_address_country_code"
        ])
        out["party_head_office_address"] = join_cols([
            "party_head_office_address_line_1_text", "party_head_office_address_line_2_text", "party_head_office_address_line_3_text",
            "party_head_office_address_city_name", "party_head_office_address_state_code", "party_head_office_address_postal_code", "party_head_office_address_country_code"
        ])
        out["Standardized_composite_prdm_legal_address"] = np.where(
            out["party_head_office_address"] != "", out["party_head_office_address"],
            np.where(out["party_principal_address"] != "", out["party_principal_address"], out["party_legal_address"])
        )
        out["Composite_prdm_legal_address"] = out["Standardized_composite_prdm_legal_address"]

        address_cols = [
            "party_head_office_address", "party_principal_address", "party_legal_address",
            "party_legal_address_city_name", "party_legal_address_state_code", "party_legal_address_postal_code",
            "party_principal_address_city_name", "party_principal_address_state_code", "party_principal_address_postal_code",
            "party_head_office_address_city_name", "party_head_office_address_state_code", "party_head_office_address_postal_code",
            "party_legal_address_line_1_text", "party_principal_address_line_1_text", "party_head_office_address_line_1_text"
        ]
        for col in address_cols:
            if col in out.columns:
                out["Standardized_" + col] = out[col]
        std_addr_cols = ["Standardized_" + c for c in address_cols if c in out.columns] + ["Standardized_composite_prdm_legal_address"]
        out = StandardizationTools.address_standardize(out, std_addr_cols)
        out = StandardizationTools.name_standardize(out, ["Standardized_composite_prdm_legal_address"])
        return out

    def run(self, data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        if self.config.input_type == "raw":
            data["df_standard_bbg"] = self._standardize_bbg(data["df_bbg"])
            data["df_standard_prd"] = self._standardize_prdm(data["df_prd"])
        else:
            df = data["df_combined"].copy()
            # Combined data may already have both PRDM and BBG fields. Apply both routines safely.
            df = self._standardize_bbg(df)
            df = self._standardize_prdm(df)
            data["df_combined"] = df
        return data


class CandidateGenerationAgent:
    def __init__(self, config: ERConfig):
        self.config = config

    def run(self, data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        if self.config.input_type == "raw":
            print("Candidate Generation Agent: cross joining standardized BBG and PRDM")
            df_combined = data["df_standard_bbg"].merge(data["df_standard_prd"], how="cross")
        else:
            df_combined = data["df_combined"].copy()

        if self.config.input_type == "raw" and self.config.zip_match == "yes":
            bbg_zip_col = "bbg_legal_ZipCode"
            prdm_zip_cols = ["party_head_office_address_postal_code", "party_principal_address_postal_code", "party_legal_address_postal_code"]
            for c in [bbg_zip_col] + prdm_zip_cols:
                if c not in df_combined.columns:
                    df_combined[c] = ""
            bbg_zip = StandardizationTools.safe_str_series(df_combined[bbg_zip_col])
            mask = False
            for c in prdm_zip_cols:
                mask = mask | (bbg_zip == StandardizationTools.safe_str_series(df_combined[c]))
            df_combined = df_combined[mask].copy()
            print(f"ZIP blocking retained {len(df_combined)} candidate rows")

        data["df_combined"] = df_combined
        return data


class FeatureEngineeringAgent:
    def run(self, data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        df = data["df_combined"].copy()

        bbg_feature_name_cols = [
            "Standardized_composite_bbg_legal_name", "Standardized_entity_legal_name",
            "Standardized_entity_lei_name", "Standardized_entity_short_name"
        ]
        prdm_feature_name_cols = [
            "Standardized_composite_prdm_legal_name", "Standardized_party_legal_name", "Standardized_master_party_legal_name"
        ]
        for b in bbg_feature_name_cols:
            for p in prdm_feature_name_cols:
                if b in df.columns and p in df.columns:
                    category = b + "_" + p
                    print(category)
                    df = FeatureTools.row_wise_fuzzy(df, category, b, p)

        address_pairs = [
            ("Standardized_bbg_legal_street_address", "Standardized_party_head_office_address_line_1_text"),
            ("Standardized_bbg_legal_city", "Standardized_party_head_office_address_city_name"),
            ("Standardized_bbg_legal_State", "Standardized_party_head_office_address_state_code"),
            ("Standardized_bbg_legal_ZipCode", "Standardized_party_head_office_address_postal_code"),
            ("Standardized_bbg_legal_street_address", "Standardized_party_principal_address_line_1_text"),
            ("Standardized_bbg_legal_city", "Standardized_party_principal_address_city_name"),
            ("Standardized_bbg_legal_State", "Standardized_party_principal_address_state_code"),
            ("Standardized_bbg_legal_ZipCode", "Standardized_party_principal_address_postal_code"),
            ("Standardized_bbg_legal_street_address", "Standardized_party_legal_address_line_1_text"),
            ("Standardized_bbg_legal_city", "Standardized_party_legal_address_city_name"),
            ("Standardized_bbg_legal_State", "Standardized_party_legal_address_state_code"),
            ("Standardized_bbg_legal_ZipCode", "Standardized_party_legal_address_postal_code"),
            ("Standardized_composite_bbg_legal_address", "Standardized_composite_prdm_legal_address"),
            ("Standardized_composite_bbg_legal_address", "Standardized_party_head_office_address"),
            ("Standardized_composite_bbg_legal_address", "Standardized_party_principal_address"),
            ("Standardized_composite_bbg_legal_address", "Standardized_party_legal_address"),
        ]
        for b, p in address_pairs:
            if b in df.columns and p in df.columns:
                category = b + "_" + p
                print(category)
                df = FeatureTools.row_wise_fuzzy(df, category, b, p)

        data["df_combined"] = df
        return data


class ValidationDatasetAgent:
    def run(self, data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        df = data["df_combined"].copy()
        original_cols = [
            "master_party_smun_identifier", "entity_bloomberg_id", "Composite_bbg_legal_name", "entity_legal_name",
            "entity_lei_name", "entity_short_name", "master_party_legal_name", "party_legal_name",
            "Composite_prdm_legal_name", "party_incorporation_country_code", "party_us_naics_code",
            "country_domicile_3_char_iso_code", "naics_national_industry_code"
        ]
        manual_cols = [
            "Name Match (Y/N)", "Street Address Match (Y/N)", "City Match (Y/N)", "Zip Code Match (Y/N)",
            "State Match (Y/N)", "Observed Patterns (To be Updated)", "Composite_prdm_street_address",
            "Composite_bbg_street_address", "standardized_prdm_street_address", "standardized_bbg_street_address",
            "entity_website_address", "Match Label(New)", "Summarize", "Evidence", "Match Label Justification (New)",
            "Critical Flag", "Recommended Street Address"
        ]
        existing_original = [c for c in original_cols if c in df.columns]
        existing_manual = [c for c in manual_cols if c in df.columns]
        fuzzy_cols = [c for c in df.columns if c.startswith("fuzzy_score_")]
        useful_cols = existing_original + existing_manual + fuzzy_cols
        useful_cols = list(dict.fromkeys(useful_cols))
        df_validation = df[useful_cols].copy() if useful_cols else df.copy()
        df_validation = df_validation.loc[:, ~df_validation.columns.duplicated()]
        data["df_combined_validation"] = df_validation
        print(f"Validation Dataset Agent output shape: {df_validation.shape}")
        return data


class ExportAgent:
    def __init__(self, config: ERConfig):
        self.config = config

    def run(self, data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        if self.config.write_to_table == "yes":
            DataTools.write_delta_in_batches(data["df_combined_validation"], self.config.output_table, self.config.chunk_size)
        return data


class ReviewAndAnalysisAgent:
    """Simple non-LLM review rules. This can be expanded after ML model scoring is added."""

    @staticmethod
    def add_review_decision(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        name_col = "fuzzy_score_Standardized_composite_bbg_legal_name_Standardized_composite_prdm_legal_name"
        addr_col = "fuzzy_score_Standardized_composite_bbg_legal_address_Standardized_composite_prdm_legal_address"
        if name_col not in out.columns:
            out[name_col] = 0
        if addr_col not in out.columns:
            out[addr_col] = 0
        out["agent_decision"] = np.select(
            [
                (out[name_col] >= 95) & (out[addr_col] >= 90),
                (out[name_col] >= 85) & (out[addr_col] >= 70),
                (out[name_col] < 70) & (out[addr_col] < 60),
            ],
            ["ACCEPT_HIGH_CONFIDENCE", "SEND_TO_REVIEW", "REJECT_LOW_CONFIDENCE"],
            default="SEND_TO_REVIEW"
        )
        return out


class EntityResolutionAgenticPipeline:
    def __init__(self, config: ERConfig):
        self.config = config
        self.agents = [
            DataIntakeAgent(config),
            ColumnSelectionAgent(),
            StandardizationAgent(config),
            CandidateGenerationAgent(config),
            FeatureEngineeringAgent(),
            ValidationDatasetAgent(),
        ]
        self.export_agent = ExportAgent(config)

    def run(self, export: bool = True) -> Dict[str, pd.DataFrame]:
        data: Dict[str, pd.DataFrame] = {}
        for agent in self.agents:
            print(f"\n=== Running {agent.__class__.__name__} ===")
            if isinstance(agent, ColumnSelectionAgent):
                data = agent.run(data, self.config.input_type)
            else:
                data = agent.run(data)
        data["df_combined_validation"] = ReviewAndAnalysisAgent.add_review_decision(data["df_combined_validation"])
        if export:
            print("\n=== Running ExportAgent ===")
            data = self.export_agent.run(data)
        return data


# COMMAND ----------

# =========================
# 3. Run Pipeline
# =========================

config = ERConfig(
    input_type="raw",        # raw or combined
    zip_match="no",         # yes or no
    read_from_table="no",
    write_to_table="yes",
    output_table="mrd_ered_dev.ered_bronze.ML_input_features_test_484121",
)

# To execute in Databricks:
# pipeline = EntityResolutionAgenticPipeline(config)
# results = pipeline.run(export=True)
# df_combined_validation = results["df_combined_validation"]
# display(df_combined_validation.head())

