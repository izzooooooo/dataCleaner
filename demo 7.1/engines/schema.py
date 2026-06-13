"""
Schema Manager — header detection, normalization, Adult Census template
"""
import re
import pandas as pd
import numpy as np


ADULT_CENSUS_SCHEMA = [
    "age", "workclass", "fnlwgt", "education", "education_num",
    "marital_status", "occupation", "relationship", "race", "sex",
    "capital_gain", "capital_loss", "hours_per_week", "native_country", "income"
]

MISSING_SENTINELS = {"?", "n/a", "na", "null", "none", "undefined", "", "nan"}


def normalize_header(name: str) -> str:
    """' Capital Gain ' → 'capital_gain'"""
    name = str(name).strip().lower()
    name = re.sub(r"[^\w\s]", "", name)
    name = re.sub(r"\s+", "_", name)
    name = re.sub(r"_+", "_", name).strip("_")
    return name or "unnamed"


def make_unique_headers(headers: list[str]) -> list[str]:
    seen: dict[str, int] = {}
    result = []
    for h in headers:
        if h in seen:
            seen[h] += 1
            result.append(f"{h}_{seen[h]}")
        else:
            seen[h] = 0
            result.append(h)
    return result


def detect_header_issues(df: pd.DataFrame) -> dict:
    issues = []
    cols = df.columns.tolist()

    # Unnamed columns
    unnamed = [c for c in cols if str(c).startswith("col_") or str(c).startswith("Unnamed")]
    if unnamed:
        issues.append(f"{len(unnamed)} sütun başlığı yoxdur (unnamed).")

    # Duplicates
    if len(cols) != len(set(cols)):
        issues.append("Təkrarlanan sütun adları var.")

    # Headers with spaces/symbols
    messy = [c for c in cols if c != normalize_header(c)]
    if messy:
        issues.append(f"{len(messy)} sütun adı normallaşdırma tələb edir.")

    # First row looks like header
    first_row_numeric = pd.to_numeric(df.iloc[0], errors="coerce").notna().mean() if len(df) > 0 else 0
    first_row_is_header = first_row_numeric < 0.3

    return {
        "issues": issues,
        "has_issues": len(issues) > 0,
        "unnamed_cols": unnamed,
        "messy_cols": messy,
        "first_row_may_be_header": first_row_is_header,
    }


def detect_schema_template(df: pd.DataFrame) -> str | None:
    """Detect if dataset resembles a known schema."""
    if df.shape[1] == 15:
        # Check if numeric columns align with adult census
        num_numeric = df.select_dtypes(include=np.number).shape[1]
        if num_numeric >= 3:
            return "adult_census"
    return None


def normalize_missing_sentinels(df: pd.DataFrame) -> pd.DataFrame:
    """Replace all sentinel missing values with NaN."""
    df = df.copy()
    for col in df.columns:
        mask = df[col].astype(str).str.strip().str.lower().isin(MISSING_SENTINELS)
        df.loc[mask, col] = np.nan
    return df


def apply_schema_template(df: pd.DataFrame, template: str) -> pd.DataFrame:
    df = df.copy()
    if template == "adult_census" and df.shape[1] == len(ADULT_CENSUS_SCHEMA):
        df.columns = ADULT_CENSUS_SCHEMA
    return df


def normalize_all_headers(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    new_cols = [normalize_header(c) for c in df.columns]
    new_cols = make_unique_headers(new_cols)
    df.columns = new_cols
    return df
