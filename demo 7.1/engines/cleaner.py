"""
Cleaning Engine — all transformation functions, pure & side-effect free
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest


#  Missing Values 

def fill_missing(df: pd.DataFrame, cols: list[str], method: str, custom_val: str = "") -> pd.DataFrame:
    df = df.copy()
    for col in cols:
        if col not in df.columns:
            continue
        if method == "mean":
            df[col] = df[col].fillna(df[col].mean())
        elif method == "median":
            df[col] = df[col].fillna(df[col].median())
        elif method == "mode":
            mode = df[col].mode()
            if not mode.empty:
                df[col] = df[col].fillna(mode.iloc[0])
        elif method == "unknown":
            df[col] = df[col].fillna("Unknown")
        elif method == "ffill":
            df[col] = df[col].ffill()
        elif method == "bfill":
            df[col] = df[col].bfill()
        elif method == "interpolate":
            df[col] = df[col].interpolate()
        elif method == "custom":
            df[col] = df[col].fillna(custom_val)
        elif method == "drop_rows":
            df = df.dropna(subset=[col])
    return df


def auto_fill_missing(df: pd.DataFrame, drop_threshold: float = 0.60) -> pd.DataFrame:
    df = df.copy()
    for col in df.columns:
        pct = df[col].isna().mean()
        if pct > drop_threshold:
            df.drop(columns=[col], inplace=True)
        elif pct > 0:
            if pd.api.types.is_numeric_dtype(df[col]):
                df[col] = df[col].fillna(df[col].median())
            else:
                mode = df[col].mode()
                df[col] = df[col].fillna(mode.iloc[0] if not mode.empty else "Unknown")
    return df


def drop_columns(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    return df.drop(columns=[c for c in cols if c in df.columns])


#  Duplicates 

def remove_exact_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop_duplicates().reset_index(drop=True)


#  Outliers 

def handle_outliers_iqr(df: pd.DataFrame, col: str, action: str) -> pd.DataFrame:
    df = df.copy()
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    if action == "remove":
        df = df[(df[col] >= lower) & (df[col] <= upper)]
    elif action in ("cap", "winsorize"):
        df[col] = df[col].clip(lower, upper)
    return df


def handle_outliers_zscore(df: pd.DataFrame, col: str, action: str, threshold: float = 3.0) -> pd.DataFrame:
    df = df.copy()
    z = (df[col] - df[col].mean()) / df[col].std()
    mask = z.abs() > threshold
    if action == "remove":
        df = df[~mask]
    elif action in ("cap", "winsorize"):
        upper = df[col].mean() + threshold * df[col].std()
        lower = df[col].mean() - threshold * df[col].std()
        df[col] = df[col].clip(lower, upper)
    return df


def handle_outliers_isolation_forest(df: pd.DataFrame, cols: list[str], action: str, contamination: float = 0.05) -> tuple[pd.DataFrame, np.ndarray]:
    df = df.copy()
    sub = df[cols].dropna()
    if sub.empty:
        return df, np.array([])
    clf = IsolationForest(contamination=contamination, random_state=42)
    preds = clf.fit_predict(sub)
    outlier_idx = sub.index[preds == -1]
    if action == "remove":
        df = df.drop(index=outlier_idx)
    return df, outlier_idx.to_numpy()


#  Text Cleaning 

def clean_text(df: pd.DataFrame, col: str, ops: list[str]) -> pd.DataFrame:
    df = df.copy()
    s = df[col].astype(str)
    if "trim" in ops:
        s = s.str.strip()
    if "lowercase" in ops:
        s = s.str.lower()
    if "uppercase" in ops:
        s = s.str.upper()
    if "remove_symbols" in ops:
        s = s.str.replace(r"[^a-zA-Z0-9\s]", "", regex=True)
    if "remove_extra_spaces" in ops:
        s = s.str.replace(r"\s+", " ", regex=True).str.strip()
    df[col] = s
    return df


def normalize_case(df: pd.DataFrame, col: str) -> pd.DataFrame:
    df = df.copy()
    df[col] = df[col].astype(str).str.strip().str.lower()
    return df


#  Type Conversion 

def convert_dtype(df: pd.DataFrame, col: str, target: str) -> pd.DataFrame:
    df = df.copy()
    if target == "int":
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
    elif target == "float":
        df[col] = pd.to_numeric(df[col], errors="coerce").astype(float)
    elif target == "string":
        df[col] = df[col].astype(str)
    elif target == "datetime":
        df[col] = pd.to_datetime(df[col], errors="coerce")
    elif target == "bool":
        mapping = {"true": True, "false": False, "1": True, "0": False, "yes": True, "no": False}
        df[col] = df[col].astype(str).str.lower().map(mapping)
    return df


#  Scaling 

def scale_column(df: pd.DataFrame, col: str, method: str) -> pd.DataFrame:
    df = df.copy()
    x = df[col].astype(float)
    if method == "minmax":
        df[col] = (x - x.min()) / (x.max() - x.min() + 1e-9)
    elif method == "standard":
        df[col] = (x - x.mean()) / (x.std() + 1e-9)
    elif method == "robust":
        q1, q3 = x.quantile(0.25), x.quantile(0.75)
        df[col] = (x - x.median()) / (q3 - q1 + 1e-9)
    elif method == "log":
        df[col] = np.log1p(x.clip(lower=0))
    return df


#  Apply AI Suggestion (sklear/isolation-forest)

def apply_suggestion(df: pd.DataFrame, action: str, col: str) -> pd.DataFrame:
    if action == "drop_column":
        return drop_columns(df, [col])
    elif action == "fill_median":
        return fill_missing(df, [col], "median")
    elif action == "fill_unknown":
        return fill_missing(df, [col], "unknown")
    elif action == "log_transform":
        return scale_column(df, col, "log")
    elif action == "normalize_case":
        return normalize_case(df, col)
    elif action == "cap_outliers":
        return handle_outliers_iqr(df, col, "cap")
    elif action == "convert_numeric":
        return convert_dtype(df, col, "float")
    return df
