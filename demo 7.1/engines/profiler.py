"""
Profiling Engine — dataset statistics, quality scoring, AI suggestions
"""
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Any


MISSING_SENTINELS = {"?", "n/a", "na", "null", "none", "undefined", "", "nan", "N/A", "NA"}


#  Quality Score 

def compute_quality_score(df: pd.DataFrame) -> dict:
    if df.empty:
        return {"total": 0, "components": {}}

    total_cells = df.shape[0] * df.shape[1]

    # 1. Missing
    missing_pct = df.isna().sum().sum() / total_cells
    missing_score = max(0.0, 1.0 - missing_pct * 2)

    # 2. Duplicates
    dup_pct = df.duplicated().sum() / len(df)
    dup_score = max(0.0, 1.0 - dup_pct * 3)

    # 3. Data type consistency
    type_scores = []
    for col in df.columns:
        if df[col].dtype == object:
            # how many values could be numeric but stored as string
            numeric_attempt = pd.to_numeric(df[col], errors="coerce").notna().mean()
            if numeric_attempt > 0.9:
                type_scores.append(0.5)
            else:
                type_scores.append(1.0)
        else:
            type_scores.append(1.0)
    type_score = np.mean(type_scores) if type_scores else 1.0

    # 4. Outliers (IQR) across numeric cols
    outlier_fractions = []
    for col in df.select_dtypes(include=np.number).columns:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        if iqr > 0:
            frac = ((df[col] < q1 - 1.5 * iqr) | (df[col] > q3 + 1.5 * iqr)).mean()
            outlier_fractions.append(frac)
    outlier_score = max(0.0, 1.0 - np.mean(outlier_fractions) * 4) if outlier_fractions else 1.0

    # 5. Category consistency — casing
    cat_scores = []
    for col in df.select_dtypes(include=["object", "category"]).columns:
        vals = df[col].dropna().astype(str)
        if len(vals) == 0:
            continue
        lower_unique = vals.str.lower().nunique()
        orig_unique  = vals.nunique()
        if orig_unique > 0:
            cat_scores.append(lower_unique / orig_unique)
    cat_score = np.mean(cat_scores) if cat_scores else 1.0

    weights = {"missing": 0.30, "duplicates": 0.20, "types": 0.20, "outliers": 0.15, "categories": 0.15}
    total = (
        missing_score  * weights["missing"]  +
        dup_score      * weights["duplicates"] +
        type_score     * weights["types"]    +
        outlier_score  * weights["outliers"] +
        cat_score      * weights["categories"]
    )

    return {
        "total": round(total * 100, 1),
        "components": {
            "Missing Values":    round(missing_score  * 100, 1),
            "Duplicates":        round(dup_score       * 100, 1),
            "Type Consistency":  round(type_score      * 100, 1),
            "Outliers":          round(outlier_score   * 100, 1),
            "Category Consistency": round(cat_score    * 100, 1),
        }
    }


#  AI Suggestions 

@dataclass
class Suggestion:
    text: str
    col: str
    action: str
    params: dict = field(default_factory=dict)


def generate_suggestions(df: pd.DataFrame) -> list[Suggestion]:
    suggestions: list[Suggestion] = []

    null_pct = df.isna().mean()

    for col, pct in null_pct.items():
        if pct > 0.60:
            suggestions.append(Suggestion(
                text=f"'{col}' sütununda {pct:.0%} boş dəyər var. Tövsiyə: sütunu silin.",
                col=col, action="drop_column"
            ))
        elif pct > 0.20:
            if df[col].dtype in [np.float64, np.int64]:
                suggestions.append(Suggestion(
                    text=f"'{col}' sütununda {pct:.0%} boş dəyər var. Tövsiyə: median ilə doldurun.",
                    col=col, action="fill_median"
                ))
            else:
                suggestions.append(Suggestion(
                    text=f"'{col}' sütununda {pct:.0%} boş dəyər var. Tövsiyə: 'Unknown' ilə doldurun.",
                    col=col, action="fill_unknown"
                ))

    for col in df.select_dtypes(include=np.number).columns:
        s = df[col].dropna()
        if len(s) < 10:
            continue
        skew = s.skew()
        if abs(skew) > 2:
            suggestions.append(Suggestion(
                text=f"'{col}' sütunu yüksək çarpıqlıqdadır (skew={skew:.2f}). Tövsiyə: log çevrilməsi.",
                col=col, action="log_transform"
            ))
        # Outlier suggestion
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        if iqr > 0:
            out_frac = ((s < q1 - 1.5*iqr) | (s > q3 + 1.5*iqr)).mean()
            if out_frac > 0.05:
                suggestions.append(Suggestion(
                    text=f"'{col}' sütununda {out_frac:.1%} aşırı dəyər (outlier) aşkarlandı. Tövsiyə: IQR ilə kəsin.",
                    col=col, action="cap_outliers"
                ))

    for col in df.select_dtypes(include=["object"]).columns:
        vals = df[col].dropna().astype(str)
        if len(vals) == 0:
            continue
        lower_u = vals.str.lower().nunique()
        orig_u  = vals.nunique()
        if lower_u < orig_u:
            suggestions.append(Suggestion(
                text=f"'{col}' sütununda qeyri-ardıcıl harf (Male/MALE/male) aşkarlandı. Tövsiyə: normallaşdırın.",
                col=col, action="normalize_case"
            ))
        # Check if it's actually numeric
        num_ratio = pd.to_numeric(vals, errors="coerce").notna().mean()
        if num_ratio > 0.90:
            suggestions.append(Suggestion(
                text=f"'{col}' sütunu əslində ədədi görünür. Tövsiyə: float tipinə çevirin.",
                col=col, action="convert_numeric"
            ))

    # Duplicate columns (high correlation)
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    if len(num_cols) > 1:
        corr = df[num_cols].corr().abs()
        for i in range(len(num_cols)):
            for j in range(i + 1, len(num_cols)):
                if corr.iloc[i, j] > 0.99:
                    c1, c2 = num_cols[i], num_cols[j]
                    suggestions.append(Suggestion(
                        text=f"'{c1}' və '{c2}' sütunları demək olar ki eynidir (r={corr.iloc[i,j]:.3f}). Biri artıq ola bilər.",
                        col=c2, action="drop_column"
                    ))

    return suggestions


#  Column Profile 

def profile_column(df: pd.DataFrame, col: str) -> dict[str, Any]:
    s = df[col]
    info: dict[str, Any] = {
        "dtype": str(s.dtype),
        "count": int(s.notna().sum()),
        "missing": int(s.isna().sum()),
        "missing_pct": round(s.isna().mean() * 100, 1),
        "unique": int(s.nunique()),
    }
    if pd.api.types.is_numeric_dtype(s):
        info.update({
            "mean":   round(s.mean(), 4),
            "median": round(s.median(), 4),
            "std":    round(s.std(), 4),
            "min":    round(s.min(), 4),
            "max":    round(s.max(), 4),
            "skew":   round(s.skew(), 4),
        })
    else:
        top = s.value_counts().head(5).to_dict()
        info["top_values"] = top
    return info
