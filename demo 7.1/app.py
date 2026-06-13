"""
Universal Data Cleansing Dashboard — Professional Edition
"""
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

#  engine imports 
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from engines.profiler   import compute_quality_score, generate_suggestions, profile_column
from engines.schema     import (detect_header_issues, normalize_all_headers,
                                 apply_schema_template, detect_schema_template,
                                 normalize_missing_sentinels, ADULT_CENSUS_SCHEMA)
from engines.cleaner    import (fill_missing, auto_fill_missing, drop_columns,
                                 remove_exact_duplicates, handle_outliers_iqr,
                                 handle_outliers_zscore, handle_outliers_isolation_forest,
                                 clean_text, normalize_case, convert_dtype,
                                 scale_column, apply_suggestion)
from engines.pipeline   import Pipeline
from engines.visualizer import (missing_bar, dtype_pie, correlation_heatmap,
                                 boxplot, histogram, quality_gauge, render_chart)


# PAGE CONFIG & CSS


st.set_page_config(
    page_title="DataCleaner Pro",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="1"
)

# Professional Dark Theme (Inter)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif !important;
    }

    /* Main Background */
    .stApp { background-color: #0B1220; color: #E2E8F0; }

    /* Typography */
    .pro-header { 
        font-size: 2.2rem; font-weight: 700; color: #F8FAFC; 
        letter-spacing: -0.02em; margin-bottom: 0.2rem; margin-top: -1rem;
    }
    .pro-subheader { 
        font-size: 1.05rem; color: #94A3B8; font-weight: 400; 
        margin-bottom: 2rem; border-bottom: 1px solid #1E293B; padding-bottom: 1rem;
    }

    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: #0A0F1C;
        border-right: 1px solid #1E293B;
    }
    [data-testid="stSidebar"] * { color: #E2E8F0 !important; }
    div[data-testid="stSidebar"] hr { border-color: #1E293B; }

    /* Custom Metric Cards */
    .metric-card {
        background: #0F172A;
        border: 1px solid #1E293B;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 4px 10px rgba(0,0,0,0.25);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 12px 18px rgba(0,0,0,0.35);
    }
    .metric-title { font-size: 0.8rem; color: #94A3B8; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 8px; }
    .metric-value { font-size: 1.8rem; color: #E2E8F0; font-weight: 700; }

    /* Badges & Pills */
    .step-badge {
        display: inline-flex; align-items: center; background: #0F172A; color: #38BDF8 !important;
        border-radius: 9999px; padding: 4px 12px; font-size: 0.8rem;
        font-weight: 600; margin: 4px 0; border: 1px solid #1E293B;
    }

    /* AI Suggestion Cards */
    .sug-card {
        background: #0F172A; border-left: 4px solid #3B82F6;
        border-radius: 8px; padding: 16px; margin-bottom: 12px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.3); font-size: 0.95rem; color: #E2E8F0;
    }

    /* Quality Bar */
    .quality-bar-bg { background: #1E293B; border-radius: 9999px; height: 8px; overflow: hidden; margin-bottom: 4px; }
    .quality-bar-fill { height: 100%; border-radius: 9999px; transition: width 0.5s ease; }

    /* Buttons */
    .stButton>button { border-radius: 8px; font-weight: 600; background: #1D4ED8; color: white; border: none; }
    .stButton>button:hover { background: #2563EB; }

    /* DataFrame & Expanders */
    [data-testid="stDataFrame"] { border-radius: 8px; overflow: hidden; box-shadow: 0 1px 6px rgba(0,0,0,0.4); }
    .st-emotion-cache-1z1w1d8 { border-radius: 8px; border: 1px solid #1E293B; background: #0F172A; }
</style>
""", unsafe_allow_html=True)

# SESSION STATE & HELPERS


def _init_state():
    defaults = {
        "pipeline": None,
        "suggestions": [],
        "suggestions_generated": False,
        "x_col": None,
        "y_col": None,
        "chart_type": "Bar",
        "active_tab": "upload",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()
pipe: Pipeline | None = st.session_state.pipeline

def push(label: str, df: pd.DataFrame):
    pipe.push(label, df)
    st.session_state.suggestions_generated = False

def get_df() -> pd.DataFrame:
    return pipe.current if pipe else pd.DataFrame()

def render_metric(title, value):
    return f"""
    <div class="metric-card">
        <div class="metric-title">{title}</div>
        <div class="metric-value">{value}</div>
    </div>
    """


# SIDEBAR — NAVIGATION

has_data = pipe is not None

with st.sidebar:
    st.markdown("<h2 style='color: white; font-weight: 700; letter-spacing: -0.5px;'>DataCleaner Pro</h2>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    nav = st.radio(
        "Navigation",
        options=["upload", "schema", "cleaning", "filters", "visualize"],
        format_func=lambda x: {
            "upload":    "Dataset Upload" if not has_data else "Dataset Upload",
            "schema":    "Schema Management",
            "cleaning":  "Data Cleansing",
            "filters":   "Filters & Exploration",
            "visualize": "Visualization",
        }[x],
        label_visibility="collapsed",
    )
    st.session_state.active_tab = nav

    if has_data:
        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown("<h4 style='color: #94A3B8; font-size: 0.9rem; text-transform: uppercase;'>Pipeline History</h4>", unsafe_allow_html=True)
        for i, step in enumerate(pipe.steps[-6:]):
            label = step if i > 0 else "Original"
            st.markdown(f'<div class="step-badge">{label}</div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            if st.button("Undo", disabled=not pipe.can_undo(), use_container_width=True):
                pipe.undo()
                st.rerun()
        with c2:
            if st.button("Redo", disabled=not pipe.can_redo(), use_container_width=True):
                pipe.redo()
                st.rerun()
        if st.button("Revert to Original", use_container_width=True):
            pipe.revert()
            st.rerun()


# TAB: UPLOAD


if nav == "upload":
    st.markdown('<div class="pro-header">Upload Dataset</div>', unsafe_allow_html=True)
    st.markdown('<div class="pro-subheader">Upload your data and start cleansing (CSV, XLSX, .data)</div>', unsafe_allow_html=True)

    uploaded = st.file_uploader(
        "Upload file",
        type=["csv", "xls", "xlsx", "data"],
        label_visibility="collapsed"
    )

    if uploaded:
        try:
            with st.spinner("Loading dataset..."):
                name = uploaded.name.lower()
                if name.endswith((".csv", ".data")):
                    df = pd.read_csv(
                        uploaded,
                        header=None if name.endswith(".data") else "infer",
                        sep=r",\s*",
                        engine="python",
                        na_values=["?", "N/A", "NA", "null", "none", "undefined", ""]
                    )
                    if name.endswith(".data"):
                        df.columns = [f"col_{i}" for i in range(df.shape[1])]
                else:
                    df = pd.read_excel(uploaded)

            st.session_state.pipeline = Pipeline(df)
            st.session_state.suggestions = []
            st.session_state.suggestions_generated = False
            pipe = st.session_state.pipeline

            st.success(f"Loaded: {uploaded.name}")

            m1, m2, m3, m4 = st.columns(4)
            m1.markdown(render_metric("Total Rows", f"{df.shape[0]:,}"), unsafe_allow_html=True)
            m2.markdown(render_metric("Total Columns", f"{df.shape[1]}"), unsafe_allow_html=True)
            m3.markdown(render_metric("Missing Cells", f"{int(df.isna().sum().sum()):,}"), unsafe_allow_html=True)
            m4.markdown(render_metric("Memory Usage", f"{round(df.memory_usage(deep=True).sum()/1024/1024, 2)} MB"), unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("Initial Preview (First 10 Rows)")
            st.dataframe(df.head(10), use_container_width=True)

        except Exception as e:
            st.error(f"Error: {e}")

    elif has_data:
        df = get_df()
        st.info("Current dataset loaded.")
        m1, m2 = st.columns(2)
        m1.markdown(render_metric("Row Count", f"{df.shape[0]:,}"), unsafe_allow_html=True)
        m2.markdown(render_metric("Column Count", f"{df.shape[1]}"), unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        st.dataframe(df.head(5), use_container_width=True)


# TAB: SCHEMA


elif nav == "schema":
    st.markdown('<div class="pro-header">Schema Management</div>', unsafe_allow_html=True)
    st.markdown('<div class="pro-subheader">Detect and normalize column headers.</div>', unsafe_allow_html=True)

    if not has_data:
        st.warning("Please upload a dataset first.")
    else:
        df = get_df()
        issues_info = detect_header_issues(df)
        template = detect_schema_template(df)

        if issues_info["has_issues"]:
            st.warning("Schema issues detected:")
            for issue in issues_info["issues"]:
                st.markdown(f"- {issue}")
        else:
            st.success("Headers look clean.")

        if template == "adult_census":
            st.info("Detected dataset matches Adult Census schema.")
            if st.button("Apply Adult Census Schema", type="primary"):
                push("Applied Adult Census schema", apply_schema_template(df, "adult_census"))
                st.rerun()

        st.markdown("<br>", unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Normalize Headers (snake_case)", use_container_width=True):
                push("Normalized headers", normalize_all_headers(df))
                st.rerun()
        with col2:
            if st.button("Normalize Missing Sentinels", use_container_width=True):
                push("Normalized missing sentinels", normalize_missing_sentinels(df))
                st.rerun()

        st.markdown("---")
        st.markdown("Manual Header Editing")
        with st.container():
            new_names: dict[str, str] = {}
            cols_per_row = 4
            col_list = df.columns.tolist()
            for i in range(0, len(col_list), cols_per_row):
                row_cols = st.columns(cols_per_row)
                for j, col in enumerate(col_list[i:i+cols_per_row]):
                    with row_cols[j]:
                        new_name = st.text_input(f"Original: {col}", value=col, key=f"rename_{col}")
                        new_names[col] = new_name

            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("Apply Column Names", type="primary"):
                df_renamed = df.rename(columns=new_names)
                push("Renamed columns", df_renamed)
                st.rerun()


# TAB: CLEANING


elif nav == "cleaning":
    st.markdown('<div class="pro-header">Data Cleansing Module</div>', unsafe_allow_html=True)
    st.markdown('<div class="pro-subheader">Advanced data cleansing with AI assistance.</div>', unsafe_allow_html=True)

    if not has_data:
        st.warning("Please upload a dataset first.")
    else:
        df = get_df()
        left, center, right = st.columns([2.5, 4.5, 3])

        # LEFT — Profiling
        with left:
            st.markdown("Quality Profile")
            score_data = compute_quality_score(df)
            st.plotly_chart(quality_gauge(score_data["total"]), use_container_width=True, config={"displayModeBar": False})

            st.markdown("<div style='font-weight:600; font-size:0.9rem; margin-bottom:10px; color:#94A3B8;'>QUALITY COMPONENTS</div>", unsafe_allow_html=True)
            for name, val in score_data["components"].items():
                color = "#22C55E" if val >= 80 else "#F59E0B" if val >= 60 else "#EF4444"
                st.markdown(f"""
                <div style="margin-bottom:12px">
                    <div style="display:flex; justify-content:space-between; font-size:0.85rem; color:#94A3B8; margin-bottom:4px">
                        <span>{name}</span><span style="font-weight:600; color:{color}">{val}%</span>
                    </div>
                    <div class="quality-bar-bg">
                        <div class="quality-bar-fill" style="width:{val}%; background:{color};"></div>
                    </div>
                </div>""", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            st.plotly_chart(missing_bar(df), use_container_width=True, config={"displayModeBar": False})
            st.plotly_chart(dtype_pie(df), use_container_width=True, config={"displayModeBar": False})

        # CENTER — Cleaning Tools
        with center:
            st.markdown("Cleansing Tools")

            with st.expander("Missing Value Handling", expanded=True):
                mode = st.radio("Mode", ["Automatic", "Manual"], horizontal=True, key="mv_mode")
                if mode == "Automatic":
                    threshold = st.slider("Max missing threshold (%)", 0, 100, 60, key="mv_thr")
                    if st.button("Run Automatic Cleansing", key="mv_auto"):
                        with st.spinner("Processing..."):
                            df_new = auto_fill_missing(df, threshold / 100)
                        push("Auto missing handling", df_new)
                        st.rerun()
                else:
                    target = st.multiselect("Target columns", df.columns.tolist(), key="mv_cols")
                    method = st.selectbox("Fill method", ["mean", "median", "mode", "unknown", "ffill", "bfill", "interpolate", "custom", "drop_rows"], key="mv_method")
                    custom_v = ""
                    if method == "custom":
                        custom_v = st.text_input("Custom value", key="mv_custom")
                    if st.button("Apply to selected", key="mv_manual"):
                        if target:
                            df_new = fill_missing(df, target, method, custom_v)
                            push(f"Missing values: {method}", df_new)
                            st.rerun()
                        else:
                            st.warning("Select at least one column.")

            with st.expander("Duplicate Removal"):
                dup_count = int(df.duplicated().sum())
                st.info(f"Exact duplicate rows: {dup_count:,}")
                if st.button("Remove Duplicates", disabled=(dup_count == 0), type="primary"):
                    push("Duplicates removed", remove_exact_duplicates(df))
                    st.rerun()

            with st.expander("Outlier Handling"):
                num_cols = df.select_dtypes(include=np.number).columns.tolist()
                if not num_cols:
                    st.info("No numeric columns found.")
                else:
                    c1, c2, c3 = st.columns(3)
                    with c1: out_col = st.selectbox("Column", num_cols, key="out_col")
                    with c2: out_method = st.selectbox("Algorithm", ["IQR", "Z-Score", "Isolation Forest"], key="out_m")
                    with c3: out_action = st.selectbox("Action", ["cap", "remove", "winsorize", "ignore"], key="out_a")

                    st.plotly_chart(boxplot(df, out_col), use_container_width=True, config={"displayModeBar": False})

                    if out_action != "ignore" and st.button("Apply Outlier Handling", key="out_btn"):
                        if out_method == "IQR": df_new = handle_outliers_iqr(df, out_col, out_action)
                        elif out_method == "Z-Score": df_new = handle_outliers_zscore(df, out_col, out_action)
                        else: df_new, _ = handle_outliers_isolation_forest(df, [out_col], out_action)
                        push(f"Outliers handled: {out_col}", df_new)
                        st.rerun()

            with st.expander("Text Normalization"):
                cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
                if not cat_cols: st.info("No text columns found.")
                else:
                    tc_col = st.selectbox("Select column", cat_cols, key="tc_col")
                    tc_ops = st.multiselect("Options", ["trim", "lowercase", "uppercase", "remove_symbols", "remove_extra_spaces"], default=["trim", "lowercase"], key="tc_ops")
                    if st.button("Clean Text", key="tc_btn"):
                        df_new = clean_text(df, tc_col, tc_ops)
                        push(f"Text cleanup: {tc_col}", df_new)
                        st.rerun()

            with st.expander("Type Conversion"):
                c1, c2 = st.columns(2)
                with c1: dt_col = st.selectbox("Select column", df.columns.tolist(), key="dt_col")
                with c2: dt_target = st.selectbox("Target type", ["int", "float", "string", "datetime", "bool"], key="dt_t")
                st.caption(f"Current type: `{df[dt_col].dtype}`")
                if st.button("Convert Type", key="dt_btn"):
                    try:
                        df_new = convert_dtype(df, dt_col, dt_target)
                        push(f"Type conversion: {dt_col}", df_new)
                        st.rerun()
                    except Exception as e:
                        st.error(f"Conversion error: {e}")

        # RIGHT — AI Suggestions + Column Profiler
        with right:
            st.markdown("AI Assistant")
            if st.button("Analyze Dataset", use_container_width=True, type="primary"):
                with st.spinner("Analyzing..."):
                    st.session_state.suggestions = generate_suggestions(df)
                    st.session_state.suggestions_generated = True

            suggestions = st.session_state.suggestions
            if suggestions:
                for idx, sug in enumerate(suggestions):
                    st.markdown(f'<div class="sug-card"><b>Suggestion</b><br>{sug.text}</div>', unsafe_allow_html=True)
                    c1, c2 = st.columns(2)
                    with c1:
                        if st.button("Apply", key=f"acc_{idx}", use_container_width=True):
                            df_new = apply_suggestion(df, sug.action, sug.col)
                            push(f"AI action: {sug.action}", df_new)
                            st.session_state.suggestions.pop(idx)
                            st.rerun()
                    with c2:
                        if st.button("Dismiss", key=f"rej_{idx}", use_container_width=True):
                            st.session_state.suggestions.pop(idx)
                            st.rerun()
            elif st.session_state.suggestions_generated:
                st.success("No critical issues found.")

            st.markdown("<hr>", unsafe_allow_html=True)
            st.markdown("Column Profile")
            profile_col = st.selectbox("Select column", df.columns.tolist(), key="profiler_col")
            info = profile_column(df, profile_col)

            for k, v in info.items():
                if k == "top_values":
                    st.markdown("<div style='font-size:0.85rem; color:#94A3B8; margin-top:10px;'>Most Frequent Values:</div>", unsafe_allow_html=True)
                    for val, cnt in v.items():
                        st.markdown(f"- **{val}**: `{cnt}`")
                else:
                    st.markdown(f"<div style='display:flex; justify-content:space-between; border-bottom:1px solid #1E293B; padding:6px 0;'><span style='color:#94A3B8;'>{k.capitalize()}</span><span style='font-weight:600;'>{v}</span></div>", unsafe_allow_html=True)

            if pd.api.types.is_numeric_dtype(df[profile_col]):
                st.plotly_chart(histogram(df, profile_col), use_container_width=True, config={"displayModeBar": False})


# TAB: FILTERS


elif nav == "filters":
    st.markdown('<div class="pro-header">Filters & Exploration</div>', unsafe_allow_html=True)
    st.markdown('<div class="pro-subheader">Filter and export cleaned data.</div>', unsafe_allow_html=True)

    if not has_data:
        st.warning("Please upload a dataset first.")
    else:
        df = get_df()
        df_filtered = df.copy()

        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("Category Filter")
            cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
            if cat_cols:
                fcat = st.selectbox("Column", ["—"] + cat_cols, key="fcat", label_visibility="collapsed")
                if fcat != "—":
                    vals = df[fcat].astype(str).unique()
                    sel = st.multiselect("Select values", sorted(vals), key="fcat_vals")
                    if sel: df_filtered = df_filtered[df_filtered[fcat].astype(str).isin(sel)]
            else: st.info("No categorical columns.")

        with col2:
            st.markdown("Text Search")
            text_col = st.selectbox("Column", df.columns.tolist(), key="ftxt_col", label_visibility="collapsed")
            kw = st.text_input("Enter search keyword...", key="ftxt_kw")
            if kw: df_filtered = df_filtered[df_filtered[text_col].astype(str).str.contains(kw, case=False, na=False)]

        with col3:
            st.markdown("Numeric Range")
            num_cols = df.select_dtypes(include=np.number).columns.tolist()
            if num_cols:
                fnum = st.selectbox("Column", num_cols, key="fnum_col", label_visibility="collapsed")
                mn, mx = float(df[fnum].min()), float(df[fnum].max())
                if mn < mx:
                    rng = st.slider("Range", mn, mx, (mn, mx), key="fnum_rng", label_visibility="collapsed")
                    df_filtered = df_filtered[(df_filtered[fnum] >= rng[0]) & (df_filtered[fnum] <= rng[1])]
            else: st.info("No numeric columns.")

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(f"Result: **{len(df_filtered):,}** rows (Original: {len(df):,})")
        st.dataframe(df_filtered, use_container_width=True, height=500)

        csv = df_filtered.to_csv(index=False).encode("utf-8")
        st.download_button("Download Cleaned Dataset (CSV)", csv, "cleaned_data.csv", "text/csv", use_container_width=True, type="primary")


# TAB: VISUALIZE


elif nav == "visualize":
    st.markdown('<div class="pro-header">Visualization</div>', unsafe_allow_html=True)
    st.markdown('<div class="pro-subheader">Build professional charts from cleaned data.</div>', unsafe_allow_html=True)

    if not has_data:
        st.warning("Please upload a dataset first.")
    else:
        df = get_df()

        c1, c2, c3, c4 = st.columns([2, 2, 2, 1])
        with c1: x_col = st.selectbox("X Axis", df.columns.tolist(), key="viz_x")
        with c2: 
            num_cols = df.select_dtypes(include=np.number).columns.tolist()
            y_col = st.selectbox("Y Axis", num_cols if num_cols else df.columns.tolist(), key="viz_y")
        with c3: chart_type = st.selectbox("Chart Type", ["Bar", "Line", "Scatter", "Pie", "Histogram"], key="viz_ct")
        with c4: 
            st.markdown("<div style='margin-top: 28px;'></div>", unsafe_allow_html=True)
            render_btn = st.button("Render", type="primary", use_container_width=True)

        if render_btn:
            with st.spinner("Rendering chart..."):
                fig = render_chart(df, x_col, y_col, chart_type)
                st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")
        st.markdown("Distribution Analysis")
        num_cols = df.select_dtypes(include=np.number).columns.tolist()
        if num_cols:
            dist_col = st.selectbox("Select numeric column:", num_cols, key="dist_col")
            c1, c2 = st.columns(2)
            with c1: st.plotly_chart(histogram(df, dist_col), use_container_width=True, config={"displayModeBar": False})
            with c2: st.plotly_chart(boxplot(df, dist_col), use_container_width=True, config={"displayModeBar": False})

        st.markdown("---")
        st.markdown("Correlation Heatmap")
        corr_fig = correlation_heatmap(df)
        if corr_fig: st.plotly_chart(corr_fig, use_container_width=True)
        else: st.info("At least 2 numeric columns required for correlation heatmap.")