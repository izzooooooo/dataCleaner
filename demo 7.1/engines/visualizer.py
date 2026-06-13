"""
Visualization Layer — Plotly charts for profiling and exploration
"""
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Professional SaaS Palette
PALETTE = ["#4F46E5", "#0EA5E9", "#10B981", "#F59E0B", "#F43F5E", "#8B5CF6", "#14B8A6", "#F97316"]

def _apply_pro_layout(fig: go.Figure, title: str = "") -> go.Figure:
    """Applies a clean, professional layout to all charts."""
    fig.update_layout(
        title={"text": title, "font": {"family": "Plus Jakarta Sans", "size": 18, "color": "#0F172A"}},
        template="plotly_white",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Plus Jakarta Sans", color="#475569"),
        margin=dict(t=50, l=20, r=20, b=20),
        xaxis=dict(showgrid=False, zeroline=False, linecolor="#E2E8F0", tickfont=dict(color="#64748B")),
        yaxis=dict(showgrid=True, gridcolor="#F1F5F9", gridwidth=1, zeroline=False, tickfont=dict(color="#64748B")),
        colorway=PALETTE
    )
    return fig

def missing_bar(df: pd.DataFrame) -> go.Figure:
    null_pct = (df.isna().mean() * 100).round(1).sort_values(ascending=False)
    null_pct = null_pct[null_pct > 0]
    if null_pct.empty:
        fig = go.Figure()
        fig.add_annotation(text="Boş dəyər yoxdur ✓", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False, font=dict(size=16, color="#10B981"))
        fig.update_layout(template="plotly_white", height=200, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
        return fig
    colors = ["#EF4444" if v > 50 else "#F59E0B" if v > 20 else "#4F46E5" for v in null_pct.values]
    fig = px.bar(x=null_pct.index, y=null_pct.values, labels={"x": "", "y": "Boş (%)"})
    fig.update_traces(marker_color=colors, marker_line_width=0, opacity=0.9)
    _apply_pro_layout(fig, "Boş Dəyər Analizi")
    fig.update_layout(height=260, showlegend=False)
    return fig

def dtype_pie(df: pd.DataFrame) -> go.Figure:
    counts = df.dtypes.astype(str).value_counts()
    fig = px.pie(values=counts.values, names=counts.index, hole=0.45)
    fig.update_traces(marker=dict(colors=PALETTE), textposition='inside', textinfo='percent+label')
    _apply_pro_layout(fig, "Verilən Tipləri (Data Types)")
    fig.update_layout(height=260, showlegend=False, margin=dict(t=50, b=10, l=10, r=10))
    return fig

def correlation_heatmap(df: pd.DataFrame) -> go.Figure | None:
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    if len(num_cols) < 2: return None
    corr = df[num_cols].corr().round(2)
    fig = px.imshow(corr, text_auto=True, aspect="auto", color_continuous_scale="Blues")
    _apply_pro_layout(fig)
    fig.update_layout(height=max(350, len(num_cols) * 45), coloraxis_showscale=False)
    return fig

def boxplot(df: pd.DataFrame, col: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Box(y=df[col].dropna(), name=col, marker_color="#6366F1", boxpoints="outliers", jitter=0.3, line_width=1.5))
    _apply_pro_layout(fig, f"'{col}' — Outlier Görünümü")
    fig.update_layout(height=320)
    return fig

def histogram(df: pd.DataFrame, col: str) -> go.Figure:
    fig = px.histogram(df, x=col, nbins=40, color_discrete_sequence=["#38BDF8"], opacity=0.8)
    _apply_pro_layout(fig, f"'{col}' — Tezlik Paylanması")
    fig.update_layout(height=320, bargap=0.1)
    return fig

def quality_gauge(score: float) -> go.Figure:
    color = "#10B981" if score >= 80 else "#F59E0B" if score >= 60 else "#EF4444"
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=score,
        number={"suffix": "%", "font": {"size": 42, "color": "#0F172A", "family": "Plus Jakarta Sans"}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#CBD5E1"},
            "bar": {"color": color, "thickness": 0.25},
            "bgcolor": "#F1F5F9",
            "borderwidth": 0,
            "steps": [
                {"range": [0, 60], "color": "#FEE2E2"},
                {"range": [60, 80], "color": "#FEF3C7"},
                {"range": [80, 100], "color": "#D1FAE5"},
            ]
        }
    ))
    fig.update_layout(height=240, margin=dict(t=20, b=10, l=20, r=20), paper_bgcolor="rgba(0,0,0,0)")
    return fig

def render_chart(df: pd.DataFrame, x: str, y: str, chart_type: str) -> go.Figure:
    df = df.copy()
    total = len(df)

    if chart_type == "Pie":
        pie_df = df.groupby(x, as_index=False)[y].sum().sort_values(y, ascending=False)
        if len(pie_df) > 12:
            top = pie_df.head(12)
            other = pd.DataFrame({x: ["Digər"], y: [pie_df.iloc[12:][y].sum()]})
            pie_df = pd.concat([top, other], ignore_index=True)
        fig = px.pie(pie_df, names=x, values=y, hole=0.3)
    else:
        if total > 50_000:
            if pd.api.types.is_numeric_dtype(df[x]): df = df.groupby(x, as_index=False)[y].sum()
            else: df = df.groupby(x, as_index=False)[y].sum()

        if chart_type == "Bar": fig = px.bar(df, x=x, y=y)
        elif chart_type == "Line": fig = px.line(df, x=x, y=y)
        elif chart_type == "Scatter":
            sample = df.sample(min(50_000, len(df)), random_state=42)
            fig = px.scatter(sample, x=x, y=y, opacity=0.7)
        elif chart_type == "Histogram": fig = px.histogram(df, x=x, nbins=40)
        else: fig = px.bar(df, x=x, y=y)

    _apply_pro_layout(fig, f"{x} vs {y} ({chart_type})")
    fig.update_layout(height=450)
    if chart_type in ["Bar", "Histogram"]: fig.update_traces(marker_line_width=0)
    return fig