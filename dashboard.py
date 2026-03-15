import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from analysis import (
    load_and_clean, z_test, t_test_rounds,
    bootstrap, bayesian_analysis, calculate_mde
)

# ─────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────

st.set_page_config(
    page_title="Cookie Cats A/B Test",
    page_icon="🎮",
    layout="wide"
)

# ─────────────────────────────────────────
# LOAD DATA & RUN ANALYSIS
# ─────────────────────────────────────────

@st.cache_data
def run_analysis():
    df     = load_and_clean()
    r1     = z_test(df, "retention_1")
    r7     = z_test(df, "retention_7")
    rounds = t_test_rounds(df)
    boot1  = bootstrap(df, "retention_1")
    boot7  = bootstrap(df, "retention_7")
    bayes1 = bayesian_analysis(df, "retention_1")
    bayes7 = bayesian_analysis(df, "retention_7")
    mde1   = calculate_mde(df, "retention_1")
    mde7   = calculate_mde(df, "retention_7")
    return df, r1, r7, rounds, boot1, boot7, bayes1, bayes7, mde1, mde7

df, r1, r7, rounds, boot1, boot7, bayes1, bayes7, mde1, mde7 = run_analysis()


# ─────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────

st.title("🎮 Cookie Cats — A/B Test Analysis")
st.markdown(
    "**Experiment:** Does moving the gate from level 30 → 40 "
    "improve player retention? | "
    f"**Sample:** {len(df):,} players"
)
st.markdown("---")


# ─────────────────────────────────────────
# ROW 1 — TOP METRICS
# ─────────────────────────────────────────

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("1-Day Retention (gate_30)", f"{r1['gate_30']}%")
with col2:
    st.metric("1-Day Retention (gate_40)", f"{r1['gate_40']}%",
              delta=f"{r1['difference']:+.3f}%",
              delta_color="inverse")
with col3:
    st.metric("7-Day Retention (gate_30)", f"{r7['gate_30']}%")
with col4:
    st.metric("7-Day Retention (gate_40)", f"{r7['gate_40']}%",
              delta=f"{r7['difference']:+.3f}%",
              delta_color="inverse")

st.markdown("---")


# ─────────────────────────────────────────
# ROW 2 — RETENTION BAR CHARTS
# ─────────────────────────────────────────

st.subheader("📊 Retention Rate Comparison")
col1, col2 = st.columns(2)

def retention_bar(result, title):
    fig = go.Figure()
    colors = ["#6366f1", "#f43f5e"]
    for i, (group, val) in enumerate([
        ("gate_30", result["gate_30"]),
        ("gate_40", result["gate_40"])
    ]):
        fig.add_trace(go.Bar(
            x=[group], y=[val],
            name=group,
            marker_color=colors[i],
            text=f"{val}%",
            textposition="outside"
        ))
    fig.update_layout(
        title=title,
        yaxis_title="Retention Rate (%)",
        paper_bgcolor="#0f0f0f",
        plot_bgcolor="#0f0f0f",
        font_color="#e2e2e2",
        height=350,
        showlegend=False,
        yaxis=dict(gridcolor="#1a1a1a"),
        margin=dict(l=0, r=0, t=40, b=0)
    )
    return fig

with col1:
    st.plotly_chart(retention_bar(r1, "1-Day Retention"), use_container_width=True)
with col2:
    st.plotly_chart(retention_bar(r7, "7-Day Retention"), use_container_width=True)


# ─────────────────────────────────────────
# ROW 3 — STATISTICAL SIGNIFICANCE
# ─────────────────────────────────────────

st.markdown("---")
st.subheader("🔬 Statistical Significance")

col1, col2 = st.columns(2)

def significance_card(result, title):
    sig   = result["significant"]
    color = "#00C9A7" if sig else "#f59e0b"
    label = "✅ SIGNIFICANT" if sig else "⚠️ NOT SIGNIFICANT"
    st.markdown(f"""
    <div style='background:#111;border:2px solid {color};border-radius:12px;padding:20px;'>
        <div style='font-size:13px;color:#888;margin-bottom:6px;font-family:monospace'>{title}</div>
        <div style='font-size:22px;font-weight:900;color:{color};margin-bottom:12px'>{label}</div>
        <div style='display:grid;grid-template-columns:1fr 1fr;gap:8px;font-family:monospace;font-size:12px'>
            <div style='color:#888'>p-value</div>
            <div style='color:#fff'>{result['p_value']}</div>
            <div style='color:#888'>Difference</div>
            <div style='color:#fff'>{result['difference']:+.3f}%</div>
            <div style='color:#888'>95% CI</div>
            <div style='color:#fff'>[{result['ci_low']:.3f}%, {result['ci_high']:.3f}%]</div>
            <div style='color:#888'>Winner</div>
            <div style='color:{color}'>{result['winner'].upper()}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

with col1:
    significance_card(r1, "1-Day Retention — Z-Test")
with col2:
    significance_card(r7, "7-Day Retention — Z-Test")


# ─────────────────────────────────────────
# ROW 4 — BOOTSTRAP DISTRIBUTIONS
# ─────────────────────────────────────────

st.markdown("---")
st.subheader("🔄 Bootstrap Simulation (1,000 iterations)")
col1, col2 = st.columns(2)

def bootstrap_chart(boot, title):
    diffs = np.array(boot["diffs"]) * 100
    fig   = go.Figure()
    fig.add_trace(go.Histogram(
        x=diffs, nbinsx=50,
        marker_color="#6366f1",
        opacity=0.8,
        name="Simulated differences"
    ))
    fig.add_vline(x=0, line_dash="dash", line_color="#f43f5e", line_width=2,
                  annotation_text="No effect", annotation_font_color="#f43f5e")
    fig.add_vline(x=boot["mean_diff"], line_dash="solid",
                  line_color="#00C9A7", line_width=2,
                  annotation_text=f"Mean: {boot['mean_diff']:+.3f}%",
                  annotation_font_color="#00C9A7")
    fig.update_layout(
        title=f"{title}<br><sup>P(gate_30 wins): {boot['prob_control_wins']}%</sup>",
        xaxis_title="Difference in retention (%)",
        paper_bgcolor="#0f0f0f",
        plot_bgcolor="#0f0f0f",
        font_color="#e2e2e2",
        height=320,
        showlegend=False,
        xaxis=dict(gridcolor="#1a1a1a"),
        yaxis=dict(gridcolor="#1a1a1a"),
        margin=dict(l=0, r=0, t=50, b=0)
    )
    return fig

with col1:
    st.plotly_chart(bootstrap_chart(boot1, "1-Day Retention Bootstrap"),
                    use_container_width=True)
with col2:
    st.plotly_chart(bootstrap_chart(boot7, "7-Day Retention Bootstrap"),
                    use_container_width=True)


# ─────────────────────────────────────────
# ROW 5 — BAYESIAN POSTERIORS
# ─────────────────────────────────────────

st.markdown("---")
st.subheader("🎲 Bayesian Posterior Distributions")
col1, col2 = st.columns(2)

def bayesian_chart(bayes, title):
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=bayes["samples_control"], nbinsx=80,
        name="gate_30", marker_color="#6366f1",
        opacity=0.7, histnorm="probability density"
    ))
    fig.add_trace(go.Histogram(
        x=bayes["samples_treatment"], nbinsx=80,
        name="gate_40", marker_color="#f43f5e",
        opacity=0.7, histnorm="probability density"
    ))
    fig.update_layout(
        title=f"{title}<br>"
              f"<sup>P(gate_30 better): {bayes['prob_gate30_better']}%</sup>",
        barmode="overlay",
        xaxis_title="Retention Rate",
        paper_bgcolor="#0f0f0f",
        plot_bgcolor="#0f0f0f",
        font_color="#e2e2e2",
        height=320,
        xaxis=dict(gridcolor="#1a1a1a", tickformat=".1%"),
        yaxis=dict(gridcolor="#1a1a1a"),
        margin=dict(l=0, r=0, t=50, b=0)
    )
    return fig

with col1:
    st.plotly_chart(bayesian_chart(bayes1, "1-Day Retention Posteriors"),
                    use_container_width=True)
with col2:
    st.plotly_chart(bayesian_chart(bayes7, "7-Day Retention Posteriors"),
                    use_container_width=True)


# ─────────────────────────────────────────
# ROW 6 — ROUNDS DISTRIBUTION
# ─────────────────────────────────────────

st.markdown("---")
st.subheader("🎯 Rounds Played Distribution")

fig = go.Figure()
for group, color in [("gate_30", "#6366f1"), ("gate_40", "#f43f5e")]:
    subset = df[df["version"] == group]["sum_gamerounds"]
    fig.add_trace(go.Histogram(
        x=subset, nbinsx=60,
        name=group, marker_color=color,
        opacity=0.7, histnorm="probability density"
    ))
fig.update_layout(
    barmode="overlay",
    xaxis_title="Rounds Played",
    yaxis_title="Density",
    paper_bgcolor="#0f0f0f",
    plot_bgcolor="#0f0f0f",
    font_color="#e2e2e2",
    height=320,
    xaxis=dict(gridcolor="#1a1a1a"),
    yaxis=dict(gridcolor="#1a1a1a"),
    margin=dict(l=0, r=0, t=10, b=0)
)
st.plotly_chart(fig, use_container_width=True)


# ─────────────────────────────────────────
# ROW 7 — MDE & FINAL DECISION
# ─────────────────────────────────────────

st.markdown("---")
col1, col2 = st.columns(2)

with col1:
    st.subheader("📏 Minimum Detectable Effect")
    mde_df = pd.DataFrame([
        {"Metric": "1-Day Retention", "Sample/Group": f"{mde1['sample_size']:,}",
         "Baseline": f"{mde1['baseline']}%", "MDE (abs)": f"{mde1['mde']}%",
         "MDE (rel)": f"{mde1['mde_relative']}%"},
        {"Metric": "7-Day Retention", "Sample/Group": f"{mde7['sample_size']:,}",
         "Baseline": f"{mde7['baseline']}%", "MDE (abs)": f"{mde7['mde']}%",
         "MDE (rel)": f"{mde7['mde_relative']}%"},
    ])
    st.dataframe(mde_df, use_container_width=True, hide_index=True)

with col2:
    st.subheader("✅ Final Decision")
    st.markdown("""
    <div style='background:#0a0a0a;border:2px solid #00C9A7;
                border-radius:12px;padding:20px;'>
        <div style='font-size:20px;font-weight:900;color:#00C9A7;
                    margin-bottom:12px;'>
            🏆 KEEP GATE AT LEVEL 30
        </div>
        <div style='font-size:13px;color:#aaa;line-height:1.8'>
            • 7-day retention is significantly higher with gate_30<br>
            • p-value = 0.0019 (well below α = 0.05)<br>
            • Bootstrap: gate_30 wins in 99.8% of simulations<br>
            • Bayesian: 99.91% probability gate_30 is better<br>
            • Moving gate to level 40 would reduce long-term retention<br>
            • Recommendation: Do NOT ship the gate_40 change
        </div>
    </div>
    """, unsafe_allow_html=True)