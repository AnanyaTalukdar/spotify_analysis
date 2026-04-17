import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Spotify Premium Drivers",
    page_icon="🎧",
    layout="wide"
)

# ── Styling ───────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .block-container { padding-top: 2rem; }
    .metric-card {
        background: #1a1a2e;
        border: 1px solid #1DB954;
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        text-align: center;
    }
    .metric-label { color: #aaa; font-size: 13px; margin-bottom: 4px; }
    .metric-value { color: #1DB954; font-size: 32px; font-weight: 700; }
    .metric-sub   { color: #ccc; font-size: 12px; margin-top: 4px; }
    .section-title {
        color: #1DB954;
        font-size: 18px;
        font-weight: 600;
        margin: 2rem 0 0.5rem;
        border-bottom: 1px solid #333;
        padding-bottom: 6px;
    }
    .insight-box {
        background: #0f2a1a;
        border-left: 3px solid #1DB954;
        border-radius: 6px;
        padding: 0.75rem 1rem;
        margin: 0.4rem 0;
        color: #ccc;
        font-size: 14px;
    }
    .insight-box strong { color: #1DB954; }
</style>
""", unsafe_allow_html=True)

# ── Data loading & model ──────────────────────────────────────────────────────
@st.cache_data
def load_data():
    return pd.read_excel("data/raw/Spotify_data.xlsx")

@st.cache_data
def run_model(df):
    features = [
        "Age", "Gender", "spotify_usage_period",
        "spotify_listening_device", "preferred_listening_content",
        "music_time_slot", "music_lis_frequency"
    ]
    target = "premium_sub_willingness"
    model_df = df[features + [target]].copy()
    model_df[target] = model_df[target].map({"Yes": 1, "No": 0})
    X = pd.get_dummies(model_df[features], drop_first=True)
    y = model_df[target]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    coef_df = pd.DataFrame({
        "Feature": X.columns,
        "Coefficient": model.coef_[0]
    }).sort_values("Coefficient", ascending=False)
    coef_df["Label"] = coef_df["Feature"].apply(
        lambda n: n.split("_", 1)[1] if "_" in n else n
    )
    return model, X_test, y_test, y_pred, coef_df, accuracy_score(y_test, y_pred)

@st.cache_data
def build_corr(df):
    df2 = df.copy()
    if df2["Age"].dtype == "object":
        def convert_age(x):
            s = str(x)
            if "-" in s:
                lo, hi = s.split("-")
                return (int(lo) + int(hi)) / 2
            elif "+" in s:
                return int(s.replace("+", ""))
            return pd.to_numeric(x, errors="coerce")
        df2["Age"] = df2["Age"].apply(convert_age)
    df2["premium_sub_willingness"] = df2["premium_sub_willingness"].map({"Yes": 1, "No": 0})
    cat_cols = df2.select_dtypes(include=["object"]).columns
    df2 = pd.get_dummies(df2, columns=cat_cols, drop_first=True)
    return df2.corr()

def generate_insights(fdf):
    target = "premium_sub_willingness"
    insights = []
    if len(fdf) == 0:
        return ["No data available for the current filters."]
    overall_pct = (fdf[target] == "Yes").mean() * 100

    checks = [
        ("Age",                        "aged"),
        ("Gender",                     "identifying as"),
        ("preferred_listening_content","who prefer"),
        ("spotify_listening_device",   "listening on"),
        ("spotify_usage_period",       "who have used Spotify for"),
        ("music_time_slot",            "who listen during"),
        ("music_lis_frequency",        "who listen"),
    ]

    for col, phrase in checks:
        if col not in fdf.columns:
            continue
        rates = (
            fdf.groupby(col)[target]
            .apply(lambda x: (x == "Yes").mean() * 100)
            .sort_values(ascending=False)
        )
        if len(rates) == 0:
            continue
        top_val  = rates.index[0]
        top_pct  = rates.iloc[0]
        diff     = top_pct - overall_pct
        direction = f"+{diff:.0f}pp above" if diff >= 0 else f"{diff:.0f}pp below"
        insights.append(
            f"Users <strong>{phrase} {top_val}</strong> are most likely to upgrade "
            f"(<strong>{top_pct:.0f}%</strong> — {direction} the {overall_pct:.0f}% overall rate)."
        )

    return insights

# ── Load everything ───────────────────────────────────────────────────────────
df = load_data()
model, X_test, y_test, y_pred, coef_df, accuracy = run_model(df)
corr_matrix = build_corr(df)

# ── Header ────────────────────────────────────────────────────────────────────
st.title("🎧 Spotify Premium — What drives users to upgrade?")
st.caption("Logistic regression analysis on Spotify user survey data")

# ══════════════════════════════════════════════════════════════════════════════
# KPI CARDS — always at top, computed on full dataset, always visible
# ══════════════════════════════════════════════════════════════════════════════
total_all   = len(df)
willing_all = (df["premium_sub_willingness"] == "Yes").sum()
pct_all     = round(willing_all / total_all * 100, 1)
top_feature = coef_df.iloc[0]["Label"]

col1, col2, col3, col4 = st.columns(4)
for col, label, value, sub in [
    (col1, "Total users",        f"{total_all:,}",   "full dataset"),
    (col2, "Willing to upgrade", f"{willing_all:,}", f"{pct_all}% of total"),
    (col3, "Model accuracy",     f"{accuracy:.0%}",  "logistic regression"),
    (col4, "Top driver",         top_feature[:28],   "highest positive coefficient"),
]:
    col.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
        <div class="metric-sub">{sub}</div>
    </div>""", unsafe_allow_html=True)

st.markdown("")

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR FILTERS — affect breakdown + insights sections only
# ══════════════════════════════════════════════════════════════════════════════
st.sidebar.header("Filters")
st.sidebar.caption("Applies to breakdown charts and insights below.")

sel_gender  = st.sidebar.selectbox("Gender",
    ["All"] + sorted(df["Gender"].dropna().unique().tolist()))
sel_age     = st.sidebar.selectbox("Age group",
    ["All"] + sorted(df["Age"].dropna().unique().tolist()))
sel_device  = st.sidebar.selectbox("Listening device",
    ["All"] + sorted(df["spotify_listening_device"].dropna().unique().tolist()))
sel_content = st.sidebar.selectbox("Preferred content",
    ["All"] + sorted(df["preferred_listening_content"].dropna().unique().tolist()))

filtered = df.copy()
if sel_gender  != "All": filtered = filtered[filtered["Gender"] == sel_gender]
if sel_age     != "All": filtered = filtered[filtered["Age"] == sel_age]
if sel_device  != "All": filtered = filtered[filtered["spotify_listening_device"] == sel_device]
if sel_content != "All": filtered = filtered[filtered["preferred_listening_content"] == sel_content]

f_total   = len(filtered)
f_willing = (filtered["premium_sub_willingness"] == "Yes").sum()
f_pct     = round(f_willing / f_total * 100, 1) if f_total > 0 else 0
st.caption(
    f"Filters active — **{f_total:,}** users · **{f_willing:,}** willing to upgrade (**{f_pct}%**)"
)

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — Correlation heatmap (raw data, before model)
# ══════════════════════════════════════════════════════════════════════════════
st.markdown(
    '<div class="section-title">Correlation with premium willingness — before modelling</div>',
    unsafe_allow_html=True
)
st.caption(
    "Measures each feature's direct relationship with willingness, in isolation. "
    "+1 = perfectly together, -1 = perfectly opposite, 0 = no relationship."
)

target_corr = (
    corr_matrix[["premium_sub_willingness"]]
    .drop("premium_sub_willingness", errors="ignore")
    .sort_values("premium_sub_willingness", ascending=False)
    .head(20)
)
target_corr.index = [i.split("_", 1)[-1] for i in target_corr.index]

fig, ax = plt.subplots(figsize=(5, 7))
fig.patch.set_facecolor("#0e0e0e")
ax.set_facecolor("#0e0e0e")
sns.heatmap(target_corr, annot=True, fmt=".2f", cmap="coolwarm",
            center=0, ax=ax, linewidths=0.3)
ax.tick_params(colors="white")
st.pyplot(fig)
plt.close()

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — Feature importance / coefficients (after model)
# ══════════════════════════════════════════════════════════════════════════════
st.markdown(
    '<div class="section-title">Feature importance — what the model learned</div>',
    unsafe_allow_html=True
)
st.caption(
    "Coefficient = each feature's contribution to the prediction with all other features present. "
    "Green = pushes toward upgrading. Red = pushes away."
)

top_n   = st.slider("Show top / bottom N features", 5, 20, 10)
plot_df = pd.concat([coef_df.head(top_n), coef_df.tail(top_n)]).drop_duplicates()
plot_df = plot_df.sort_values("Coefficient")

fig, ax = plt.subplots(figsize=(10, max(4, len(plot_df) * 0.35)))
fig.patch.set_facecolor("#0e0e0e")
ax.set_facecolor("#0e0e0e")
colors = ["#1DB954" if v > 0 else "#e05c5c" for v in plot_df["Coefficient"]]
ax.barh(plot_df["Label"], plot_df["Coefficient"], color=colors)
ax.axvline(0, color="#555", linewidth=0.8)
ax.tick_params(colors="white")
ax.xaxis.label.set_color("white")
for spine in ax.spines.values():
    spine.set_edgecolor("#333")
st.pyplot(fig)
plt.close()

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — Who is willing to upgrade (filtered breakdown)
# ══════════════════════════════════════════════════════════════════════════════
st.markdown(
    '<div class="section-title">Who is willing to upgrade?</div>',
    unsafe_allow_html=True
)
st.caption("% of users within each group who said Yes. Sidebar filters apply.")

tabs = st.tabs(["By age", "By gender", "By usage period", "By device", "By content", "By time slot"])
breakdown_cols = [
    "Age", "Gender", "spotify_usage_period",
    "spotify_listening_device", "preferred_listening_content", "music_time_slot"
]

def willingness_bar(data, col):
    ct = pd.crosstab(data[col], data["premium_sub_willingness"], normalize="index") * 100
    if "Yes" not in ct.columns or len(ct) == 0:
        st.info("No data for this selection.")
        return
    ct = ct.sort_values("Yes", ascending=True)
    fig, ax = plt.subplots(figsize=(8, max(3, len(ct) * 0.45)))
    fig.patch.set_facecolor("#0e0e0e")
    ax.set_facecolor("#0e0e0e")
    bars = ax.barh(ct.index.astype(str), ct["Yes"], color="#1DB954")
    ax.bar_label(bars, fmt="%.0f%%", color="white", padding=4, fontsize=10)
    ax.set_xlabel("% willing to upgrade", color="white")
    ax.set_xlim(0, min(ct["Yes"].max() * 1.25, 100))
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#333")
    st.pyplot(fig)
    plt.close()

for tab, col in zip(tabs, breakdown_cols):
    with tab:
        willingness_bar(filtered, col)

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — Auto-generated insights
# ══════════════════════════════════════════════════════════════════════════════
st.markdown(
    '<div class="section-title">Insights — what the data says</div>',
    unsafe_allow_html=True
)
st.caption("Auto-generated from the filtered data. Adjust sidebar filters to see how insights change.")

for insight in generate_insights(filtered):
    st.markdown(f'<div class="insight-box">{insight}</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — Confusion matrix
# ══════════════════════════════════════════════════════════════════════════════
st.markdown(
    '<div class="section-title">Model performance — confusion matrix</div>',
    unsafe_allow_html=True
)
st.caption(f"Accuracy: {accuracy:.1%} on held-out test set (25% of data)")

col_a, col_b = st.columns([1, 2])
with col_a:
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(4, 3))
    fig.patch.set_facecolor("#0e0e0e")
    ax.set_facecolor("#0e0e0e")
    sns.heatmap(cm, annot=True, fmt="d", cmap="Greens",
                xticklabels=["No", "Yes"], yticklabels=["No", "Yes"], ax=ax)
    ax.set_xlabel("Predicted", color="white")
    ax.set_ylabel("Actual", color="white")
    ax.tick_params(colors="white")
    st.pyplot(fig)
    plt.close()
with col_b:
    st.markdown("""
    **How to read this:**
    - **Top-left**: Correctly predicted "No" (won't upgrade)
    - **Bottom-right**: Correctly predicted "Yes" (will upgrade)
    - **Top-right**: Said "No" but model predicted "Yes" *(false positive)*
    - **Bottom-left**: Said "Yes" but model predicted "No" *(false negative)*

    A good model has large numbers on the diagonal and small numbers off it.
    """)

# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.caption("Built with Streamlit · Logistic regression via scikit-learn · Data: Spotify user survey")