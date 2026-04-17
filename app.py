import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score

# ── Page config ──────────────────────────────────────────────────────────────
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
</style>
""", unsafe_allow_html=True)

# ── Load data ─────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_excel("data/raw/Spotify_data.xlsx")
    return df

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

    # Simplify feature names (strip prefix)
    def clean_name(name):
        parts = name.split("_", 1)
        return parts[1] if len(parts) > 1 else name

    coef_df["Label"] = coef_df["Feature"].apply(clean_name)

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

df = load_data()
model, X_test, y_test, y_pred, coef_df, accuracy = run_model(df)
corr_matrix = build_corr(df)

# ── Header ────────────────────────────────────────────────────────────────────
st.title("🎧 Spotify Premium — What drives users to upgrade?")
st.caption("Logistic regression analysis on Spotify user survey data")

# ── Sidebar filters ───────────────────────────────────────────────────────────
st.sidebar.header("Filters")
genders   = ["All"] + sorted(df["Gender"].dropna().unique().tolist())
sel_gender = st.sidebar.selectbox("Gender", genders)
ages      = ["All"] + sorted(df["Age"].dropna().unique().tolist())
sel_age   = st.sidebar.selectbox("Age group", ages)

filtered = df.copy()
if sel_gender != "All":
    filtered = filtered[filtered["Gender"] == sel_gender]
if sel_age != "All":
    filtered = filtered[filtered["Age"] == sel_age]

# ── KPI row ───────────────────────────────────────────────────────────────────
total       = len(filtered)
willing     = (filtered["premium_sub_willingness"] == "Yes").sum()
pct_willing = round(willing / total * 100, 1) if total > 0 else 0
top_feature = coef_df.iloc[0]["Label"]

col1, col2, col3, col4 = st.columns(4)
for col, label, value, sub in [
    (col1, "Total users",          f"{total:,}",        "in filtered view"),
    (col2, "Willing to upgrade",   f"{willing:,}",      f"{pct_willing}% of total"),
    (col3, "Model accuracy",       f"{accuracy:.0%}",   "logistic regression"),
    (col4, "Top driver",           top_feature[:28],    "highest positive coefficient"),
]:
    col.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
        <div class="metric-sub">{sub}</div>
    </div>""", unsafe_allow_html=True)

# ── Section 1: Feature importance ─────────────────────────────────────────────
st.markdown('<div class="section-title">Feature importance — what pushes users toward Premium</div>', unsafe_allow_html=True)
st.caption("Positive = increases likelihood of upgrading. Negative = decreases it.")

top_n  = st.slider("Show top/bottom N features", 5, 20, 10)
top    = coef_df.head(top_n)
bottom = coef_df.tail(top_n)
plot_df = pd.concat([top, bottom]).drop_duplicates()
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

# ── Section 2: Willingness breakdown ──────────────────────────────────────────
st.markdown('<div class="section-title">Who is willing to upgrade?</div>', unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["By age", "By gender", "By usage period"])

def willingness_bar(df, col):
    ct = pd.crosstab(df[col], df["premium_sub_willingness"], normalize="index") * 100
    if "Yes" not in ct.columns:
        st.info("No data for this selection.")
        return
    ct = ct.sort_values("Yes", ascending=True)
    fig, ax = plt.subplots(figsize=(8, max(3, len(ct) * 0.4)))
    fig.patch.set_facecolor("#0e0e0e")
    ax.set_facecolor("#0e0e0e")
    ax.barh(ct.index, ct["Yes"], color="#1DB954")
    ax.set_xlabel("% willing to upgrade", color="white")
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#333")
    st.pyplot(fig)
    plt.close()

with tab1:
    willingness_bar(filtered, "Age")
with tab2:
    willingness_bar(filtered, "Gender")
with tab3:
    willingness_bar(filtered, "spotify_usage_period")

# ── Section 3: Confusion matrix ───────────────────────────────────────────────
st.markdown('<div class="section-title">Model performance — confusion matrix</div>', unsafe_allow_html=True)
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

# ── Section 4: Correlation heatmap ────────────────────────────────────────────
st.markdown('<div class="section-title">Correlation with premium willingness</div>', unsafe_allow_html=True)
st.caption("How strongly each feature correlates with a user wanting to upgrade.")

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

# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.caption("Built with Streamlit · Logistic regression via scikit-learn · Data: Spotify user survey")