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
    .note-box {
        background: #1c1c1c;
        border-left: 3px solid #aaa;
        border-radius: 6px;
        padding: 0.6rem 1rem;
        margin: 0.5rem 0 1rem;
        color: #aaa;
        font-size: 13px;
    }
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

# ── Load everything ───────────────────────────────────────────────────────────
df = load_data()
model, X_test, y_test, y_pred, coef_df, accuracy = run_model(df)
corr_matrix = build_corr(df)

# ── Header ────────────────────────────────────────────────────────────────────
st.title("🎧 Spotify Premium — What drives users to upgrade?")
st.caption("Logistic regression analysis on Spotify user survey data")

# ══════════════════════════════════════════════════════════════════════════════
# KPI CARDS
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
# SECTION 1 — Dataset demographics (pie charts)
# ══════════════════════════════════════════════════════════════════════════════
# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — Dataset demographics (final polished)
# ══════════════════════════════════════════════════════════════════════════════
st.markdown(
    '<div class="section-title">Dataset demographics — who is in this dataset?</div>',
    unsafe_allow_html=True
)
st.caption("Understand dataset composition before interpreting results.")

demo_options = {
    "Gender": "Gender",
    "Age group": "Age",
    "Usage period": "spotify_usage_period",
    "Content preference": "preferred_listening_content",
    "Time slot": "music_time_slot",
}

# Layout: left (controls) | right (chart)
left_col, right_col = st.columns([1, 2])

# ── LEFT: selection ───────────────────────────────────────────────────────────
with left_col:
    selected_demo = st.selectbox(
        "Select feature",
        list(demo_options.keys())
    )

# Prepare data
demo_col = demo_options[selected_demo]
counts = df[demo_col].value_counts(dropna=True)
counts = counts.sort_values(ascending=False)
counts.index = counts.index.astype(str)

# ── RIGHT: chart ──────────────────────────────────────────────────────────────
with right_col:
    fig, ax = plt.subplots(figsize=(5, 4))

    fig.patch.set_facecolor("#0e0e0e")
    ax.set_facecolor("#0e0e0e")

    wedges, _, autotexts = ax.pie(
        counts.values,
        labels=None,  # remove clutter inside
        autopct=lambda p: f"{p:.0f}%" if p > 5 else "",  # hide tiny slices
        startangle=90,
        textprops={"color": "white", "fontsize": 9}
    )

    # Style percentages
    for at in autotexts:
        at.set_color("white")
        at.set_fontsize(9)

    # Legend outside (clean + readable)
    legend = ax.legend(
        wedges,
        [f"{label} ({count})" for label, count in zip(counts.index, counts.values)],
        title=selected_demo,
        loc="center left",
        bbox_to_anchor=(1, 0.5),
        fontsize=9,
        labelcolor="white",
        facecolor="#1a1a2e",
        framealpha=0.3
    )

    legend.get_title().set_color("white")
    legend.get_title().set_fontsize(10)
    legend.get_title().set_fontweight("bold")

    ax.set_title(f"{selected_demo} distribution", color="white", fontsize=12)

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

# ── Note ──────────────────────────────────────────────────────────────────────
st.markdown(
    '<div class="note-box">⚠ Groups with very small counts can show misleading percentages. '
    'Always interpret percentages alongside the number of users in each group.</div>',
    unsafe_allow_html=True
)
# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — Who is willing to upgrade (stacked bars, Yes/No only)
# ══════════════════════════════════════════════════════════════════════════════
st.markdown(
    '<div class="section-title">Who is willing to upgrade? — breakdown by feature</div>',
    unsafe_allow_html=True
)
st.caption(
    "Each bar shows the Yes/No split within that group as a percentage. "
    "Sorted by highest willingness. Check the demographics section above for group sizes — "
    "small groups can show misleading percentages."
)

tabs = st.tabs(["By age", "By gender", "By usage period", "By device", "By content", "By time slot"])
breakdown_cols = [
    "Age", "Gender", "spotify_usage_period",
    "spotify_listening_device", "preferred_listening_content", "music_time_slot"
]

def willingness_bar(data, col):
    ct = pd.crosstab(
        data[col],
        data["premium_sub_willingness"],
        normalize="index"
    ) * 100

    if len(ct) == 0:
        st.info("No data for this selection.")
        return

    for c in ["Yes", "No"]:
        if c not in ct.columns:
            ct[c] = 0

    ct = ct[["Yes", "No"]]
    ct = ct.sort_values("Yes", ascending=True)

    # show count of users in each group next to label
    group_counts = data[col].value_counts()
    ct.index = [
        f"{str(idx)}  (n={group_counts.get(idx, 0)})"
        for idx in ct.index
    ]

    bar_colors = {"Yes": "#1DB954", "No": "#e05c5c"}

    fig, ax = plt.subplots(figsize=(8, max(3, len(ct) * 0.55)))
    fig.patch.set_facecolor("#0e0e0e")
    ax.set_facecolor("#0e0e0e")

    lefts = np.zeros(len(ct))
    for response in ["Yes", "No"]:
        vals = ct[response].values
        bars = ax.barh(ct.index.astype(str), vals, left=lefts,
                       color=bar_colors[response], label=response)
        for bar, val, left in zip(bars, vals, lefts):
            if val >= 8:
                ax.text(
                    left + val / 2,
                    bar.get_y() + bar.get_height() / 2,
                    f"{val:.0f}%",
                    ha="center", va="center",
                    color="white", fontsize=9, fontweight="bold"
                )
        lefts += vals

    ax.set_xlabel("% of group", color="white")
    ax.set_xlim(0, 100)
    ax.tick_params(colors="white")
    ax.legend(loc="lower right", framealpha=0.2, labelcolor="white", facecolor="#1a1a2e")
    for spine in ax.spines.values():
        spine.set_edgecolor("#333")
    st.pyplot(fig)
    plt.close()

for tab, col in zip(tabs, breakdown_cols):
    with tab:
        willingness_bar(df, col)

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — Drill down
# ══════════════════════════════════════════════════════════════════════════════
st.markdown(
    '<div class="section-title">Drill down — within a specific group, who upgrades?</div>',
    unsafe_allow_html=True
)
st.caption("Pick a group to zoom into, then pick a second feature to break it down by.")

drill_options = {
    "Gender":             "Gender",
    "Age group":          "Age",
    "Content preference": "preferred_listening_content",
    "Usage period":       "spotify_usage_period",
    "Time slot":          "music_time_slot",
}

dcol1, dcol2, dcol3 = st.columns(3)
with dcol1:
    primary_label = st.selectbox("Zoom into", list(drill_options.keys()), key="drill_primary")
    primary_col   = drill_options[primary_label]
with dcol2:
    primary_val = st.selectbox(
        f"Select {primary_label.lower()}",
        sorted(df[primary_col].dropna().unique().tolist()),
        key="drill_val"
    )
with dcol3:
    secondary_label = st.selectbox(
        "Break down by",
        [k for k in drill_options.keys() if k != primary_label],
        key="drill_secondary"
    )
    secondary_col = drill_options[secondary_label]

drill_df      = df[df[primary_col] == primary_val]
drill_total   = len(drill_df)
drill_willing = (drill_df["premium_sub_willingness"] == "Yes").sum()
drill_pct     = round(drill_willing / drill_total * 100, 1) if drill_total > 0 else 0

st.caption(
    f"**{drill_total:,}** users where {primary_label} = **{primary_val}** · "
    f"**{drill_pct}%** willing to upgrade within this group"
)

if drill_total < 30:
    st.warning(
        f"⚠ Only {drill_total} users in this group. "
        "Percentages may not be statistically reliable — interpret with caution."
    )

willingness_bar(drill_df, secondary_col)

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — Correlation heatmap (before model)
# ══════════════════════════════════════════════════════════════════════════════
st.markdown(
    '<div class="section-title">Feature correlation with premium willingness — before modelling</div>',
    unsafe_allow_html=True
)
st.caption(
    "Measures each feature's direct relationship with willingness, in isolation. "
    "+1 = moves together, -1 = moves opposite, 0 = no relationship."
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
# SECTION 5 — Feature importance (after model)
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
# SECTION 6 — Confusion matrix
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