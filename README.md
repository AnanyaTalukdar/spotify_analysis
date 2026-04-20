# Spotify Premium Drivers

An end-to-end machine learning project that analyses Spotify user survey data to identify which behaviours and demographics drive willingness to upgrade to Premium.

---

## Live dashboard

https://app-triall.streamlit.app/

---

## Project structure

```
spotify_user_analysis/
├── data/
│   ├── raw/                  Spotify_data.xlsx (source data)
│   └── processed/            cleaned data (future use)
├── notebooks/
│   ├── eda1.ipynb            EDA, feature analysis, logistic regression model
│   └── eda2.ipynb            correlation heatmap and advanced EDA
├── app.py                    Streamlit dashboard
├── requirements.txt          Python dependencies
└── .gitignore
```

---

## Tech stack

- **Python** — pandas, numpy, scikit-learn, matplotlib, seaborn
- **Streamlit** — dashboard and deployment
- **GitHub** — version control and Streamlit Cloud integration

---

## Run locally

```bash
# 1. Clone the repo
git clone https://github.com/your-username/spotify_user_analysis.git
cd spotify_user_analysis

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the dashboard
streamlit run app.py
```

---
