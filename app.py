import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# ---- Page Config-----
st.set_page_config(
    page_title="Crime Analysis Dashboard",
    layout="wide"
)

st.title("🚔 Crime Analysis Dashboard")
st.markdown("### Year-wise and State-wise Crime Heatmap")

# -- Upload CSV
uploaded_file = st.file_uploader(
    "Upload Crime Dataset CSV",
    type=["csv"]
)

if uploaded_file is None:
    st.info("👆 Please upload the crime CSV file to continue.")
    st.stop()

# ----- Load Data --
df = pd.read_csv(uploaded_file)

# ------- Preprocessing ------------
df.columns = df.columns.str.strip().str.lower()

df["state_name"] = df["state_name"].ffill()
df["district_name"] = df["district_name"].ffill()
df["registration_circles"] = df["registration_circles"].ffill()

# Identify crime columns
exclude_cols = ["year", "state_name", "district_name", "registration_circles"]
crime_cols = [c for c in df.columns if c not in exclude_cols]

df[crime_cols] = df[crime_cols].apply(
    pd.to_numeric, errors="coerce"
).fillna(0)

# -- Sidebar Filters ---
st.sidebar.header("🎯 Filters")

years = sorted(df["year"].dropna().unique())
selected_year = st.sidebar.selectbox(
    "Select Year",
    years
)

states = sorted(
    df[df["year"] == selected_year]["state_name"].unique()
)
selected_state = st.sidebar.selectbox(
    "Select State",
    states
)

# ------- Filter Data ---------
filtered_df = df[
    (df["year"] == selected_year) &
    (df["state_name"] == selected_state)
]

# ----- Aggregate at District Level -------
district_df = filtered_df.groupby(
    ["district_name"],
    as_index=False
)[crime_cols].sum()

# ------------- Crime Index -------
district_df["total_crime"] = district_df[crime_cols].sum(axis=1)

scaler = MinMaxScaler()
district_df["crime_index"] = scaler.fit_transform(
    district_df[["total_crime"]]
)

# -------- Display Data ---------
st.subheader(
    f"📍 {selected_state} – District-wise Crime Data ({selected_year})"
)
st.dataframe(district_df)

# ------ Heatmap ------
st.subheader("🔥 Crime Heatmap")

crime_only_cols = district_df.select_dtypes(include=np.number).columns
crime_only_cols = [c for c in crime_only_cols if c != "crime_index"]

heatmap_data = district_df.set_index("district_name")[crime_only_cols]

if heatmap_data.empty:
    st.warning("No crime data available for this selection.")
else:
    fig, ax = plt.subplots(
        figsize=(12, max(6, len(heatmap_data) * 0.35))
    )
    sns.heatmap(
        heatmap_data,
        cmap="Reds",
        ax=ax
    )
    ax.set_xlabel("Crime Type")
    ax.set_ylabel("District")
    ax.set_title(
        f"Crime Heatmap – {selected_state} ({selected_year})"
    )
    st.pyplot(fig)

# ------- Top Districts --------
st.subheader("📈 Top Districts by Crime Index")

top_n = st.slider(
    "Select number of districts",
    min_value=5,
    max_value=25,
    value=10
)

top_districts = district_df.sort_values(
    "crime_index",
    ascending=False
).head(top_n)

st.dataframe(
    top_districts[
        ["district_name", "total_crime", "crime_index"]
    ]
)