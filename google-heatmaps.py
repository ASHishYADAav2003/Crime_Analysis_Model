import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

import folium
from folium.plugins import HeatMap
from streamlit_folium import st_folium

# ---------------- Page Config ----------------
st.set_page_config(
    page_title="Crime Analysis Dashboard",
    layout="wide"
)

st.title("🚔 Crime Analysis Dashboard")
st.markdown("### District-wise Crime Analysis & Heatmaps")

# ---------------- Upload CSV ----------------
uploaded_file = st.file_uploader(
    "Upload Crime Dataset CSV",
    type=["csv"]
)

if uploaded_file is None:
    st.info("👆 Please upload the crime CSV file to continue.")
    st.stop()

# ---------------- Load Data ----------------
df = pd.read_csv(uploaded_file)

# ---------------- Preprocessing ----------------
df.columns = df.columns.str.strip().str.lower()

# Forward fill important columns
df["state_name"] = df["state_name"].ffill()
df["district_name"] = df["district_name"].ffill()

# Required columns check
required_columns = {"lat", "lon"}
if not required_columns.issubset(df.columns):
    st.error("❌ CSV must contain 'lat' and 'lon' columns for Google Maps heatmap.")
    st.stop()

# Identify crime columns
exclude_cols = ["year", "state_name", "district_name", "registration_circles", "lat", "lon"]
crime_cols = [c for c in df.columns if c not in exclude_cols]

df[crime_cols] = df[crime_cols].apply(
    pd.to_numeric, errors="coerce"
).fillna(0)

# ---------------- Sidebar Filters ----------------
st.sidebar.header("🎯 Filters")

years = sorted(df["year"].dropna().unique())
selected_year = st.sidebar.selectbox("Select Year", years)

states = sorted(df[df["year"] == selected_year]["state_name"].unique())
selected_state = st.sidebar.selectbox("Select State", states)

# ---------------- Filter Data ----------------
filtered_df = df[
    (df["year"] == selected_year) &
    (df["state_name"] == selected_state)
]

# ---------------- Aggregate District Data ----------------
district_df = filtered_df.groupby(
    ["district_name", "lat", "lon"],
    as_index=False
)[crime_cols].sum()

# ---------------- Crime Index ----------------
district_df["total_crime"] = district_df[crime_cols].sum(axis=1)

scaler = MinMaxScaler()
district_df["crime_index"] = scaler.fit_transform(
    district_df[["total_crime"]]
)

# ---------------- Display Table ----------------
st.subheader(
    f"📍 {selected_state} – District-wise Crime Data ({selected_year})"
)
st.dataframe(district_df)

# =========================================================
# 🔥 HEATMAP 1: NORMAL (SEABORN HEATMAP)
# =========================================================
st.subheader("🔥 Crime Heatmap (Table View)")

heatmap_cols = crime_cols + ["total_crime"]

heatmap_data = district_df.set_index("district_name")[heatmap_cols]

if heatmap_data.empty:
    st.warning("No crime data available.")
else:
    fig, ax = plt.subplots(
        figsize=(12, max(6, len(heatmap_data) * 0.35))
    )
    sns.heatmap(
        heatmap_data,
        cmap="Reds",
        linewidths=0.5,
        ax=ax
    )
    ax.set_xlabel("Crime Type")
    ax.set_ylabel("District")
    ax.set_title(
        f"Crime Heatmap – {selected_state} ({selected_year})"
    )
    st.pyplot(fig)

# =========================================================
# 🗺️ HEATMAP 2: GOOGLE MAPS STYLE (FOLIUM)
# =========================================================
st.subheader("🗺️ Crime Heatmap on Map (Google Maps Style)")

# Prepare map data [lat, lon, intensity]
map_data = district_df[
    ["lat", "lon", "crime_index"]
].values.tolist()

center_lat = district_df["lat"].mean()
center_lon = district_df["lon"].mean()

m = folium.Map(
    location=[center_lat, center_lon],
    zoom_start=6,
    tiles="CartoDB positron"  # Clean Google-like map
)

HeatMap(
    map_data,
    radius=30,
    blur=25,
    min_opacity=0.4,
    gradient={
        0.2: "blue",
        0.4: "lime",
        0.6: "orange",
        0.8: "red"
    }
).add_to(m)

# Optional district markers
for _, row in district_df.iterrows():
    folium.CircleMarker(
        location=[row["lat"], row["lon"]],
        radius=4,
        popup=f"""
        <b>{row['district_name']}</b><br>
        Total Crime: {int(row['total_crime'])}<br>
        Crime Index: {row['crime_index']:.2f}
        """,
        color="red",
        fill=True,
        fill_opacity=0.7
    ).add_to(m)

st_folium(m, width=1200, height=600)

# =========================================================
# 📈 TOP DISTRICTS
# =========================================================
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
