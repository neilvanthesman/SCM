# -*- coding: utf-8 -*-

import pandas as pd
import os
import urllib.request
import streamlit as st
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

# ── Page Configuration ──────────────────────────────────────────────────────
st.set_page_config(
    page_title="Spotify Song Recommender",
    page_icon="🎵",
    layout="wide"
)

st.title("🎵 Spotify Song Recommender")
st.write(
    "Get similar songs based on audio features.\n\n"
    "Songs by the same artist are excluded, and only songs with the same mode (major/minor) are considered."
)

# ── Download spotify.csv if not already present ─────────────────────────────
CSV_URL = "https://raw.githubusercontent.com/neilvanthesman/Machine-Learning/refs/heads/main/spotify.csv"
CSV_PATH = "spotify.csv"

if not os.path.exists(CSV_PATH):
    with st.spinner("Downloading dataset..."):
        urllib.request.urlretrieve(CSV_URL, CSV_PATH)

# ── Load and Prepare Data ───────────────────────────────────────────────────
@st.cache_data
def load_data():
    data = pd.read_csv(CSV_PATH)

    audio_features = [
        "energy",
        "loudness",
        "acousticness"
    ]

    # Fill missing values
    data[audio_features] = data[audio_features].fillna(0)
    data["artists"] = data["artists"].fillna("")

    # Clean artist column
    data["artists"] = (
        data["artists"]
        .str.replace("['", "", regex=False)
        .str.replace("']", "", regex=False)
        .str.replace("'", "", regex=False)
    )

    # Create combined name
    data["combined_name"] = data["artists"] + " <> " + data["name"]

    # Remove duplicates
    data = data.drop_duplicates(subset=["combined_name"]).reset_index(drop=True)

    # Scale audio features
    scaler = StandardScaler()
    audio_matrix = scaler.fit_transform(data[audio_features])

    return data, audio_matrix


data, audio_matrix = load_data()

# ── Recommendation Function ─────────────────────────────────────────────────
def get_recommendations(combined_name_query, top_n=10):

    matches = data[
        data["combined_name"].str.lower() == combined_name_query.lower()
    ]

    if matches.empty:
        return None

    idx = matches.index[0]
    origin_artist = data.loc[idx, "artists"].lower()
    origin_mode = data.loc[idx, "mode"]

    # Stage 1: exclude same artist
    candidate_mask = data["artists"].str.lower() != origin_artist

    # Stage 2: same mode only
    candidate_mask &= (data["mode"] == origin_mode)

    candidate_idx = data[candidate_mask].index.tolist()

    if len(candidate_idx) == 0:
        return None

    # Stage 3: cosine similarity
    query_vec = audio_matrix[idx].reshape(1, -1)
    candidate_vecs = audio_matrix[candidate_idx]

    similarities = cosine_similarity(query_vec, candidate_vecs)[0]

    top_local_idx = similarities.argsort()[::-1][:top_n]
    top_global_idx = [candidate_idx[i] for i in top_local_idx]
    top_scores = similarities[top_local_idx]

    recommendations = data.iloc[top_global_idx][
        ["artists", "name", "mode"]
    ].copy()

    recommendations["similarity_score"] = top_scores
    recommendations = recommendations.reset_index(drop=True)

    return recommendations


# ── User Interface ──────────────────────────────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    artist = st.text_input(
        "Artist",
        placeholder="Coldplay"
    )

with col2:
    song = st.text_input(
        "Song Title",
        placeholder="Yellow"
    )

top_n = st.slider(
    "Number of recommendations",
    min_value=1,
    max_value=20,
    value=10
)

# ── Recommendation Button ───────────────────────────────────────────────────
if st.button("Recommend Songs"):

    if artist == "" or song == "":
        st.warning("Please fill in both Artist and Song Title.")

    else:
        query = f"{artist} <> {song}"

        recommendations = get_recommendations(query, top_n)

        if recommendations is None:
            st.error("Song not found or no recommendations available.")
            st.info("Example: Artist = Coldplay, Song = Yellow")

        else:
            st.success(f"Top {top_n} recommendations")

            display_df = recommendations.copy()
            display_df["similarity_score"] = (
                display_df["similarity_score"].round(3)
            )

            st.dataframe(
                display_df,
                use_container_width=True
            )

            csv = recommendations.to_csv(index=False)

            st.download_button(
                label="📥 Download CSV",
                data=csv,
                file_name="recommendations.csv",
                mime="text/csv"
            )
