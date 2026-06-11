# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import urllib.request
import os
import sys

pip install streamlit

# ── Download spotify.csv if not already present ──────────────────────────────
CSV_URL  = "https://raw.githubusercontent.com/neilvanthesman/Machine-Learning/refs/heads/main/spotify.csv"
CSV_PATH = os.path.join(os.path.dirname(sys.executable if getattr(sys, 'frozen', False) else __file__), "spotify.csv")

if not os.path.exists(CSV_PATH):
    print("Downloading spotify.csv …")
    urllib.request.urlretrieve(CSV_URL, CSV_PATH)
    print("Download complete.")

data = pd.read_csv(CSV_PATH)

"""# EDA"""

all_audio_features = [
    'danceability', 'energy', 'key', 'loudness', 'mode',
    'valence', 'tempo', 'speechiness',
    'acousticness', 'instrumentalness', 'liveness'
]

Numerical_audio_features = [
    'danceability', 'energy', 'loudness',
    'valence', 'tempo', 'speechiness',
    'acousticness', 'instrumentalness', 'liveness'
]

# mode is intentionally excluded from similarity features —
# it is used as a hard filter in Stage 2 instead
audio_features = [#'danceability',
                  'energy',
                  'loudness',
                  #'valence',
                  #'tempo',
                  'acousticness']

data[audio_features] = data[audio_features].fillna(0)
# data['artist_genres'] = data['artist_genres'].fillna('')
data['artists'] = data['artists'].fillna('')

# Clean the artists column by removing brackets and quotes
data['artists'] = data['artists'].str.replace("['", "", regex=False).str.replace("']", "", regex=False).str.replace("'", "", regex=False)

# Update the combined_name to reflect the cleaned artist names
data['combined_name'] = data['artists'] + ' <> ' + data['name']

# Drop duplicates based on the newly created 'combined_name'
data = data.drop_duplicates(subset=['combined_name']).reset_index(drop=True)

scaler = StandardScaler()
audio_matrix = scaler.fit_transform(data[audio_features])

# Reset index so positional indexing stays consistent
data = data.reset_index(drop=True)

def get_recommendations(combined_name_query, top_n=10):
    """
    Three-stage recommendation:
      Stage 1: Exclude songs by the same artist      (diversity filter)
      Stage 2: Exclude songs with a different mode   (mood filter)
      Stage 3: Rank remaining by audio cosine similarity

    Query format: 'Artist <> Track Title'
    Example:      'Coldplay <> Yellow'
    """
    matches = data[data['combined_name'].str.lower() == combined_name_query.lower()]

    if matches.empty:
        print(f"Song not found: '{combined_name_query}'")
        print("Tip: use format 'Artist <> Track Title'")
        return None

    idx           = matches.index[0]
    origin_artist = data.loc[idx, 'artists'].lower()
    origin_mode   = data.loc[idx, 'mode']

    # ── Stage 1: filter out same-artist tracks ───────────────────────
    candidate_mask = data['artists'].str.lower() != origin_artist

    # ── Stage 2: filter out different-mode tracks ────────────────────
    candidate_mask = candidate_mask & (data['mode'] == origin_mode)

    candidate_idx = data[candidate_mask].index.tolist()

    if len(candidate_idx) == 0:
        print('No candidates remain after filtering. Try a different song.')
        return None

    # ── Stage 3: cosine similarity on audio features only ────────────
    query_vec      = audio_matrix[idx].reshape(1, -1)
    candidate_vecs = audio_matrix[candidate_idx]

    sims = cosine_similarity(query_vec, candidate_vecs)[0]

    top_local_idx  = sims.argsort()[::-1][:top_n]
    top_global_idx = [candidate_idx[i] for i in top_local_idx]
    top_scores     = sims[top_local_idx]

    recommendations = data.iloc[top_global_idx][
        ['combined_name', 'name', 'artists', 'mode']
    ].copy()
    recommendations['similarity_score'] = top_scores
    recommendations = recommendations.reset_index(drop=True)

    return recommendations

import streamlit as st

st.set_page_config(page_title="Spotify Song Recommender")

st.title("🎵 Spotify Song Recommender")
st.write("Enter a song using the format:")

st.code("Artist <> Track Title")

query = st.text_input(
    "Song Query",
    placeholder="Coldplay <> Yellow"
)

top_n = st.slider("Number of recommendations", 1, 20, 10)

if st.button("Recommend"):

    recommendations = get_recommendations(query, top_n)

    if recommendations is not None:
        st.success("Recommendations found!")

        display_df = recommendations[
            ["combined_name", "similarity_score"]
        ].copy()

        display_df["similarity_score"] = (
            display_df["similarity_score"]
            .round(3)
        )

        st.dataframe(display_df)

        csv = recommendations.to_csv(index=False)

        st.download_button(
            "Download CSV",
            csv,
            file_name="recommendations.csv",
            mime="text/csv"
        )
