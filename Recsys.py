# -*- coding: utf-8 -*-

import os
import uuid
import json
import urllib.request
from datetime import datetime

import pandas as pd
import streamlit as st
import gspread

from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from google.oauth2.service_account import Credentials


# ─────────────────────────────────────────────────────────────
# Page Config
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Spotify Song Recommender",
    page_icon="🎵",
    layout="wide"
)

st.title("🎵 Spotify Song Recommender")

st.write(
    "Enter an artist and song title to receive similar songs."
)


# ─────────────────────────────────────────────────────────────
# Visitor ID
# ─────────────────────────────────────────────────────────────
if "visitor_id" not in st.session_state:
    st.session_state.visitor_id = str(uuid.uuid4())[:8]


# ─────────────────────────────────────────────────────────────
# Google Sheets Connection
# ─────────────────────────────────────────────────────────────
SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive"
]

credentials = Credentials.from_service_account_info(
    st.secrets["gcp_service_account"],
    scopes=SCOPES
)

gc = gspread.authorize(credentials)

sheet = gc.open_by_key(
    "1ml2nWmWy8s0lFLmeBKhIzUT9UM_ZDhYu3Xu1Pg6e_d4"
).sheet1


# ─────────────────────────────────────────────────────────────
# Download CSV if needed
# ─────────────────────────────────────────────────────────────
CSV_URL = (
    "https://raw.githubusercontent.com/"
    "neilvanthesman/Machine-Learning/refs/heads/main/spotify.csv"
)

CSV_PATH = "spotify.csv"

if not os.path.exists(CSV_PATH):
    with st.spinner("Downloading dataset..."):
        urllib.request.urlretrieve(CSV_URL, CSV_PATH)


# ─────────────────────────────────────────────────────────────
# Load Data
# ─────────────────────────────────────────────────────────────
@st.cache_data
def load_data():

    data = pd.read_csv(CSV_PATH)

    audio_features = [
        "energy",
        "loudness",
        "acousticness"
    ]

    data[audio_features] = data[audio_features].fillna(0)
    data["artists"] = data["artists"].fillna("")

    data["artists"] = (
        data["artists"]
        .str.replace("['", "", regex=False)
        .str.replace("']", "", regex=False)
        .str.replace("'", "", regex=False)
    )

    data["combined_name"] = (
        data["artists"] + " <> " + data["name"]
    )

    data = data.drop_duplicates(
        subset=["combined_name"]
    ).reset_index(drop=True)

    scaler = StandardScaler()

    audio_matrix = scaler.fit_transform(
        data[audio_features]
    )

    return data, audio_matrix


data, audio_matrix = load_data()


# ─────────────────────────────────────────────────────────────
# Recommendation Function
# ─────────────────────────────────────────────────────────────
def get_recommendations(combined_name_query, top_n=10):

    matches = data[
        data["combined_name"].str.lower()
        == combined_name_query.lower()
    ]

    if matches.empty:
        return None

    idx = matches.index[0]

    origin_artist = data.loc[idx, "artists"].lower()
    origin_mode = data.loc[idx, "mode"]

    # Stage 1: remove same artist
    candidate_mask = (
        data["artists"].str.lower() != origin_artist
    )

    # Stage 2: same mode only
    candidate_mask &= (
        data["mode"] == origin_mode
    )

    candidate_idx = data[candidate_mask].index.tolist()

    if len(candidate_idx) == 0:
        return None

    query_vec = audio_matrix[idx].reshape(1, -1)

    candidate_vecs = audio_matrix[candidate_idx]

    similarities = cosine_similarity(
        query_vec,
        candidate_vecs
    )[0]

    top_local_idx = similarities.argsort()[::-1][:top_n]

    top_global_idx = [
        candidate_idx[i]
        for i in top_local_idx
    ]

    recommendations = data.iloc[top_global_idx].copy()

    recommendations["song_id"] = top_global_idx

    return recommendations


# ─────────────────────────────────────────────────────────────
# Inputs
# ─────────────────────────────────────────────────────────────
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


# ─────────────────────────────────────────────────────────────
# Recommend
# ─────────────────────────────────────────────────────────────
if st.button("Recommend Songs"):

    if artist == "" or song == "":
        st.warning("Please fill both fields.")

    else:

        query = f"{artist} <> {song}"

        recommendations = get_recommendations(
            query,
            top_n
        )

        if recommendations is None:

            st.error("Song not found.")

        else:

            st.success("Recommendations found!")

            display_df = recommendations[
                ["artists", "name"]
            ].copy()

            display_df.columns = [
                "Artist",
                "Track"
            ]

            st.dataframe(
                display_df,
                hide_index=True,
                use_container_width=True
            )

            st.subheader(
                "Which songs do you like?"
            )

            liked_song_ids = []

            for _, row in recommendations.iterrows():

                label = (
                    f"{row['artists']} - "
                    f"{row['name']}"
                )

                if st.checkbox(
                    label,
                    key=f"song_{row['song_id']}"
                ):
                    liked_song_ids.append(
                        int(row["song_id"])
                    )

            if st.button("Submit Feedback"):

                record = {
                    "timestamp":
                        datetime.now().strftime(
                            "%Y-%m-%d %H:%M:%S"
                        ),

                    "visitor_id":
                        st.session_state.visitor_id,

                    "query_song":
                        query,

                    "recommended_song_ids":
                        json.dumps(
                            [
                                int(x)
                                for x in recommendations[
                                    "song_id"
                                ].tolist()
                            ]
                        ),

                    "liked_song_ids":
                        json.dumps(
                            liked_song_ids
                        )
                }

                sheet.append_row(
                    list(record.values())
                )

                st.success(
                    "Feedback submitted successfully!"
                )
