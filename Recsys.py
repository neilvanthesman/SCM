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


# -------------------------------------------------
# Page Config
# -------------------------------------------------
st.set_page_config(
    page_title="Spotify Song Recommender",
    page_icon="✧",
    layout="wide"
)

st.title("✧ Spotify Explore")
st.caption("Discover songs through audio similarity")

with st.expander("(ℹ) How is this different from Spotify?"):

    st.markdown("""
Spotify's recommendation engine uses listening history, popularity, collaborative filtering, and many proprietary factors.
This project recommends songs solely by technical similarity without gathering your personal information.
[Find out how it works on GitHub](https://github.com/neilvanthesman/Machine-Learning/blob/main/README.md)
""")
# -------------------------------------------------
# Session State
# -------------------------------------------------
if "visitor_id" not in st.session_state:
    st.session_state.visitor_id = str(uuid.uuid4())[:8]

if "recommendations" not in st.session_state:
    st.session_state.recommendations = None

if "query_song" not in st.session_state:
    st.session_state.query_song = ""


# -------------------------------------------------
# Google Sheets
# -------------------------------------------------
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


# -------------------------------------------------
# Download Dataset
# -------------------------------------------------
CSV_URL = (
    "https://raw.githubusercontent.com/"
    "neilvanthesman/Machine-Learning/refs/heads/main/spotify.csv"
)

CSV_PATH = "spotify.csv"

if not os.path.exists(CSV_PATH):
    urllib.request.urlretrieve(CSV_URL, CSV_PATH)


# -------------------------------------------------
# Load Data
# -------------------------------------------------
@st.cache_data
def load_data():

    data = pd.read_csv(CSV_PATH)

    all_features = [
        "danceability",
        "liveness",
        "valence",
        "energy",
        "instrumentalness",
        "acousticness",
        "loudness",
        "tempo"
    ]

    data[all_features] = data[all_features].fillna(0)
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

    return data


data = load_data()


# -------------------------------------------------
# Recommendation Function
# -------------------------------------------------
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

    candidate_mask = (
        data["artists"].str.lower() != origin_artist
    )

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

    recommendations["song_id"] = recommendations["id"]

    return recommendations


# -------------------------------------------------
# Song Inputs
# -------------------------------------------------
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


# -------------------------------------------------
# Settings
# -------------------------------------------------
left_settings, right_settings = st.columns([1, 1])

with left_settings:

    st.subheader("Audio Features")

    st.info(
        "Recommended: choose at least 3 features. [Learn more about Audio Features](https://developer.spotify.com/documentation/web-api/reference/get-audio-features)\n"
        "Loudness and Tempo are experimental and generally not recommended."
    )


    selected_features = st.multiselect(
        "Select audio features used for similarity",
        options=[
            "danceability",
            "liveness",
            "valence",
            "energy",
            "instrumentalness",
            "acousticness",
            "loudness",
            "tempo"
        ],
        default=[
            "danceability",
            "energy",
            "valence",
            "acousticness",
            "instrumentalness"
        ]
    )

    if len(selected_features) == 0:
        st.warning("Please select at least one audio feature.")
        st.stop()

    if len(selected_features) < 3:
        st.warning(
            "Using fewer than 3 features may produce less reliable recommendations."
        )

with right_settings:

    st.subheader("Settings")

    top_n = st.slider(
        "Number of recommendations",
        1,
        20,
        10
    )


# -------------------------------------------------
# Create Audio Matrix
# -------------------------------------------------
scaler = StandardScaler()

audio_matrix = scaler.fit_transform(
    data[selected_features]
)


# -------------------------------------------------
# Recommend Button
# -------------------------------------------------
if st.button("Recommend Songs"):

    query = f"{artist} <> {song}"

    st.session_state.query_song = query

    with st.spinner("Finding similar songs..."):

        st.session_state.recommendations = get_recommendations(
            query,
            top_n
        )


# -------------------------------------------------
# Display Recommendations
# -------------------------------------------------
recommendations = st.session_state.recommendations
if (
    recommendations is None
    and st.session_state.query_song != ""
):
    st.warning(
        "Song not found.\n\nTry another artist or title."
    )
if recommendations is not None:

    left_col, right_col = st.columns([2, 1.5])

    with left_col:

        st.subheader("Recommendations")
        for _, row in recommendations.iterrows():
        
            with st.container(border=True):
        
                st.markdown(
                    f"####  {row['name']}"
                )
        
                st.caption(
                    f"by {row['artists']}"
                )

    
    with right_col:

        st.subheader("Which songs do you like?")
        st.markdown(
        "*Feedback is greatly appreciated to improve this app.* "
)
        with st.form("feedback_form"):

            liked_song_ids = []

            for _, row in recommendations.iterrows():

                label = f"{row['artists']} - {row['name']}"

                checked = st.checkbox(
                    label,
                    key=f"song_{row['song_id']}"
                )

                if checked:
                    liked_song_ids.append(row["song_id"])

            submitted = st.form_submit_button(
                "Submit Feedback"
            )

            if submitted:

                record = {
                    "Timestamp":
                        datetime.now().strftime(
                            "%Y-%m-%d %H:%M:%S"
                        ),

                    "Id":
                        st.session_state.visitor_id,

                    "QuerySong":
                        st.session_state.query_song,

                    "selected_features":
                        json.dumps(selected_features),

                    "Recommendations":
                        json.dumps(
                            recommendations["song_id"].tolist()
                        ),

                    "Liked Songs":
                        json.dumps(liked_song_ids)
                }

                sheet.append_row(
                    list(record.values())
                )

                st.success(
                    "Feedback submitted successfully!"
                )
st.divider()

st.caption(
    "Spotify Explore - Build with Python & Streamlit"
)
