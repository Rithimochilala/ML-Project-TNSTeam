import streamlit as st
import joblib
import pandas as pd


# ------------------------------------------------------------------
# STEP 1: CONFIGURE PAGE & UI HEADERS
# ------------------------------------------------------------------
st.set_page_config(
    page_title="Music Listener Segmentation",
    page_icon="🎧",
    layout="centered"
)

st.title("🎧 Music Listener Segmentation")
st.markdown("Discover your listener type using Unsupervised Machine Learning (K-Means Clustering).")
st.divider()


# ------------------------------------------------------------------
# STEP 2: LOAD THE TRAINED MODEL AND SCALER
# ------------------------------------------------------------------

try:
    model = joblib.load("model.pkl")
    scaler = joblib.load("scaler.pkl")
except Exception as e:
    st.error(f"Could not load model files: {e}")
    st.stop()

cluster_labels = {
    2: "Casual Listener",
    0: "Music Explorer",
    1: "Heavy Listener"
}

segment_descriptions = {
    "Casual Listener": "Lower listening activity and a higher skip rate — dips in occasionally rather than deep engagement.",
    "Music Explorer": "Moderate listening activity with a fair number of playlists — enjoys music regularly and explores a bit.",
    "Heavy Listener": "High listening activity, low skip rate, and many playlists — deeply engaged, music is a big part of the day."
}


# ------------------------------------------------------------------
# STEP 3: RENDER INTERACTIVE INPUT FORM
# ------------------------------------------------------------------

with st.form("listener_form"):
    col1, col2 = st.columns(2)

    with col1:
        listening_hours = st.slider("Listening Hours per Week", min_value=0, max_value=50, value=10)
        songs_per_day = st.slider("Songs per Day", min_value=0, max_value=150, value=20)

    with col2:
        skip_rate = st.slider("Skip Rate (%)", min_value=0, max_value=100, value=40)
        playlist_count = st.number_input("Playlist Count", min_value=0, max_value=100, value=3)

    submit_btn = st.form_submit_button("Find My Listener Segment", width="stretch")


# ------------------------------------------------------------------
# STEP 4: HANDLE SUBMISSION & PREDICTION
# ------------------------------------------------------------------
if submit_btn:
    input_df = pd.DataFrame([{
        "listening_hours_per_week": listening_hours,
        "songs_per_day": songs_per_day,
        "skip_rate": skip_rate,
        "playlist_count": playlist_count
    }])

    scaled_input = scaler.transform(input_df)

    cluster_number = int(model.predict(scaled_input)[0])
    segment_name = cluster_labels[cluster_number]
    st.subheader("Your Listener Segment")
    st.metric("Segment", segment_name)
    st.info(segment_descriptions[segment_name])

    with st.expander("See technical details"):
        st.write(f"Raw cluster number assigned by K-Means: **{cluster_number}**")
        st.write("Your input:")
        st.dataframe(input_df, width="stretch")