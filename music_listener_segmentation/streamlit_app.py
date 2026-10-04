#moduleimports
import streamlit as st
import joblib
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from fpdf import FPDF
import io

#page config

st.set_page_config(page_title="Listener Segmentation", page_icon="🎚️", layout="wide")

#load model + scaler
try:
    model = joblib.load("model.pkl")
    scaler = joblib.load("scaler.pkl")
except Exception as e:
    st.error(f"Could not load model files: {e}")
    st.stop()

cluster_labels = {2: "Casual Listener", 0: "Music Explorer", 1: "Heavy Listener"}

segment_descriptions = {
    "Casual Listener": "Listens in short bursts and skips often. Music is background, not a routine.",
    "Music Explorer": "Listens regularly, skips sometimes, keeps a handful of playlists on rotation.",
    "Heavy Listener": "Listens for hours daily, rarely skips, and curates playlists constantly."
}

cluster_profiles = {
    "Casual Listener":  {"hours": 4.8,  "songs": 19.1,  "skip": 49.8, "playlists": 4.1},
    "Music Explorer":   {"hours": 14.5, "songs": 58.0,  "skip": 24.9, "playlists": 15.6},
    "Heavy Listener":   {"hours": 30.4, "songs": 120.6, "skip": 7.8,  "playlists": 29.0},
}

try:
    training_df = pd.read_csv("music_listeners.csv")
    training_scaled = scaler.transform(training_df[
        ["listening_hours_per_week", "songs_per_day", "skip_rate", "playlist_count"]
    ])
    training_df["cluster"] = model.predict(training_scaled)
    training_df["segment"] = training_df["cluster"].map(cluster_labels)
    has_training_data = True
except Exception:
    has_training_data = False

#theme toggle + color tokens
if "theme" not in st.session_state:
    st.session_state.theme = "Light"

top_l, top_r = st.columns([5, 1])
with top_r:
    st.session_state.theme = st.radio(
        "Theme", ["Light", "Dark"], horizontal=True,
        index=0 if st.session_state.theme == "Light" else 1,
        label_visibility="collapsed"
    )

if st.session_state.theme == "Light":
    T = dict(bg="#FFFFFF", surface="#FFFFFF", surface_alt="#F6F7F9", border="#E4E6EA",
              text="#1A1C20", muted="#6B7280", accent="#D62828", accent_text="#FFFFFF",
              plot_bg="#FFFFFF", grid="#E4E6EA", teal="#1F9D8A")
else:
    T = dict(bg="#121418", surface="#1A1D23", surface_alt="#20242B", border="#2B303A",
              text="#F0F1F3", muted="#9098A6", accent="#E8534F", accent_text="#1A1D23",
              plot_bg="#1A1D23", grid="#2B303A", teal="#45C9B0")

segment_colors = {
    "Casual Listener": T["muted"],
    "Music Explorer": T["teal"],
    "Heavy Listener": T["accent"],
}

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Inter:wght@400;500;600&display=swap');

html, body, [class*="css"] {{ font-family: 'Inter', sans-serif; }}
.stApp {{ background: {T['bg']}; color: {T['text']}; }}
.block-container {{ max-width: 1000px; padding-top: 1.5rem; padding-bottom: 3rem; }}
h1, h2, h3 {{ font-family: 'Space Grotesk', sans-serif; color: {T['text']}; }}

.app-eyebrow {{ color: {T['muted']}; font-size: 0.9rem; margin-bottom: 0.2rem; }}
.app-title {{ font-family: 'Space Grotesk', sans-serif; font-size: 2.1rem; font-weight: 700;
    margin: 0 0 0.4rem 0; letter-spacing: -0.01em; color: {T['text']}; }}
.app-sub {{ color: {T['muted']}; font-size: 1rem; max-width: 62ch; line-height: 1.5; margin-bottom: 1.5rem; }}

.panel-label {{ font-size: 0.85rem; color: {T['muted']}; margin-bottom: 0.9rem; font-weight: 600; }}

/* Native Streamlit bordered container, restyled to match our theme */
div[data-testid="stVerticalBlockBorderWrapper"] {{
    background: {T['surface']}; border: 1px solid {T['border']} !important;
    border-radius: 8px; padding: 0.4rem 0.3rem;
}}

.meter-row {{ display: flex; align-items: flex-end; gap: 14px; height: 140px; padding: 0 4px; }}
.meter-col {{ flex: 1; display: flex; flex-direction: column; align-items: center; justify-content: flex-end; height: 100%; }}
.meter-track {{ width: 100%; max-width: 46px; height: 100%; background: {T['surface_alt']};
    border-radius: 3px; display: flex; align-items: flex-end; overflow: hidden; border: 1px solid {T['border']}; }}
.meter-fill {{ width: 100%; background: {T['accent']}; border-radius: 2px 2px 0 0; transition: height 0.3s ease; }}
.meter-tag {{ font-size: 0.72rem; color: {T['muted']}; margin-top: 0.5rem; text-align: center; line-height: 1.2; }}
.meter-val {{ font-size: 0.78rem; color: {T['text']}; font-weight: 600; margin-top: 0.15rem; }}

.result-label {{ color: {T['accent']}; font-size: 0.9rem; font-weight: 600; margin-bottom: 0.3rem; }}
.result-name {{ font-family: 'Space Grotesk', sans-serif; font-size: 1.9rem; font-weight: 700;
    margin-bottom: 0.5rem; color: {T['text']}; }}
.result-desc {{ color: {T['muted']}; font-size: 0.98rem; line-height: 1.5; max-width: 62ch; }}

.compare-row {{ display: grid; grid-template-columns: 130px 1fr 56px; align-items: center; gap: 12px; margin: 0.55rem 0; }}
.compare-label {{ font-size: 0.85rem; color: {T['text']}; }}
.compare-track {{ height: 9px; background: {T['surface_alt']}; border-radius: 5px; overflow: hidden; border: 1px solid {T['border']}; }}
.compare-fill {{ height: 100%; border-radius: 5px; }}
.compare-val {{ font-size: 0.8rem; color: {T['muted']}; text-align: right; }}

.stSlider label, .stNumberInput label {{ color: {T['text']} !important; font-weight: 500; }}

div.stButton > button, div.stDownloadButton > button {{
    background: {T['accent']}; color: {T['accent_text']}; border: none; border-radius: 5px;
    font-weight: 600; padding: 0.6rem 1.4rem; font-family: 'Space Grotesk', sans-serif;
}}
div.stButton > button:hover, div.stDownloadButton > button:hover {{ opacity: 0.88; color: {T['accent_text']}; }}
</style>
""", unsafe_allow_html=True)

#header
st.markdown('<div class="app-eyebrow">Unsupervised Learning · K-Means Clustering</div>', unsafe_allow_html=True)
st.markdown('<div class="app-title">Music Listener Segmentation</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="app-sub">Enter a listener\'s habits below. The model compares them against patterns '
    'learned from real listening data and places them into one of three behavioural segments — '
    'no predefined categories, the groups were discovered by the model itself.</div>',
    unsafe_allow_html=True
)

#input+livemeter
left, right = st.columns([1.1, 1], gap="large")

with left:
    with st.container(border=True):
        st.markdown('<div class="panel-label">Listener habits</div>', unsafe_allow_html=True)
        listening_hours = st.slider("Listening hours per week", 0, 50, 15)
        songs_per_day = st.slider("Songs played per day", 0, 150, 40)
        skip_rate = st.slider("Skip rate (%)", 0, 100, 30)
        playlist_count = st.number_input("Playlist count", 0, 100, 8)
        predict_clicked = st.button("Reveal listener segment")

with right:
    with st.container(border=True):
        st.markdown('<div class="panel-label">Live input signal</div>', unsafe_allow_html=True)
        meters = [
            ("Hours", listening_hours, 50, listening_hours),
            ("Songs/day", songs_per_day, 150, songs_per_day),
            ("Engagement", 100 - skip_rate, 100, f"{100 - skip_rate}%"),
            ("Playlists", playlist_count, 100, playlist_count),
        ]
        bars_html = '<div class="meter-row">'
        for label, raw, maxval, display in meters:
            pct = max(4, min(100, (raw / maxval) * 100)) if maxval else 4
            bars_html += f'''
            <div class="meter-col">
                <div class="meter-track"><div class="meter-fill" style="height:{pct}%;"></div></div>
                <div class="meter-tag">{label}</div>
                <div class="meter-val">{display}</div>
            </div>'''
        bars_html += '</div>'
        st.markdown(bars_html, unsafe_allow_html=True)
        st.markdown(
            f'<div style="color: {T["muted"]}; font-size: 0.78rem; margin-top: 1rem;">'
            '"Engagement" is the inverse of skip rate — a fuller bar means fewer skipped songs.</div>',
            unsafe_allow_html=True
        )

#prediction+result
if predict_clicked:
    input_df = pd.DataFrame([{
        "listening_hours_per_week": listening_hours,
        "songs_per_day": songs_per_day,
        "skip_rate": skip_rate,
        "playlist_count": playlist_count
    }])

    scaled_input = scaler.transform(input_df)
    cluster_number = int(model.predict(scaled_input)[0])
    segment_name = cluster_labels[cluster_number]

    st.markdown("")
    with st.container(border=True):
        st.markdown('<div class="result-label">Predicted segment</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="result-name">{segment_name}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="result-desc">{segment_descriptions[segment_name]}</div>', unsafe_allow_html=True)

    #Comparison bars
    with st.container(border=True):
        st.markdown('<div class="panel-label">How you compare to each segment\'s average listener</div>',
                     unsafe_allow_html=True)
        feature_meta = [
            ("Listening hours/week", "hours", 50),
            ("Songs per day", "songs", 150),
            ("Skip rate %", "skip", 100),
            ("Playlist count", "playlists", 100),
        ]
        for seg_name, profile in cluster_profiles.items():
            is_match = seg_name == segment_name
            bar_color = T["accent"] if is_match else T["border"]
            label_color = T["accent"] if is_match else T["muted"]
            st.markdown(
                f'<div style="color: {label_color}; font-size: 0.85rem; font-weight: 600; '
                f'margin: 1rem 0 0.3rem 0;">{seg_name}{" — your match" if is_match else ""}</div>',
                unsafe_allow_html=True
            )
            for disp_label, key, maxval in feature_meta:
                pct = max(3, min(100, (profile[key] / maxval) * 100))
                st.markdown(f'''
                <div class="compare-row">
                    <div class="compare-label">{disp_label}</div>
                    <div class="compare-track"><div class="compare-fill" style="width:{pct}%; background:{bar_color};"></div></div>
                    <div class="compare-val">{profile[key]:.1f}</div>
                </div>''', unsafe_allow_html=True)

    #Scatter plot
    distances = {}
    if has_training_data:
        with st.container(border=True):
            st.markdown(
                '<div class="panel-label">Where you land among all 30 training listeners '
                '(hours vs. skip rate)</div>', unsafe_allow_html=True
            )
            fig = go.Figure()
            for seg_name in cluster_labels.values():
                subset = training_df[training_df["segment"] == seg_name]
                fig.add_trace(go.Scatter(
                    x=subset["listening_hours_per_week"], y=subset["skip_rate"],
                    mode="markers", name=seg_name,
                    marker=dict(size=10, color=segment_colors[seg_name], opacity=0.8,
                                line=dict(width=1, color=T["bg"])),
                    hovertemplate=f"{seg_name}<br>Hours: %{{x}}<br>Skip rate: %{{y}}%<extra></extra>"
                ))
            fig.add_trace(go.Scatter(
                x=[listening_hours], y=[skip_rate], mode="markers+text", name="You",
                marker=dict(size=18, color=T["bg"], line=dict(width=3, color=segment_colors[segment_name]), symbol="star"),
                text=["You"], textposition="top center", textfont=dict(color=T["text"], size=12),
                hovertemplate=f"Your input<br>Hours: {listening_hours}<br>Skip rate: {skip_rate}%<extra></extra>"
            ))
            fig.update_layout(
                paper_bgcolor=T["plot_bg"], plot_bgcolor=T["plot_bg"],
                font=dict(color=T["text"], family="Inter"),
                xaxis=dict(title="Listening hours per week", gridcolor=T["grid"], zerolinecolor=T["grid"]),
                yaxis=dict(title="Skip rate (%)", gridcolor=T["grid"], zerolinecolor=T["grid"]),
                legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor=T["border"], borderwidth=1),
                margin=dict(l=10, r=10, t=10, b=10), height=380
            )
            st.plotly_chart(fig, width="stretch")

        #Distance breakdown
        with st.container(border=True):
            st.markdown('<div class="panel-label">Distance to each segment\'s center (closer = more similar)</div>',
                         unsafe_allow_html=True)
            centroids = model.cluster_centers_
            for cnum, label in cluster_labels.items():
                distances[label] = float(np.linalg.norm(scaled_input[0] - centroids[cnum]))
            max_dist = max(distances.values()) or 1
            for seg_name, dist in sorted(distances.items(), key=lambda x: x[1]):
                is_match = seg_name == segment_name
                pct = max(5, 100 - (dist / max_dist) * 100)
                color = T["accent"] if is_match else T["border"]
                st.markdown(f'''
                <div class="compare-row">
                    <div class="compare-label">{seg_name}</div>
                    <div class="compare-track"><div class="compare-fill" style="width:{pct}%; background:{color};"></div></div>
                    <div class="compare-val">{dist:.2f}</div>
                </div>''', unsafe_allow_html=True)
            st.markdown(
                f'<div style="color: {T["muted"]}; font-size: 0.78rem; margin-top: 0.8rem;">'
                'Distance is measured in scaled feature space, not raw units — the segment with the '
                'shortest distance is the one K-Means assigns you to.</div>',
                unsafe_allow_html=True
            )

    #PDF export feature (raw details)
    with st.container(border=True):
        st.markdown('<div class="panel-label">Raw cluster number, input, and export</div>', unsafe_allow_html=True)
        st.write(f"K-Means cluster: **{cluster_number}**")
        st.dataframe(input_df, width="stretch")

        def build_pdf():
            from fpdf.enums import XPos, YPos
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Helvetica", "B", 16)
            pdf.cell(0, 10, "Music Listener Segmentation - Result", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            pdf.set_font("Helvetica", "", 11)
            pdf.ln(4)
            pdf.cell(0, 8, f"Predicted Segment: {segment_name}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            pdf.multi_cell(0, 7, segment_descriptions[segment_name])
            pdf.ln(4)
            pdf.set_font("Helvetica", "B", 12)
            pdf.cell(0, 8, "Input Provided", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            pdf.set_font("Helvetica", "", 11)
            pdf.cell(0, 7, f"Listening hours/week: {listening_hours}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            pdf.cell(0, 7, f"Songs per day: {songs_per_day}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            pdf.cell(0, 7, f"Skip rate: {skip_rate}%", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            pdf.cell(0, 7, f"Playlist count: {playlist_count}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            if distances:
                pdf.ln(4)
                pdf.set_font("Helvetica", "B", 12)
                pdf.cell(0, 8, "Distance to Segment Centers", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
                pdf.set_font("Helvetica", "", 11)
                for seg_name_d, dist in sorted(distances.items(), key=lambda x: x[1]):
                    pdf.cell(0, 7, f"{seg_name_d}: {dist:.2f}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
            return bytes(pdf.output())

        pdf_bytes = build_pdf()
        st.download_button(
            "Download result as PDF", data=pdf_bytes,
            file_name=f"listener_segment_{segment_name.replace(' ', '_').lower()}.pdf",
            mime="application/pdf"
        )