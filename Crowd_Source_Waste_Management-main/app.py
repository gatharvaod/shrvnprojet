import os
import uuid
from pathlib import Path
import pandas as pd
from PIL import Image
import streamlit as st
from predict import predict_from_image

# ----------------------
# Page setup FIRST
# ----------------------
st.set_page_config(page_title="Crowd-Sourced Waste Management", layout="wide")

# Load CSS if present
css_path = Path("style.css")
if css_path.exists():
    st.markdown(f"<style>{css_path.read_text()}</style>", unsafe_allow_html=True)

# ----------------------
# Basic paths
# ----------------------
IMAGES_DIR = "images"
MODELS_DIR = "models"
REPORTS_CSV = "waste_reports.csv"
os.makedirs(IMAGES_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# ----------------------
# Title
# ----------------------
st.title("Crowd-Sourced Waste Management System ♻️")
st.markdown("---")

# ----------------------
# Data helpers
# ----------------------
@st.cache_data
def load_reports():
    cols = [
        "ID", "Location", "Waste_Type", "Description", "Reported_By",
        "Timestamp", "Status", "Is_Valid", "Predicted_Area", "Priority", "Image_Filename"
    ]
    try:
        df = pd.read_csv(REPORTS_CSV)
        for c in cols:
            if c not in df.columns:
                df[c] = None
    except FileNotFoundError:
        df = pd.DataFrame(columns=cols)
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
    return df

@st.cache_data
def save_reports(df: pd.DataFrame):
    df.to_csv(REPORTS_CSV, index=False)

# keep in session
st.session_state.setdefault("reports_df", load_reports())

# ----------------------
# Priority helper
# ----------------------
def priority_from_prediction(top_label: str, top_prob: float):
    # keep same mechanics the project used
    if "Hazardous" in top_label:
        return "Critical (Immediate Action)", "Awaiting Specialized Team Dispatch"
    if top_prob >= 0.75:
        waste_type = st.selectbox(
            "Waste Type",
            ["General Trash", "Construction Debris", "Organic", "Recyclable", "Hazardous"]
        )
        return "High", f"Assigned to {waste_type} Cleanup Crew"
    return "Medium", "Queued for Normal Processing"

# ----------------------
# Report form (two-column)
# ----------------------
st.markdown("## 📝 Submit a Waste Report")

c1, c2 = st.columns(2)
with c1:
    location = st.text_input("📍 Location")
with c2:
    waste_type = st.selectbox(
        "🗑️ Waste Type",
        ["General Trash", "Construction Debris", "Organic", "Recyclable", "Hazardous"]
    )

description = st.text_area("🧾 Description", height=120)

c3, c4 = st.columns(2)
with c3:
    reported_by = st.text_input("👤 Reported By")
with c4:
    image_file = st.file_uploader("📷 Upload Image (optional)", type=["jpg", "jpeg", "png"])

# Preview
if image_file is not None:
    st.image(Image.open(image_file), use_column_width=True, caption="Image preview")

# Submit
if st.button("Submit Report"):
    image_filename = None
    if image_file:
        img = Image.open(image_file).convert("RGB")
        image_filename = f"{uuid.uuid4().hex}.jpg"
        img.save(os.path.join(IMAGES_DIR, image_filename))
        prediction = predict_from_image(img)
        top_label, top_prob = prediction[0]
    else:
        top_label, top_prob = "Unknown", 0.0

    priority, status = priority_from_prediction(top_label, top_prob)
    timestamp = pd.Timestamp.now()

    new_row = {
        "ID": uuid.uuid4().hex[:8],
        "Location": location,
        "Waste_Type": waste_type,
        "Description": description,
        "Reported_By": reported_by,
        "Timestamp": timestamp,
        "Status": "Open",
        "Is_Valid": True,
        "Predicted_Area": None,
        "Priority": priority,
        "Image_Filename": image_filename,
    }

    df = st.session_state["reports_df"]
    df.loc[len(df)] = new_row
    save_reports(df)

    badge = (
        "<span class='badge badge--crit'>Critical</span>" if "Critical" in priority
        else "<span class='badge badge--warn'>High</span>" if "High" in priority
        else "<span class='badge badge--ok'>Medium</span>"
    )
    st.success(
        f"Report saved. Priority: {badge} · Prediction: <b>{top_label}</b> ({top_prob:.0%})",
        icon="✅"
    )

# ----------------------
# Dashboard + filters
# ----------------------
st.markdown("---")
st.markdown("## 📊 Reports Dashboard")

# sidebar filters
with st.sidebar:
    st.markdown("### Filters")
    df_all = st.session_state["reports_df"].copy()
    types = st.multiselect("Waste Type", sorted(df_all["Waste_Type"].dropna().unique().tolist()))
    priorities = st.multiselect("Priority", sorted(df_all["Priority"].dropna().unique().tolist()))
    loc_query = st.text_input("Search Location")
    date_range = st.date_input("Date range", [])

df = df_all
if types:
    df = df[df["Waste_Type"].isin(types)]
if priorities:
    df = df[df["Priority"].isin(priorities)]
if loc_query:
    df = df[df["Location"].astype(str).str.contains(loc_query, case=False, na=False)]
if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
    start = pd.to_datetime(date_range[0])
    end = pd.to_datetime(date_range[1]) + pd.Timedelta(days=1)
    df = df[(df["Timestamp"] >= start) & (df["Timestamp"] < end)]

if df.empty:
    st.info("No reports yet.")
else:
    st.dataframe(df.sort_values("Timestamp", ascending=False), use_container_width=True)
    st.download_button(
        "Download CSV",
        df.to_csv(index=False).encode("utf-8"),
        "waste_reports_filtered.csv",
        "text/csv"
    )
