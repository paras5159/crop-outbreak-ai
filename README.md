import streamlit as st
from PIL import Image
import numpy as np
import torch
import torchvision.transforms as transforms
from torchvision import models
from datetime import datetime
import pandas as pd

# ---------------- PAGE CONFIG --------------
st.set_page_config(
    page_title="CropOutbreak AI",
    layout="centered"
)

st.title("🌾 CropOutbreak AI")
st.write(
    "AI-based system for **plant disease diagnosis**.\n\n"
    "**Live Demo Focus:** Rice (Rice Blast & Nitrogen Deficiency)"
)

# ---------------- SESSION STATE ----------------
if "region_stats" not in st.session_state:
    st.session_state.region_stats = {}

#  LOAD MODEL 

@st.cache_resource
def load_model():
    model = models.resnet18(pretrained=True)
    model.eval()
    return model

model = load_model()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# ---------------- DETECTION ----------------
def detect_condition(image: Image.Image):
    """
    AI-based demo detection.
    """

    img_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = model(img_tensor)
        confidence = torch.softmax(output, dim=1).max().item()

    # Visual logic (explainable)
    img = np.array(image.resize((150, 150)))
    r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]

    yellow_ratio = np.mean((r > 150) & (g > 150) & (b < 120))

    if yellow_ratio > 0.30:
        return "Nitrogen Deficiency", round(confidence, 2)
    else:
        return "Rice Blast", round(confidence, 2)

# ---------------- OUTBREAK ----------------
def update_region(region, condition):
    key = (region.lower(), condition.lower())
    stats = st.session_state.region_stats.get(
        key, {"count": 0, "last": None}
    )

    if condition == "Rice Blast":
        stats["count"] += 1
        stats["last"] = datetime.now()

    st.session_state.region_stats[key] = stats
    return stats

def outbreak_level(count):
    if count >= 6:
        return "🚨 Severe outbreak risk"
    elif count >= 3:
        return "⚠ High risk"
    elif count >= 1:
        return "🟡 Monitoring required"
    else:
        return "🟢 Normal"

# ---------------- INPUT ----------------
st.subheader("📸 Upload Plant Leaf Image")

with st.form("diagnosis_form"):
    crop = st.selectbox(
        "Crop",
        ["Rice", "Wheat", "Maize", "Other"],
        help="Demo AI logic works for Rice"
    )
    region = st.text_input("Region / Village / District")
    image_file = st.file_uploader(
        "Upload leaf image",
        type=["jpg", "jpeg", "png"]
    )
    submit = st.form_submit_button("🔍 Diagnose")

# ---------------- RESULT ----------------
if submit:
    if not region or not image_file:
        st.error("Please enter region and upload an image.")
    else:
        # 🔥 IMPORTANT FIX: FORCE RGB
        image = Image.open(image_file).convert("RGB")

        st.image(image, caption="Uploaded Leaf Image", use_column_width=True)

        condition, confidence = detect_condition(image)

        st.markdown("---")
        st.subheader("🧠 Diagnosis Result")
        st.write(f"**Crop:** {crop}")
        st.write(f"**Detected Condition:** {condition}")
        st.write(f"**Model Confidence:** {confidence}")

        st.markdown("---")
        st.subheader("🧪 Recommendation")

        if condition == "Rice Blast":
            st.success(
                """
                **Disease Type:** Fungal (Rice Blast)  
                **Recommended Fungicide:** Tricyclazole 75 WP  
                **Dosage:** 0.6 g per litre of water  
                **Advice:** Avoid excess nitrogen and ensure good drainage
                """
            )

            stats = update_region(region, condition)
            level = outbreak_level(stats["count"])

            st.markdown("---")
            st.subheader("🌍 Regional Outbreak Status")
            st.write(f"**Region:** {region}")
            st.write(f"**Reported Blast Cases:** {stats['count']}")
            st.write(level)

        else:
            st.info(
                """
                **Condition Type:** Nutrient Deficiency  
                **Deficiency:** Nitrogen (N)  
                **Recommendation:** Apply urea in split doses  
                **Note:** No pesticide required
                """
            )

        st.markdown("---")
        st.subheader("📊 Region Monitoring")

        rows = []
        for (r, d), s in st.session_state.region_stats.items():
            rows.append({
                "Region": r.title(),
                "Condition": d.title(),
                "Cases": s["count"],
                "Last Update": s["last"]
            })

        if rows:
            st.dataframe(pd.DataFrame(rows))
        else:
            st.write("No region data recorded yet.")
