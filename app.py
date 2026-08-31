"""
VeriID AI - Identity Verification & Multi-Vector Fraud Detection Dashboard
File: app.py
"""

import streamlit as st
import numpy as np
from PIL import Image, ImageDraw
import json
import tempfile
import os
from datetime import datetime

# Import Internal Modular Engines
from src.ocr_engine import extract_and_validate
from src.ela_engine import analyze_image_tampering
from src.face_engine import match_faces
from src.risk_engine import calculate_risk_score

# =====================================================================
# Page Configuration
# =====================================================================
st.set_page_config(
    page_title="VeriID AI - Multi-Vector Fraud Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .reportview-container { background: #0e1117; }
    .metric-card {
        background-color: #1e2530;
        padding: 15px;
        border-radius: 8px;
        border: 1px solid #2e3846;
        margin-bottom: 10px;
    }
    .badge-pass { color: #00e676; font-weight: bold; }
    .badge-fail { color: #ff5252; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

st.title("🛡️ VeriID AI — Multi-Vector Verification Dashboard")
st.caption("Automated Multi-Modal Fraud Detection: OCR Cross-Validation | Forensic ELA | Biometrics & Liveness")

# =====================================================================
# Helper Utilities
# =====================================================================
def save_uploaded_file(uploaded_file) -> str:
    suffix = os.path.splitext(uploaded_file.name)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getbuffer())
        return tmp.name

def draw_bounding_boxes(image: Image.Image, text_fields: dict) -> Image.Image:
    """Draws visual feedback overlays for detected ID zones."""
    img_draw = image.copy().convert("RGB")
    draw = ImageDraw.Draw(img_draw)
    w, h = img_draw.size

    # Simulated visual anchors for key verification sectors
    draw.rectangle([int(w * 0.05), int(h * 0.25), int(w * 0.35), int(h * 0.85)], outline="#00e676", width=3) # Face Zone
    draw.rectangle([int(w * 0.40), int(h * 0.25), int(w * 0.95), int(h * 0.85)], outline="#2979ff", width=2) # Data Zone
    return img_draw

# =====================================================================
# Sidebar: Upload Controls
# =====================================================================
st.sidebar.header("📁 Document & Capture Ingestion")
doc_file = st.sidebar.file_uploader("Upload ID Document (JPEG/PNG)", type=["jpg", "jpeg", "png"])
live_file = st.sidebar.file_uploader("Upload Live Selfie / Capture", type=["jpg", "jpeg", "png"])

run_verification = st.sidebar.button("🚀 Run Multi-Vector Verification", type="primary", use_container_width=True)

# =====================================================================
# Main Execution Pipeline
# =====================================================================
if doc_file and live_file and run_verification:
    doc_path = save_uploaded_file(doc_file)
    live_path = save_uploaded_file(live_file)

    with st.spinner("Processing Multi-Modal Fraud Pipeline (OCR -> ELA -> FaceNet512 -> Anti-Spoofing)..."):
        # 1. Run Engines
        ocr_result = extract_and_validate(doc_path)
        ela_result = analyze_image_tampering(doc_path)
        face_result = match_faces(doc_path, live_path)

        # 2. Risk Orchestration
        risk_summary = calculate_risk_score(ocr_result, ela_result, face_result)

    # -----------------------------------------------------------------
    # Top Level Executive Verdict Banner
    # -----------------------------------------------------------------
    risk_score = risk_summary["risk_score"]
    verdict = risk_summary["verdict"]

    st.markdown("---")
    if "VERIFIED" in verdict:
        st.success(f"### Verdict: {verdict} | Risk Score: {risk_score}%")
    elif "MANUAL" in verdict:
        st.warning(f"### Verdict: {verdict} | Risk Score: {risk_score}%")
    else:
        st.error(f"### Verdict: {verdict} | Risk Score: {risk_score}%")

    # -----------------------------------------------------------------
    # Section 1: Telemetry & Multi-Vector Cards
    # -----------------------------------------------------------------
    st.subheader("📊 Engine Telemetry & Biometric Metrics")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            label="Biometric Similarity",
            value=f"{face_result.get('similarity_score', 0.0):.1f}%",
            delta="Matched" if face_result.get("is_same_person") else "-Mismatch",
            delta_color="normal" if face_result.get("is_same_person") else "inverse"
        )
    with col2:
        is_live = face_result.get("is_live", True)
        st.metric(
            label="Liveness / Anti-Spoof",
            value="PASSED" if is_live else "FLAGGED",
            delta=f"Spoof Risk: {face_result.get('spoof_confidence', 0.0):.1f}%",
            delta_color="normal" if is_live else "inverse"
        )
    with col3:
        ela_score = ela_result.get("anomaly_score", 0.0)
        st.metric(
            label="Forensic ELA Score",
            value=f"{ela_score:.2f}",
            delta="Clean" if not ela_result.get("tampering_detected") else "-Tampered",
            delta_color="normal" if not ela_result.get("tampering_detected") else "inverse"
        )
    with col4:
        st.metric(
            label="OCR Syntax Integrity",
            value="VALID" if ocr_result.get("is_valid_format") else "INVALID",
            delta=f"{len(ocr_result.get('error_flags', []))} Errors",
            delta_color="normal" if ocr_result.get("is_valid_format") else "inverse"
        )

    # -----------------------------------------------------------------
    # Section 2: Visual Evidence & Inspection Canvas
    # -----------------------------------------------------------------
    st.subheader("🔍 Visual Evidence & Overlay Inspection")
    img_col1, img_col2, img_col3 = st.columns(3)

    with img_col1:
        st.markdown("**Document Segmentation & Bounding-Boxes**")
        pil_doc = Image.open(doc_path)
        annotated_doc = draw_bounding_boxes(pil_doc, ocr_result.get("extracted_fields", {}))
        st.image(annotated_doc, use_container_width=True, caption="Visual Field Anchors (Green=Face, Blue=Data)")

    with img_col2:
        st.markdown("**Error Level Analysis (ELA Map)**")
        if "ela_image" in ela_result and ela_result["ela_image"] is not None:
            st.image(ela_result["ela_image"], use_container_width=True, caption=f"Compression Variance Map ({ela_score:.2f})")
        else:
            st.info("No ELA map rendered.")

    with img_col3:
        st.markdown("**Live Ingestion Biometrics**")
        st.image(live_path, use_container_width=True, caption="Live Capture Frame")

    # -----------------------------------------------------------------
    # Section 3: Extracted Data & Compliance Audit Export
    # -----------------------------------------------------------------
    st.subheader("📑 Extracted Records & Compliance Audit Engine")
    
    rec_col, audit_col = st.columns([1, 1])

    with rec_col:
        st.markdown("**Parsed OCR Document Fields:**")
        fields = ocr_result.get("extracted_fields", {})
        st.json({
            "Document Type": ocr_result.get("doc_type_detected"),
            "Full Name": fields.get("name") or "Extracted via Card Matrix",
            "Date of Birth": fields.get("dob"),
            "Gender": fields.get("gender"),
            "ID Number": fields.get("id_number"),
            "Issue Date": fields.get("issue_date")
        })

    with audit_col:
        st.markdown("**Formal Compliance Audit Package:**")
        audit_payload = {
            "veriid_audit_id": f"AUD-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}",
            "timestamp_utc": datetime.utcnow().isoformat(),
            "final_verdict": verdict,
            "overall_fraud_risk_score": f"{risk_score}%",
            "pipeline_telemetry": {
                "biometrics": {
                    "facial_similarity": f"{face_result.get('similarity_score', 0.0):.2f}%",
                    "same_person_authenticated": face_result.get("is_same_person", False),
                    "anti_spoof_liveness_verified": face_result.get("is_live", True),
                    "spoof_confidence_score": face_result.get("spoof_confidence", 0.0),
                    "liveness_flags": face_result.get("liveness_flags", [])
                },
                "forensics_ela": {
                    "anomaly_score": ela_result.get("anomaly_score", 0.0),
                    "tampering_detected": ela_result.get("tampering_detected", False)
                },
                "ocr_integrity": {
                    "format_valid": ocr_result.get("is_valid_format", False),
                    "document_type": ocr_result.get("doc_type_detected", "Unknown"),
                    "extracted_data": ocr_result.get("extracted_fields", {}),
                    "error_flags": ocr_result.get("error_flags", [])
                }
            },
            "flagged_risk_reasons": risk_summary.get("flagged_reasons", [])
        }

        audit_json = json.dumps(audit_payload, indent=2)
        st.download_button(
            label="📥 Download Formal Compliance Audit (JSON)",
            data=audit_json,
            file_name=f"VeriID_Audit_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json",
            use_container_width=True
        )

    # Clean up temp files
    try:
        os.remove(doc_path)
        os.remove(live_path)
    except Exception:
        pass

else:
    st.info("👈 Please upload an ID Document and a Live Capture from the sidebar and click **Run Multi-Vector Verification**.")