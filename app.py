"""
VeriID AI - Multi-Modal Identity Verification & Forensic Risk Dashboard
File: app.py
"""

import os
import json
import tempfile
from datetime import datetime
import streamlit as st
import cv2
from PIL import Image

from src.ocr_engine import extract_and_validate
from src.ela_engine import analyze_image_tampering, detect_tampered_regions
from src.face_engine import match_faces
from src.risk_engine import calculate_risk_score

st.set_page_config(
    page_title="VeriID AI - Border Screening Engine",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Dark Styling
st.markdown("""
<style>
    .reportview-container { background: #0E1117; }
    .metric-card {
        background-color: #1E293B;
        border-radius: 8px;
        padding: 15px;
        border: 1px solid #334155;
        text-align: center;
    }
    .verdict-banner-low {
        background-color: #064E3B;
        color: #6EE7B7;
        padding: 14px;
        border-radius: 8px;
        font-weight: bold;
        font-size: 1.15rem;
        border: 1px solid #059669;
        margin-bottom: 15px;
    }
    .verdict-banner-med {
        background-color: #78350F;
        color: #FCD34D;
        padding: 14px;
        border-radius: 8px;
        font-weight: bold;
        font-size: 1.15rem;
        border: 1px solid #D97706;
        margin-bottom: 15px;
    }
    .verdict-banner-high {
        background-color: #7F1D1D;
        color: #FCA5A5;
        padding: 14px;
        border-radius: 8px;
        font-weight: bold;
        font-size: 1.15rem;
        border: 1px solid #DC2626;
        margin-bottom: 15px;
    }
</style>
""", unsafe_allow_html=True)


def save_temp_file(uploaded_file) -> str:
    suffix = os.path.splitext(uploaded_file.name)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getbuffer())
        return tmp.name


st.sidebar.title("🛂 VeriID Screening Control")
st.sidebar.markdown("**Multi-Modal Border Checkpoint AI Engine**")

demo_samples = {
    "Select Sample...": (None, None),
    "1. Genuine / Valid ID (Low Risk)": (
        "data/test_samples/genuine/genuine1.jpeg",
        "data/test_samples/genuine/genuine1.jpeg"
    ),
    "2. Tampered / ELA Text Splicing (Medium Risk)": (
        "data/test_samples/tampered/tempered1.jpeg",
        "data/test_samples/tampered/tempered1.jpeg"
    ),
    "3. Impersonation Fraud (High Risk)": (
        "data/test_samples/impersonation/impersonation1.jpeg",
        "data/test_samples/impersonation/impersonation1.jpeg"
    ),
    "4. Screen Replay Spoof (High Risk)": (
        "data/test_samples/stress_tests/stress_screen_spoof.jpeg",
        "data/test_samples/stress_tests/stress_screen_spoof.jpeg"
    ),
    "5. Underage Chronological Fraud (Medium Risk)": (
        "data/test_samples/stress_tests/stress_underage.jpeg",
        "data/test_samples/stress_tests/stress_underage.jpeg"
    )
}

selected_demo = st.sidebar.selectbox("⚡ Quick Demo Presets", list(demo_samples.keys()))

doc_path, live_path = None, None

if selected_demo != "Select Sample...":
    p_doc, p_live = demo_samples[selected_demo]
    if os.path.exists(p_doc) and os.path.exists(p_live):
        doc_path, live_path = p_doc, p_live

st.sidebar.markdown("---")
uploaded_doc = st.sidebar.file_uploader("Upload ID Document (JPEG/PNG)", type=["jpeg", "jpg", "png"])
uploaded_live = st.sidebar.file_uploader("Upload Live Selfie / Capture", type=["jpeg", "jpg", "png"])

if uploaded_doc is not None:
    doc_path = save_temp_file(uploaded_doc)
if uploaded_live is not None:
    live_path = save_temp_file(uploaded_live)

run_btn = st.sidebar.button("🚀 Run Multi-Vector Verification", use_container_width=True)

st.title("🛡️ VeriID AI — Screening & Identity Verification Dashboard")

if run_btn:
    if not doc_path or not live_path:
        st.warning("Please upload both Document and Live Capture files or select a Quick Demo Preset.")
    else:
        with st.spinner("Executing Multi-Modal Forensic Pipeline..."):
            ocr_res = extract_and_validate(doc_path)
            ela_res = analyze_image_tampering(doc_path)
            ela_detail = detect_tampered_regions(doc_path)
            face_res = match_faces(doc_path, live_path)
            risk_res = calculate_risk_score(ocr_res, ela_res, face_res)

        # Verdict Header Banner
        score_val = risk_res.get("risk_score", 0.0)
        verdict = risk_res.get("verdict", "UNKNOWN")

        if score_val <= 20.0:
            st.markdown(f'<div class="verdict-banner-low">Verdict: {verdict} | Risk Score: {score_val:.1f}%</div>', unsafe_allow_html=True)
        elif score_val <= 50.0:
            st.markdown(f'<div class="verdict-banner-med">Verdict: {verdict} | Risk Score: {score_val:.1f}%</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="verdict-banner-high">Verdict: {verdict} | Risk Score: {score_val:.1f}%</div>', unsafe_allow_html=True)

        # Telemetry Metrics Grid
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            sim_score = face_res.get("similarity_score", 0.0)
            st.metric("Biometric Match", f"{sim_score:.1f}%", "Matched" if face_res.get("is_same_person") else "Mismatch")
        with m2:
            is_live = face_res.get("is_live", True)
            st.metric("Liveness / PAD", "PASSED" if is_live else "SPOOF DETECTED", f"Spoof Risk: {face_res.get('spoof_confidence', 0.0):.1f}%")
        with m3:
            anomaly = ela_res.get("anomaly_score", 0.0)
            is_tamp = ela_res.get("tampering_detected", False)
            st.metric("Forensic ELA Score", f"{anomaly:.2f}", "Tampered" if is_tamp else "Clean Integrity")
        with m4:
            is_valid = ocr_res.get("is_valid_format", False)
            st.metric("OCR Syntax & Schema", "VALID" if is_valid else "INVALID", f"{len(ocr_res.get('raw_tokens', []))} Tokens")

        st.markdown("### 🔍 Visual Evidence & Overlay Inspection")
        c1, c2, c3 = st.columns(3)

        with c1:
            st.markdown("**Document Localization & Splicing Bounding-Boxes**")
            if ela_detail.get("annotated_image") is not None:
                st.image(ela_detail["annotated_image"], use_container_width=True)
            else:
                st.image(doc_path, use_container_width=True)

        with c2:
            st.markdown(f"**Error Level Analysis Heatmap (Variance: {anomaly:.2f})**")
            if ela_res.get("ela_image") is not None:
                st.image(ela_res["ela_image"], use_container_width=True)
            else:
                st.info("No ELA variance generated.")

        with c3:
            st.markdown("**Live Ingestion Biometrics**")
            st.image(live_path, use_container_width=True)

        st.markdown("### 📑 Extracted Records & Compliance Audit Engine")
        r1, r2 = st.columns([1.2, 1])

        with r1:
            st.markdown("**Parsed OCR Document Fields:**")
            st.json({
                "Document Type": ocr_res.get("doc_type_detected"),
                "Full Name": ocr_res.get("extracted_fields", {}).get("name"),
                "Date of Birth": ocr_res.get("extracted_fields", {}).get("dob"),
                "Gender": ocr_res.get("extracted_fields", {}).get("gender"),
                "ID Number": ocr_res.get("extracted_fields", {}).get("id_number") or ocr_res.get("extracted_fields", {}).get("id number"),
                "Issue Date": ocr_res.get("extracted_fields", {}).get("issue_date")
            })

        with r2:
            st.markdown("**Formal Compliance Audit Package:**")
            audit_package = {
                "veriid_audit_id": f"AUD-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}",
                "timestamp_utc": datetime.utcnow().isoformat(),
                "final_verdict": verdict,
                "overall_fraud_risk_score": f"{score_val:.1f}%",
                "pipeline_telemetry": {
                    "biometrics": {
                        "facial_similarity": f"{sim_score:.2f}%",
                        "same_person_authenticated": face_res.get("is_same_person"),
                        "anti_spoof_liveness_verified": face_res.get("is_live"),
                        "spoof_confidence_score": face_res.get("spoof_confidence"),
                        "latency_ms": face_res.get("latency_ms", 0.0)
                    },
                    "forensics_ela": {
                        "anomaly_score": anomaly,
                        "tampering_detected": is_tamp,
                        "flagged_regions_count": len(ela_detail.get("bounding_boxes", []))
                    },
                    "ocr_integrity": {
                        "format_valid": is_valid,
                        "document_type": ocr_res.get("doc_type_detected"),
                        "extracted_data": ocr_res.get("extracted_fields"),
                        "chronological_discrepancy": ocr_res.get("chronological_discrepancy")
                    }
                },
                "flagged_risk_reasons": risk_res.get("flagged_reasons", [])
            }

            audit_json_str = json.dumps(audit_package, indent=2)
            st.download_button(
                label="📥 Download Formal Compliance Audit (JSON)",
                data=audit_json_str,
                file_name=f"VeriID_Audit_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
                use_container_width=True
            )
else:
    st.info("👈 Select a Quick Demo Preset or upload images in the sidebar, then click 'Run Multi-Vector Verification'.")