import streamlit as st
import os
import sys
import json
from datetime import datetime

sys.path.append(os.path.abspath("src"))

from ocr_engine import extract_and_validate
from tampering_engine import generate_ela
from face_engine import match_faces
from risk_engine import calculate_risk_score

st.set_page_config(page_title="VeriID AI - Document Screening", layout="wide")

st.title("🛡️ VeriID AI: Document & Identity Forensic Screening")
st.markdown("Automated Multi-Vector Authentication & Fraud Detection")

col1, col2 = st.columns(2)
with col1:
    st.subheader("1. Document Ingestion")
    doc_file = st.file_uploader("Upload ID Document", type=["jpg", "jpeg", "png"])
    
with col2:
    st.subheader("2. Live Capture Ingestion")
    live_file = st.file_uploader("Upload Live Photo", type=["jpg", "jpeg", "png"])

if st.button("Execute Forensic Screening", use_container_width=True):
    if doc_file and live_file:
        os.makedirs(os.path.join("data", "output"), exist_ok=True)
        doc_path = os.path.join("data", "output", doc_file.name)
        live_path = os.path.join("data", "output", live_file.name)
        
        with open(doc_path, "wb") as f:
            f.write(doc_file.getbuffer())
        with open(live_path, "wb") as f:
            f.write(live_file.getbuffer())

        # Vector Execution
        ocr_res = extract_and_validate(doc_path)
        ela_img, anomaly_score, is_tampered = generate_ela(doc_path)
        tamper_res = {"anomaly_score": anomaly_score, "is_tampered": is_tampered}
        face_res = match_faces(doc_path, live_path)

        # Risk Orchestration
        risk_summary = calculate_risk_score(ocr_res, tamper_res, face_res)

        st.divider()
        st.subheader(f"Screening Verdict: {risk_summary['verdict']}")
        st.metric("Total Risk Score", f"{risk_summary['risk_score']}%")

        m_col1, m_col2, m_col3 = st.columns(3)
        m_col1.metric("Doc Integrity", "PASSED" if ocr_res["is_valid_format"] else "FAILED")
        m_col2.metric("ELA Forensics", f"{anomaly_score:.1f}/100", delta="Tampered" if is_tampered else "Clean", delta_color="inverse")
        m_col3.metric("Biometric Match", f"{face_res['similarity_score']}%", "Match" if face_res["is_same_person"] else "Mismatch")

        with st.expander("🔍 Detailed Forensic Observations", expanded=True):
            for reason in risk_summary["flagged_reasons"]:
                st.markdown(f"- {reason}")

        v_col1, v_col2 = st.columns(2)
        with v_col1:
            st.image(doc_path, caption="Original Document", use_container_width=True)
        with v_col2:
            st.image(ela_img, caption="Error Level Analysis (ELA) Heatmap", use_container_width=True)

        # Audit Export
        report_data = {
            "timestamp": datetime.now().isoformat(),
            "verdict": risk_summary["verdict"],
            "risk_score": risk_summary["risk_score"],
            "ocr_analysis": ocr_res,
            "tampering_analysis": tamper_res,
            "biometric_analysis": face_res,
            "flags": risk_summary["flagged_reasons"]
        }

        st.download_button(
            label="📥 Download Audit Report (JSON)",
            data=json.dumps(report_data, indent=2),
            file_name=f"veriid_audit_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json",
            use_container_width=True
        )
    else:
        st.warning("Please upload both an ID document and a live photo to run screening.")