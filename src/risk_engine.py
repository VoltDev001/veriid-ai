"""
VeriID AI - Risk Orchestration Engine
File: src/risk_engine.py
"""

def calculate_risk_score(ocr_res: dict, tamper_res: dict, face_res: dict) -> dict:
    risk_score = 0.0
    flagged_reasons = []

    # 1. OCR Integrity (Weight: 30%)
    if not ocr_res.get("is_valid_format", True):
        risk_score += 30.0
        for err in ocr_res.get("error_flags", []):
            flagged_reasons.append(f"OCR Discrepancy: {err}")

    # 2. Forensic ELA (Weight: 40%)
    if tamper_res.get("tampering_detected", False) or tamper_res.get("is_tampered", False):
        risk_score += 40.0
        flagged_reasons.append(f"Forensic Anomaly: ELA score flagged ({tamper_res.get('anomaly_score', 0.0):.1f})")

    # 3. Biometric Match (Weight: 60%)
    if not face_res.get("is_same_person", False):
        risk_score += 60.0
        flagged_reasons.append(f"Biometric Mismatch: Similarity score {face_res.get('similarity_score', 0.0):.1f}% below threshold")

    # 4. Anti-Spoofing / Presentation Attack (Weight: 60%)
    if not face_res.get("is_live", True):
        risk_score += 60.0
        for flag in face_res.get("liveness_flags", []):
            flagged_reasons.append(f"Spoof Attack Detected: {flag}")

    risk_score = min(100.0, round(risk_score, 1))

    if risk_score <= 20.0:
        verdict = "VERIFIED (LOW RISK)"
    elif risk_score <= 50.0:
        verdict = "MANUAL REVIEW REQUIRED (MEDIUM RISK)"
    else:
        verdict = "REJECTED (HIGH FRAUD RISK)"

    if not flagged_reasons:
        flagged_reasons.append("All verification checks passed.")

    return {
        "risk_score": risk_score,
        "verdict": verdict,
        "flagged_reasons": flagged_reasons
    }