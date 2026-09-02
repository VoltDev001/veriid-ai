"""
VeriID AI - Multi-Factor Risk Aggregation Engine
File: src/risk_engine.py
"""

from typing import Dict, Any, List

# Calibrated Risk Weights
W_BIOMETRIC = 60.0
W_SPOOF = 60.0
W_ELA = 40.0
W_OCR = 30.0

# Verdict Thresholds
VERIFIED_MAX_SCORE = 20.0
MANUAL_REVIEW_MAX_SCORE = 50.0


def calculate_risk_score(
    ocr_result: Dict[str, Any],
    ela_result: Dict[str, Any],
    face_result: Dict[str, Any]
) -> Dict[str, Any]:
    risk_score = 0.0
    flagged_reasons: List[str] = []

    # 1. Biometric Match Evaluation
    is_same_person = face_result.get("is_same_person", False)
    sim_score = face_result.get("similarity_score", 0.0)

    if not is_same_person or sim_score < 85.0:
        risk_score += W_BIOMETRIC
        flagged_reasons.append(f"Biometric Mismatch: Similarity score {sim_score:.1f}% below threshold")

    # 2. Anti-Spoofing / Presentation Attack Detection
    is_live = face_result.get("is_live", True)
    spoof_flags = face_result.get("liveness_flags", [])

    if not is_live:
        risk_score += W_SPOOF
        flag_desc = ", ".join(spoof_flags) if spoof_flags else "Passive liveness threshold exceeded"
        flagged_reasons.append(f"Spoof Attack Detected: {flag_desc}")

    # 3. Forensic ELA Tampering Evaluation
    is_tampered = ela_result.get("tampering_detected", False)
    ela_score = ela_result.get("anomaly_score", 0.0)

    if is_tampered:
        risk_score += W_ELA
        flagged_reasons.append(f"Forensic Anomaly: ELA score flagged ({ela_score:.1f})")

    # 4. OCR Integrity & Chronological Validation
    format_valid = ocr_result.get("is_valid_format", True)
    chronological_discrepancy = ocr_result.get("chronological_discrepancy", False)
    ocr_error = ocr_result.get("error")

    if not format_valid or chronological_discrepancy:
        risk_score += W_OCR
        if chronological_discrepancy:
            flagged_reasons.append(f"Chronological Anomaly: {ocr_error or 'Underage issue date detected'}")
        elif ocr_error:
            flagged_reasons.append(f"OCR Format Discrepancy: {ocr_error}")
        else:
            flagged_reasons.append("OCR Format Discrepancy: Incomplete or malformed fields")

    # Clamp aggregate risk score to 100.0%
    final_score = min(100.0, round(risk_score, 1))

    # Determine 3-Tier Verdict
    if final_score <= VERIFIED_MAX_SCORE:
        verdict = "VERIFIED (LOW RISK)"
    elif final_score <= MANUAL_REVIEW_MAX_SCORE:
        verdict = "MANUAL REVIEW REQUIRED (MEDIUM RISK)"
    else:
        verdict = "REJECTED (HIGH FRAUD RISK)"

    return {
        "risk_score": final_score,
        "verdict": verdict,
        "flagged_reasons": flagged_reasons
    }