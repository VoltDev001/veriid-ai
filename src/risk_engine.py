from typing import Dict, Any, List

def calculate_risk_score(
    ocr_result: Dict[str, Any], 
    tamper_result: Dict[str, Any], 
    face_result: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Aggregates multi-vector forensic indicators into a calibrated risk score (0-100).
    Enforces hard rejection on direct biometric failure and localized tampering.
    """
    total_risk = 0.0
    flagged_reasons: List[str] = []

    # 1. Biometric Match & Face Detection (Hard Factor: Max 60%)
    id_face_present = face_result.get("face_detected_in_id", True)
    live_face_present = face_result.get("face_detected_in_live", True)
    face_matched = face_result.get("is_same_person", False)
    similarity = face_result.get("similarity_score", 0.0)

    if not id_face_present or not live_face_present:
        total_risk += 60.0
        flagged_reasons.append("Biometric capture failed: No face detected in ID or Live photo.")
    elif not face_matched:
        # Immediate High Risk for face mismatch
        total_risk += 60.0
        flagged_reasons.append(f"Biometric mismatch detected: Face does not match ID (Confidence: {similarity:.1f}%).")
    elif similarity < 80.0:
        total_risk += 20.0
        flagged_reasons.append(f"Marginal facial match confidence ({similarity:.1f}%).")

    # 2. Tampering & Forensics / ELA (Factor: Max 40%)
    tamper_score = tamper_result.get("anomaly_score", 0.0)
    is_tampered = tamper_result.get("is_tampered", False)
    
    if is_tampered or tamper_score > 55.0:
        total_risk += 40.0
        flagged_reasons.append(f"High compression/tampering anomaly detected (ELA Score: {tamper_score:.1f}).")
    elif tamper_score > 35.0:
        total_risk += 20.0
        flagged_reasons.append(f"Moderate compression variance observed (ELA Score: {tamper_score:.1f}).")

    # 3. OCR Format & Field Integrity (Factor: Max 30%)
    if not ocr_result.get("is_valid_format", True):
        total_risk += 30.0
        flags = ocr_result.get("error_flags", ["Document format validation failed."])
        flagged_reasons.extend(flags)

    # Normalize total risk
    total_risk = min(100.0, round(total_risk, 1))

    # Strict Verdict Assignment
    if total_risk >= 60.0:
        verdict = "REJECTED (HIGH FRAUD RISK)"
        color = "red"
    elif total_risk >= 30.0:
        verdict = "MANUAL REVIEW REQUIRED (MEDIUM RISK)"
        color = "orange"
    else:
        verdict = "VERIFIED (LOW RISK)"
        color = "green"

    return {
        "risk_score": total_risk,
        "verdict": verdict,
        "verdict_color": color,
        "flagged_reasons": flagged_reasons if flagged_reasons else ["All verification checks passed."]
    }