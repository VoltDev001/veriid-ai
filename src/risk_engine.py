from typing import Dict, Any, List

def calculate_risk_score(
    ocr_result: Dict[str, Any], 
    tamper_result: Dict[str, Any], 
    face_result: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Aggregates multi-vector forensic indicators into a weighted risk score (0-100).
    Weights:
      - OCR Format & Integrity: 35%
      - Image Tampering / ELA: 35%
      - Biometric Face Match: 30%
    """
    total_risk = 0.0
    flagged_reasons: List[str] = []

    # 1. OCR Integrity (Weight: 35%)
    if not ocr_result.get("is_valid_format", True):
        total_risk += 35.0
        flags = ocr_result.get("error_flags", ["Document format validation failed"])
        flagged_reasons.extend(flags)

    # 2. Tampering & Forensics (Weight: 35%)
    tamper_score = tamper_result.get("anomaly_score", 0.0)
    is_tampered = tamper_result.get("is_tampered", False)
    if is_tampered or tamper_score > 45.0:
        tamper_risk_contrib = min(35.0, (tamper_score / 100.0) * 35.0 + 10.0)
        total_risk += tamper_risk_contrib
        flagged_reasons.append(f"High compression anomaly detected (ELA Score: {tamper_score:.1f})")

    # 3. Biometric Match (Weight: 30%)
    face_matched = face_result.get("is_same_person", False)
    similarity = face_result.get("similarity_score", 0.0)
    if not face_matched:
        total_risk += 30.0
        flagged_reasons.append(f"Biometric mismatch: live face does not match ID photo (Score: {similarity:.1f}%)")
    elif similarity < 70.0:
        total_risk += 15.0
        flagged_reasons.append(f"Marginal facial match confidence ({similarity:.1f}%)")

    total_risk = min(100.0, round(total_risk, 1))

    if total_risk < 30.0:
        verdict = "VERIFIED (LOW RISK)"
        color = "green"
    elif total_risk < 65.0:
        verdict = "MANUAL REVIEW REQUIRED (MEDIUM RISK)"
        color = "orange"
    else:
        verdict = "REJECTED (HIGH FRAUD RISK)"
        color = "red"

    return {
        "risk_score": total_risk,
        "verdict": verdict,
        "verdict_color": color,
        "flagged_reasons": flagged_reasons if flagged_reasons else ["All verification checks passed."]
    }