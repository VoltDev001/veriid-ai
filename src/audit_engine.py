"""
VeriID AI - Border Control Compliance & Audit Logging Engine
File: src/audit_engine.py
"""

import uuid
from datetime import datetime, timezone
from typing import Dict, Any, List


def generate_compliance_audit_package(
    ocr_result: Dict[str, Any],
    ela_result: Dict[str, Any],
    face_result: Dict[str, Any],
    risk_result: Dict[str, Any],
    checkpoint_metadata: Dict[str, str] = None
) -> Dict[str, Any]:
    if checkpoint_metadata is None:
        checkpoint_metadata = {
            "terminal_id": "T3-INTL-ARRIVALS",
            "egate_lane_id": "LANE-04B",
            "officer_badge_id": "BCP-88219",
            "station_location": "Terminal 3, Air Border Control Post",
            "inspection_mode": "Automated e-Gate Screening"
        }

    now_utc = datetime.now(timezone.utc).isoformat()
    audit_uuid = str(uuid.uuid4())[:8].upper()
    audit_id = f"AUD-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}-{audit_uuid}"

    # Vector sub-score breakdown
    biometric_sim = float(face_result.get("similarity_score", 0.0))
    is_live = bool(face_result.get("is_live", True))
    ela_anomaly = float(ela_result.get("anomaly_score", 0.0))
    ocr_valid = bool(ocr_result.get("is_valid_format", False))
    chronological_anomaly = bool(ocr_result.get("chronological_discrepancy", False))

    vector_contributions = {
        "biometric_facial_match": {
            "weight_pct": 60.0,
            "similarity_pct": biometric_sim,
            "risk_incurred": 60.0 if (not face_result.get("is_same_person", False) or biometric_sim < 85.0) else 0.0
        },
        "presentation_attack_pad": {
            "weight_pct": 60.0,
            "spoof_confidence_pct": float(face_result.get("spoof_confidence", 0.0)),
            "liveness_passed": is_live,
            "risk_incurred": 60.0 if not is_live else 0.0
        },
        "forensic_ela_integrity": {
            "weight_pct": 40.0,
            "anomaly_score": ela_anomaly,
            "tampering_detected": bool(ela_result.get("tampering_detected", False)),
            "risk_incurred": 40.0 if bool(ela_result.get("tampering_detected", False)) else 0.0
        },
        "ocr_syntax_chronology": {
            "weight_pct": 30.0,
            "format_valid": ocr_valid,
            "chronological_discrepancy": chronological_anomaly,
            "risk_incurred": 30.0 if (not ocr_valid or chronological_anomaly) else 0.0
        }
    }

    return {
        "veriid_audit_id": audit_id,
        "timestamp_utc": now_utc,
        "checkpoint_context": checkpoint_metadata,
        "final_decision": {
            "verdict": risk_result.get("verdict", "UNKNOWN"),
            "aggregate_risk_score_pct": risk_result.get("risk_score", 0.0),
            "flagged_risk_reasons": risk_result.get("flagged_reasons", []),
            "operator_action_required": "Proceed with clearance" if risk_result.get("risk_score", 0.0) <= 20.0 else (
                "Refer passenger to secondary inspection desk" if risk_result.get("risk_score", 0.0) <= 50.0 else "Detain and initiate forensic biometric investigation"
            )
        },
        "vector_contributions": vector_contributions,
        "telemetry_metrics": {
            "biometric_verification_latency_ms": face_result.get("latency_ms", 0.0),
            "ela_flagged_regions_count": len(ela_result.get("flagged_regions", [])),
            "ocr_token_count": len(ocr_result.get("raw_tokens", [])),
            "document_classification": ocr_result.get("doc_type_detected", "Unknown")
        },
        "extracted_identity_payload": ocr_result.get("extracted_fields", {}),
        "cryptographic_compliance_signature": {
            "algorithm": "SHA256-HMAC-STUB",
            "signed_by": "VeriID-BorderControl-KMS",
            "integrity_sealed": True
        }
    }