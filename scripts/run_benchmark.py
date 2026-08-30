import json
import os
import sys
from pathlib import Path
import pandas as pd

# Add root directory to sys.path to allow imports from src
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.ocr_engine import extract_and_validate
from src.face_engine import match_faces
from src.ela_engine import analyze_image_tampering

def calculate_verdict(ocr_res: dict, ela_res: dict, face_res: dict) -> dict:
    risk_score = 0.0
    flags = []

    # 1. Biometric verification check (Hard factor: Weight 50)
    if not face_res.get("face_detected_in_id") or not face_res.get("face_detected_in_live"):
        risk_score += 40.0
        flags.append("Biometric capture missing face in document or live image.")
    elif not face_res.get("is_same_person", False):
        similarity = face_res.get("similarity_score", 0.0)
        risk_score += 60.0  # Raised to trigger high risk directly
        flags.append(f"Biometric mismatch detected (Score: {similarity:.1f}%).")

    # 2. ELA Forensics Anomaly check (Medium factor: Weight 30)
    ela_score = ela_res.get("anomaly_score", 0.0)
    if ela_score > 55.0:
        risk_score += 40.0
        flags.append(f"High compression/tampering anomaly detected (ELA: {ela_score:.1f}).")
    elif ela_score > 35.0:
        risk_score += 20.0
        flags.append(f"Moderate compression variance observed (ELA: {ela_score:.1f}).")

    # 3. Document / OCR Rule Integrity (Factor: Weight 30)
    if not ocr_res.get("is_valid", True):
        risk_score += 30.0
        flags.extend(ocr_res.get("issues", ["OCR formatting or extraction validation failed."]))

    risk_score = min(100.0, risk_score)

    # Any flagged anomaly (>= 40%) is marked as non-genuine for benchmark classification
    is_flagged_fraud = risk_score >= 40.0

    if risk_score >= 60.0:
        verdict = "REJECTED (HIGH RISK)"
    elif risk_score >= 30.0:
        verdict = "MANUAL REVIEW REQUIRED (MEDIUM RISK)"
    else:
        verdict = "VERIFIED (LOW RISK)"

    return {
        "risk_score": round(risk_score, 2),
        "verdict": verdict,
        "flags": flags,
        "is_fraud": is_flagged_fraud
    }

def run_evaluation(ground_truth_path="data/test_samples/ground_truth.json"):
    if not os.path.exists(ground_truth_path):
        print(f"[!] Ground truth file not found at '{ground_truth_path}'.")
        return

    with open(ground_truth_path, "r") as f:
        test_cases = json.load(f)

    results = []
    tp, fp, tn, fn = 0, 0, 0, 0

    print(f"\n--- Running Automated Benchmark on {len(test_cases)} Samples ---")

    for case in test_cases:
        cid = case.get("id")
        doc_path = case.get("document_path")
        live_path = case.get("live_photo_path")
        expected_genuine = case.get("is_genuine", True)

        # Run pipeline
        ocr_out = extract_and_validate(doc_path) if os.path.exists(doc_path) else {"is_valid": False}
        ela_out = analyze_image_tampering(doc_path) if os.path.exists(doc_path) else {"anomaly_score": 0.0}
        face_out = match_faces(doc_path, live_path) if (os.path.exists(doc_path) and os.path.exists(live_path)) else {"is_same_person": False, "similarity_score": 0.0}

        verdict_data = calculate_verdict(ocr_out, ela_out, face_out)
        predicted_fraud = verdict_data["is_fraud"]
        actual_fraud = not expected_genuine

        # Matrix calculation
        if actual_fraud and predicted_fraud:
            tp += 1
            status = "CORRECT (Detected Fraud)"
        elif not actual_fraud and not predicted_fraud:
            tn += 1
            status = "CORRECT (Verified Genuine)"
        elif not actual_fraud and predicted_fraud:
            fp += 1
            status = "FALSE ALARM (False Rejection)"
        else:
            fn += 1
            status = "MISSED FRAUD (False Acceptance)"

        results.append({
            "Sample ID": cid,
            "Document": os.path.basename(doc_path),
            "Expected": "Genuine" if expected_genuine else "Tampered/Spoof",
            "Risk Score": f"{verdict_data['risk_score']}%",
            "Verdict": verdict_data["verdict"],
            "Outcome": status
        })

    # Summary Metrics
    total = len(test_cases)
    accuracy = ((tp + tn) / total) * 100 if total > 0 else 0
    far = (fn / (tp + fn)) * 100 if (tp + fn) > 0 else 0
    frr = (fp / (tn + fp)) * 100 if (tn + fp) > 0 else 0

    df = pd.DataFrame(results)
    print("\n" + df.to_string(index=False))
    print("\n==========================================")
    print(f"Total Evaluated : {total}")
    print(f"Accuracy Rate   : {accuracy:.2f}%")
    print(f"False Acceptance Rate (FAR): {far:.2f}% (Target: < 5%)")
    print(f"False Rejection Rate  (FRR): {frr:.2f}% (Target: < 5%)")
    print("==========================================\n")

if __name__ == "__main__":
    run_evaluation()