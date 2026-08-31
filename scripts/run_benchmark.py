import json
import os
import sys
from pathlib import Path
import pandas as pd

# Add project root directory to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.ocr_engine import extract_and_validate
from src.face_engine import match_faces
from src.ela_engine import analyze_image_tampering
from src.risk_engine import calculate_risk_score

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

        # 1. Execute Pipeline Vectors
        ocr_out = extract_and_validate(doc_path) if os.path.exists(doc_path) else {"is_valid_format": False}
        ela_out = analyze_image_tampering(doc_path) if os.path.exists(doc_path) else {"anomaly_score": 0.0, "is_tampered": False}
        face_out = match_faces(doc_path, live_path) if (os.path.exists(doc_path) and os.path.exists(live_path)) else {"is_same_person": False, "similarity_score": 0.0}

        # 2. Risk Calculation via Shared Risk Engine
        risk_summary = calculate_risk_score(ocr_out, ela_out, face_out)
        risk_score = risk_summary["risk_score"]
        verdict = risk_summary["verdict"]

        # Any risk score >= 30.0% (Manual Review or Rejected) flags the non-genuine sample
        predicted_fraud = risk_score >= 30.0
        actual_fraud = not expected_genuine

        # 3. Matrix Classification
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
            "Risk Score": f"{risk_score}%",
            "Verdict": verdict,
            "Outcome": status
        })

    # Summary Metrics Calculation
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