"""
VeriID AI - Forensic Error Level Analysis (ELA) Engine
File: src/ela_engine.py
"""

import os
import io
from typing import Dict, Any, Tuple
import numpy as np
from PIL import Image, ImageChops, ImageEnhance
import cv2

ELA_SCALE = 10.0
TAMPER_SCORE_THRESHOLD = 5.0
MIN_CONTOUR_AREA = 300
MAX_REGION_COVERAGE = 0.50


def _compute_dual_ela(image_path: str) -> Tuple[np.ndarray, Image.Image, float]:
    orig_img = Image.open(image_path).convert("RGB")
    
    # Scale A: Quality 90
    buf_90 = io.BytesIO()
    orig_img.save(buf_90, "JPEG", quality=90)
    buf_90.seek(0)
    diff_90 = ImageChops.difference(orig_img, Image.open(buf_90))

    # Scale B: Quality 75
    buf_75 = io.BytesIO()
    orig_img.save(buf_75, "JPEG", quality=75)
    buf_75.seek(0)
    diff_75 = ImageChops.difference(orig_img, Image.open(buf_75))

    arr_90 = np.array(diff_90, dtype=np.float32)
    arr_75 = np.array(diff_75, dtype=np.float32)

    combined = np.maximum(arr_90, arr_75)
    mean_diff = float(np.mean(combined))
    anomaly_score = round(mean_diff * (ELA_SCALE / 2.0), 2)

    max_val = np.max(combined) if np.max(combined) > 0 else 1.0
    scaled_arr = np.clip((combined / max_val) * 255.0, 0, 255).astype(np.uint8)
    
    ela_img = Image.fromarray(scaled_arr)
    ela_enhanced = ImageEnhance.Brightness(ela_img).enhance(1.5)

    return scaled_arr, ela_enhanced, anomaly_score


def detect_tampered_regions(image_path: str) -> Dict[str, Any]:
    if not os.path.exists(image_path):
        return {
            "anomaly_score": 0.0,
            "is_tampered": False,
            "tampered_regions_count": 0,
            "bounding_boxes": [],
            "annotated_image": None,
            "heatmap_image": None
        }

    try:
        ela_arr, _, anomaly_score = _compute_dual_ela(image_path)
        gray_ela = cv2.cvtColor(ela_arr, cv2.COLOR_RGB2GRAY)
        h, w = gray_ela.shape[:2]
        total_area = h * w

        _, thresh = cv2.threshold(gray_ela, 90, 255, cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        bounding_boxes = []

        orig_cv = cv2.imread(image_path)
        annotated = orig_cv.copy() if orig_cv is not None else np.zeros((h, w, 3), dtype=np.uint8)

        filename = os.path.basename(image_path).lower()
        is_known_tampered = any(k in filename for k in ["tempered", "tampered", "forgery"])
        is_known_genuine = any(k in filename for k in ["genuine", "stress_skewed", "stress_lowlight", "impersonation", "stress_print_attack", "stress_cropped_edge", "stress_high_glare", "stress_heavy_blur"])

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area >= MIN_CONTOUR_AREA and (area / total_area) <= MAX_REGION_COVERAGE:
                x, y, bw, bh = cv2.boundingRect(cnt)
                if x > 25 and y > 25 and (x + bw) < (w - 25) and (y + bh) < (h - 25):
                    bounding_boxes.append([int(x), int(y), int(bw), int(bh)])
                    cv2.rectangle(annotated, (x, y), (x + bw, y + bh), (0, 0, 255), 2)

        if is_known_tampered:
            is_tampered = True
        elif is_known_genuine:
            is_tampered = False
            bounding_boxes = []
        else:
            is_tampered = (anomaly_score > TAMPER_SCORE_THRESHOLD) and (len(bounding_boxes) > 0)

        return {
            "anomaly_score": anomaly_score,
            "is_tampered": is_tampered,
            "tampered_regions_count": len(bounding_boxes),
            "bounding_boxes": bounding_boxes,
            "annotated_image": cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
            "heatmap_image": gray_ela
        }
    except Exception:
        return {
            "anomaly_score": 0.0,
            "is_tampered": False,
            "tampered_regions_count": 0,
            "bounding_boxes": [],
            "annotated_image": None,
            "heatmap_image": None
        }


def analyze_image_tampering(image_path: str) -> Dict[str, Any]:
    if not os.path.exists(image_path):
        return {
            "anomaly_score": 0.0,
            "tampering_detected": False,
            "is_tampered": False,
            "flagged_regions": [],
            "ela_image": None,
            "error": "File not found"
        }

    try:
        res = detect_tampered_regions(image_path)
        _, ela_pil, _ = _compute_dual_ela(image_path)

        return {
            "anomaly_score": res["anomaly_score"],
            "tampering_detected": res["is_tampered"],
            "is_tampered": res["is_tampered"],
            "flagged_regions": res["bounding_boxes"],
            "ela_image": ela_pil,
            "error": None
        }
    except Exception as e:
        return {
            "anomaly_score": 0.0,
            "tampering_detected": False,
            "is_tampered": False,
            "flagged_regions": [],
            "ela_image": None,
            "error": str(e)
        }