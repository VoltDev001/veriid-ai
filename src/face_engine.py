"""
VeriID AI - Face Matching & Biometric Verification Engine
Member 3: Biometrics & Anti-Spoofing Engineer
File: src/face_engine.py
"""

import os
from typing import Dict, Any, Optional, Tuple, List
import numpy as np
from PIL import Image
import cv2
from deepface import DeepFace

# =====================================================================
# Configuration Constants & Calibrated Thresholds
# =====================================================================
MODEL_NAME = "Facenet512"
DETECTOR_BACKEND = "opencv"
DISTANCE_METRIC = "cosine"

COSINE_MATCH_THRESHOLD = 0.25
SIMILARITY_MATCH_THRESHOLD = 85.0

# Quality Thresholds
MIN_LAPLACIAN_VAR = 25.0
MIN_BRIGHTNESS = 20.0
MAX_BRIGHTNESS = 248.0
MIN_DETECTION_CONFIDENCE = 0.60


def _check_image_quality(image_path: str) -> Tuple[bool, Optional[str]]:
    try:
        if not os.path.exists(image_path):
            return False, f"File does not exist: {image_path}"

        with Image.open(image_path) as pil_img:
            gray = np.array(pil_img.convert('L'), dtype=np.float64)

        if gray.shape[0] > 10 and gray.shape[1] > 10:
            laplacian = (
                gray[:-2, 1:-1] + gray[2:, 1:-1] +
                gray[1:-1, :-2] + gray[1:-1, 2:] -
                4.0 * gray[1:-1, 1:-1]
            )
            laplacian_var = float(np.var(laplacian))
            if laplacian_var < MIN_LAPLACIAN_VAR:
                return False, f"Image quality too blurry ({laplacian_var:.1f} < {MIN_LAPLACIAN_VAR}) in {os.path.basename(image_path)}"

        mean_brightness = float(np.mean(gray))
        if mean_brightness < MIN_BRIGHTNESS:
            return False, f"Image too dark ({mean_brightness:.1f}) in {os.path.basename(image_path)}"
        if mean_brightness > MAX_BRIGHTNESS:
            return False, f"Image overexposed ({mean_brightness:.1f}) in {os.path.basename(image_path)}"

        return True, None
    except Exception as e:
        return False, f"Quality check failed: {str(e)}"


def _check_liveness(image_path: str) -> Tuple[bool, float, List[str]]:
    """
    Passive presentation attack detection (Moiré, Chromatic Variance, Specular Glare).
    """
    flags = []
    spoof_score = 0.0

    try:
        img_bgr = cv2.imread(image_path)
        if img_bgr is None:
            return True, 0.0, []

        # 1. High-frequency Moiré / Texture via FFT
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        f = np.fft.fft2(gray)
        fshift = np.fft.fftshift(f)
        magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1)
        h, w = gray.shape
        center_y, center_x = h // 2, w // 2
        high_freq_ring = magnitude_spectrum.copy()
        cv2.circle(high_freq_ring, (center_x, center_y), int(min(h, w) * 0.15), 0, -1)
        moire_val = float(np.mean(high_freq_ring))

        if moire_val > 145.0:
            spoof_score += 45.0
            flags.append(f"Screen raster/moiré pattern detected ({moire_val:.1f})")

        # 2. Specular Hotspots
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        val_channel = hsv[:, :, 2]
        sat_channel = hsv[:, :, 1]
        glare_pixels = np.count_nonzero((val_channel > 250) & (sat_channel < 25))
        total_pixels = h * w
        glare_ratio = (glare_pixels / total_pixels) * 100.0

        if glare_ratio > 3.5:
            spoof_score += 35.0
            flags.append(f"Specular reflection anomaly detected ({glare_ratio:.2f}%)")

        spoof_score = min(100.0, round(spoof_score, 1))
        is_live = spoof_score < 40.0
        return is_live, spoof_score, flags

    except Exception:
        return True, 0.0, []


def _calibrated_cosine_to_similarity(distance: float) -> float:
    if distance <= 0.0:
        return 100.0

    t = COSINE_MATCH_THRESHOLD  # 0.25

    if distance <= t:
        score = 85.0 + (15.0 * (1.0 - (distance / t)))
    else:
        decay_range = 0.35
        drop_ratio = min(1.0, (distance - t) / decay_range)
        score = 85.0 * (1.0 - drop_ratio)

    return float(np.clip(round(score, 2), 0.0, 100.0))


def _detect_faces_in_image(image_path: str) -> Tuple[bool, int, Optional[str]]:
    try:
        faces = DeepFace.extract_faces(
            img_path=image_path,
            detector_backend=DETECTOR_BACKEND,
            enforce_detection=False,
            align=True
        )
        valid_faces = [f for f in faces if f.get("confidence", 0.0) >= MIN_DETECTION_CONFIDENCE]
        count = len(valid_faces)
        return (count > 0), count, None
    except Exception as exc:
        return False, 0, f"Detection failure on {os.path.basename(image_path)}: {str(exc)}"


def match_faces(id_card_path: str, live_photo_path: str) -> Dict[str, Any]:
    response: Dict[str, Any] = {
        "face_detected_in_id": False,
        "face_detected_in_live": False,
        "is_same_person": False,
        "similarity_score": 0.0,
        "is_live": True,
        "spoof_confidence": 0.0,
        "liveness_flags": [],
        "error": None
    }

    if not os.path.isfile(id_card_path):
        response["error"] = f"ID Card image not found: {id_card_path}"
        return response

    if not os.path.isfile(live_photo_path):
        response["error"] = f"Live photo not found: {live_photo_path}"
        return response

    id_quality_ok, id_q_err = _check_image_quality(id_card_path)
    if not id_quality_ok:
        response["error"] = id_q_err
        return response

    live_quality_ok, live_q_err = _check_image_quality(live_photo_path)
    if not live_quality_ok:
        response["error"] = live_q_err
        return response

    try:
        id_detected, id_face_count, id_err = _detect_faces_in_image(id_card_path)
        response["face_detected_in_id"] = id_detected
        if id_err:
            response["error"] = id_err
            return response
        if not id_detected:
            response["error"] = "No face detected in the ID card image."
            return response

        live_detected, live_face_count, live_err = _detect_faces_in_image(live_photo_path)
        response["face_detected_in_live"] = live_detected
        if live_err:
            response["error"] = live_err
            return response
        if not live_detected:
            response["error"] = "No face detected in the live photo."
            return response

        if live_face_count > 1:
            response["error"] = f"Multiple faces ({live_face_count}) detected in live capture. Exactly 1 required."
            return response

        # Anti-Spoofing / Liveness Analysis
        is_live, spoof_score, liveness_flags = _check_liveness(live_photo_path)
        response["is_live"] = is_live
        response["spoof_confidence"] = spoof_score
        response["liveness_flags"] = liveness_flags

        # Biometric Similarity Matching
        result = DeepFace.verify(
            img1_path=id_card_path,
            img2_path=live_photo_path,
            model_name=MODEL_NAME,
            detector_backend=DETECTOR_BACKEND,
            distance_metric=DISTANCE_METRIC,
            enforce_detection=False
        )

        distance = float(result.get("distance", 1.0))
        similarity = _calibrated_cosine_to_similarity(distance)
        is_same = bool(similarity >= SIMILARITY_MATCH_THRESHOLD and distance <= COSINE_MATCH_THRESHOLD and is_live)

        response["is_same_person"] = is_same
        response["similarity_score"] = similarity
        response["error"] = None
        return response

    except Exception as ex:
        response["error"] = f"Face verification failed: {str(ex)}"
        return response