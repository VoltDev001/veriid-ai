"""
VeriID AI - Face Matching & Biometric Verification Engine
Member 3: Biometrics & Anti-Spoofing Engineer
File: src/face_engine.py
"""

import os
import uuid
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
    Calibrated passive presentation attack detection (Moiré, Chromatic Variance, Specular Glare).
    """
    flags = []
    spoof_score = 0.0

    try:
        img_bgr = cv2.imread(image_path)
        if img_bgr is None:
            return True, 0.0, []

        h, w = img_bgr.shape[:2]
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

        # 1. High-frequency Moiré / Texture via FFT
        f = np.fft.fft2(gray)
        fshift = np.fft.fftshift(f)
        magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1)
        
        center_y, center_x = h // 2, w // 2
        high_freq_ring = magnitude_spectrum.copy()
        cv2.circle(high_freq_ring, (center_x, center_y), int(min(h, w) * 0.20), 0, -1)
        moire_val = float(np.mean(high_freq_ring))

        if moire_val > 190.0:
            spoof_score += 45.0
            flags.append(f"Screen raster/moiré pattern detected ({moire_val:.1f})")

        # 2. Specular Hotspots & Screen Glare
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        sat_channel = hsv[:, :, 1]
        val_channel = hsv[:, :, 2]
        
        total_pixels = max(1, h * w)
        glare_pixels = np.count_nonzero((val_channel > 252) & (sat_channel < 15))
        glare_ratio = (glare_pixels / total_pixels) * 100.0

        if glare_ratio > 5.0:
            spoof_score += 35.0
            flags.append(f"Screen specular reflection anomaly detected ({glare_ratio:.2f}%)")

        spoof_score = min(100.0, round(spoof_score, 1))
        is_live = spoof_score < 50.0
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


def _deskew_image(image_path: str) -> np.ndarray:
    """Detects document rotation angle via contour orientation and straightens it."""
    img = cv2.imread(image_path)
    if img is None:
        return None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    largest_cnt = max(contours, key=cv2.contourArea) if contours else None
    if largest_cnt is not None and cv2.contourArea(largest_cnt) > 5000:
        rect = cv2.minAreaRect(largest_cnt)
        angle = rect[-1]
        if angle < -45:
            angle = -(90 + angle)
        elif angle > 45:
            angle = 90 - angle
        else:
            angle = -angle

        if abs(angle) > 5.0:
            h, w = img.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            img = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    return img


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

    if not os.path.isfile(id_card_path) or not os.path.isfile(live_photo_path):
        response["error"] = "Input image files not found."
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
        # 1. Anti-Spoofing / Liveness Check
        is_live, spoof_score, liveness_flags = _check_liveness(live_photo_path)
        response["is_live"] = is_live
        response["spoof_confidence"] = spoof_score
        response["liveness_flags"] = liveness_flags

        # 2. Live Face Verification
        live_detected, live_face_count, live_err = _detect_faces_in_image(live_photo_path)
        response["face_detected_in_live"] = live_detected
        if not live_detected or live_err:
            response["error"] = live_err or "No face detected in live photo."
            return response

        # 3. Match Evaluation (Original with Fallback to Deskewed Array)
        img_candidates = [id_card_path]
        deskewed = _deskew_image(id_card_path)
        if deskewed is not None:
            img_candidates.append(deskewed)

        match_succeeded = False
        last_err = None

        for candidate in img_candidates:
            try:
                result = DeepFace.verify(
                    img1_path=candidate,
                    img2_path=live_photo_path,
                    model_name=MODEL_NAME,
                    detector_backend=DETECTOR_BACKEND,
                    distance_metric=DISTANCE_METRIC,
                    enforce_detection=False
                )
                distance = float(result.get("distance", 1.0))
                similarity = _calibrated_cosine_to_similarity(distance)
                is_same = bool(similarity >= SIMILARITY_MATCH_THRESHOLD and distance <= COSINE_MATCH_THRESHOLD and is_live)

                response["face_detected_in_id"] = True
                response["is_same_person"] = is_same
                response["similarity_score"] = similarity
                response["error"] = None
                match_succeeded = True
                break
            except Exception as e:
                last_err = str(e)
                continue

        if not match_succeeded:
            response["error"] = f"Face verification failed: {last_err}"

        return response

    except Exception as ex:
        response["error"] = f"Face verification failed: {str(ex)}"
        return response