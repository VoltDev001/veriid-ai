"""
VeriID AI - Face Matching & Biometric Verification Engine
Member 3: Biometrics & DeepFace Engineer
File: src/face_engine.py
"""

import os
from typing import Dict, Any, Optional
import numpy as np
from PIL import Image
from deepface import DeepFace

# =====================================================================
# Configuration Constants & Calibrated Thresholds
# =====================================================================
MODEL_NAME = "Facenet512"
DETECTOR_BACKEND = "opencv"
DISTANCE_METRIC = "cosine"

# Decision Threshold: DeepFace Facenet512 standard cosine threshold is ~0.30
COSINE_MATCH_THRESHOLD = 0.30

# Calibrated Match Threshold: >= 80.0% similarity required for verified match
SIMILARITY_MATCH_THRESHOLD = 80.0

# Image Quality Thresholds (Blur & Lighting)
MIN_LAPLACIAN_VAR = 25.0      # Below this = too blurry
MIN_BRIGHTNESS = 20.0         # Below this = too dark
MAX_BRIGHTNESS = 248.0        # Above this = overexposed / washed out
MIN_DETECTION_CONFIDENCE = 0.60


def _check_image_quality(image_path: str) -> tuple[bool, Optional[str]]:
    """
    Lightweight quality and blur check using pure PIL and NumPy.
    Eliminates cv2 dependency issues.
    """
    try:
        if not os.path.exists(image_path):
            return False, f"File does not exist: {image_path}"

        with Image.open(image_path) as pil_img:
            gray = np.array(pil_img.convert('L'), dtype=np.float64)

        # 1. Blur Detection via Discrete Laplacian 2D Kernel
        if gray.shape[0] > 10 and gray.shape[1] > 10:
            laplacian = (
                gray[:-2, 1:-1] + gray[2:, 1:-1] +
                gray[1:-1, :-2] + gray[1:-1, 2:] -
                4.0 * gray[1:-1, 1:-1]
            )
            laplacian_var = float(np.var(laplacian))
            if laplacian_var < MIN_LAPLACIAN_VAR:
                return False, f"Image quality too blurry ({laplacian_var:.1f} < {MIN_LAPLACIAN_VAR}) in {os.path.basename(image_path)}"

        # 2. Brightness Check
        mean_brightness = float(np.mean(gray))
        if mean_brightness < MIN_BRIGHTNESS:
            return False, f"Image too dark (brightness: {mean_brightness:.1f}) in {os.path.basename(image_path)}"
        if mean_brightness > MAX_BRIGHTNESS:
            return False, f"Image overexposed (brightness: {mean_brightness:.1f}) in {os.path.basename(image_path)}"

        return True, None
    except Exception as e:
        return False, f"Quality check failed: {str(e)}"


def _calibrated_cosine_to_similarity(distance: float) -> float:
    """
    Biometric Calibration Curve:
    - Identical / genuine faces (distance <= 0.15) -> Score > 90%
    - Borderline match (distance == 0.30) -> Score = 80.0%
    - Lookalike synthetic cross-person (distance >= 0.38) -> Drops below 70-75%
    - Distant faces (distance >= 0.70) -> Drops to near 0%
    """
    if distance <= 0.0:
        return 100.0

    t = COSINE_MATCH_THRESHOLD  # 0.30

    if distance <= t:
        score = 80.0 + (20.0 * (1.0 - (distance / t)))
    else:
        decay_range = 0.45
        drop_ratio = (distance - t) / decay_range
        score = 80.0 * (1.0 - drop_ratio)

    return float(np.clip(round(score, 2), 0.0, 100.0))


def _detect_faces_in_image(image_path: str) -> tuple[bool, int, Optional[str]]:
    """
    Detects faces in image using DeepFace.extract_faces.
    """
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
    """
    Matches face on ID card against live capture photo.

    Exact Contract Structure:
    {
        "face_detected_in_id": bool,
        "face_detected_in_live": bool,
        "is_same_person": bool,
        "similarity_score": float,  # 0.0 to 100.0
        "error": None or str
    }
    """
    response: Dict[str, Any] = {
        "face_detected_in_id": False,
        "face_detected_in_live": False,
        "is_same_person": False,
        "similarity_score": 0.0,
        "error": None
    }

    # 1. File existence validation
    if not os.path.isfile(id_card_path):
        response["error"] = f"ID Card image not found: {id_card_path}"
        return response

    if not os.path.isfile(live_photo_path):
        response["error"] = f"Live photo not found: {live_photo_path}"
        return response

    # 2. Quality & Blur/Brightness Validation
    id_quality_ok, id_q_err = _check_image_quality(id_card_path)
    if not id_quality_ok:
        response["error"] = id_q_err
        return response

    live_quality_ok, live_q_err = _check_image_quality(live_photo_path)
    if not live_quality_ok:
        response["error"] = live_q_err
        return response

    try:
        # 3. Face Detection in ID
        id_detected, id_face_count, id_err = _detect_faces_in_image(id_card_path)
        response["face_detected_in_id"] = id_detected
        if id_err:
            response["error"] = id_err
            return response
        if not id_detected:
            response["error"] = "No face detected in the ID card image."
            return response

        # 4. Face Detection in Live
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

        # 5. Verification with DeepFace
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
        is_same = bool(similarity >= SIMILARITY_MATCH_THRESHOLD and distance <= COSINE_MATCH_THRESHOLD)

        response["is_same_person"] = is_same
        response["similarity_score"] = similarity
        response["error"] = None
        return response

    except Exception as ex:
        response["error"] = f"Face verification failed: {str(ex)}"
        return response