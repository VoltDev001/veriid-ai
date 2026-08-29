"""
VeriID AI - Face Matching Engine
Member 3: Face Verification Module
File: src/face_engine.py
"""

import os
from typing import Dict, Any, Optional
import numpy as np
from deepface import DeepFace

MODEL_NAME = "Facenet512"
DETECTOR_BACKEND = "opencv"
DISTANCE_METRIC = "cosine"
MIN_CONFIDENCE = 0.60


def _cosine_to_percentage(distance: float) -> float:
    if distance < 0.0:
        return 100.0
    similarity = (1.0 - distance) * 100.0
    return float(np.clip(round(similarity, 2), 0.0, 100.0))


def _detect_faces(image_path: str) -> tuple[bool, int, Optional[str]]:
    if not os.path.exists(image_path):
        return False, 0, f"File not found: {image_path}"

    try:
        faces = DeepFace.extract_faces(
            img_path=image_path,
            detector_backend=DETECTOR_BACKEND,
            enforce_detection=False,
            align=True
        )
        valid_faces = [f for f in faces if f.get("confidence", 0.0) >= MIN_CONFIDENCE]
        count = len(valid_faces)
        return (count > 0), count, None
    except Exception as exc:
        return False, 0, f"Face detection error in {os.path.basename(image_path)}: {str(exc)}"


def match_faces(id_card_path: str, live_photo_path: str) -> Dict[str, Any]:
    response: Dict[str, Any] = {
        "is_same_person": False,
        "similarity_score": 0.0,
        "face_detected_in_id": False,
        "face_detected_in_live": False,
        "error": None
    }

    if not os.path.isfile(id_card_path):
        response["error"] = f"ID Card image not found: {id_card_path}"
        return response

    if not os.path.isfile(live_photo_path):
        response["error"] = f"Live photo not found: {live_photo_path}"
        return response

    try:
        id_detected, id_count, id_err = _detect_faces(id_card_path)
        response["face_detected_in_id"] = id_detected
        if id_err:
            response["error"] = id_err
            return response
        if not id_detected:
            response["error"] = "No face detected in the ID card image."
            return response

        live_detected, live_count, live_err = _detect_faces(live_photo_path)
        response["face_detected_in_live"] = live_detected
        if live_err:
            response["error"] = live_err
            return response
        if not live_detected:
            response["error"] = "No face detected in the live photo."
            return response

        if live_count > 1:
            response["error"] = f"Multiple faces ({live_count}) detected in live photo. Exactly 1 person required."
            return response

        result = DeepFace.verify(
            img1_path=id_card_path,
            img2_path=live_photo_path,
            model_name=MODEL_NAME,
            detector_backend=DETECTOR_BACKEND,
            distance_metric=DISTANCE_METRIC,
            enforce_detection=False
        )

        distance = float(result.get("distance", 1.0))
        is_same = bool(result.get("verified", False))

        response["is_same_person"] = is_same
        response["similarity_score"] = _cosine_to_percentage(distance)
        response["error"] = None
        return response

    except Exception as ex:
        response["error"] = f"Verification failed: {str(ex)}"
        return response