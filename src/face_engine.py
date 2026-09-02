"""
VeriID AI - Facial Verification & Multi-Channel Anti-Spoofing Engine
File: src/face_engine.py
"""

import time
import cv2
import numpy as np
from deepface import DeepFace

# Verification and Anti-Spoof calibrated thresholds
COSINE_MATCH_THRESHOLD = 0.25
MOIRE_HIGH_FREQ_THRESHOLD = 45.0
GLARE_SATURATION_MAX_RATIO = 0.08


def detect_presentation_attack(image_path: str) -> dict:
    """
    Multi-channel passive presentation attack detection:
    - FFT 2D frequency spectrum analysis to flag screen pixel grids / Moiré patterns.
    - Specular glare / screen reflection saturation thresholding.
    """
    img = cv2.imread(image_path)
    if img is None:
        return {
            "is_live": False,
            "spoof_confidence": 100.0,
            "liveness_flags": ["Failed to decode image frame"]
        }

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    flags = []

    # 1. FFT High-Frequency Analysis for Moiré artifacts
    dft = cv2.dft(np.float32(gray), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)
    magnitude = 20 * np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]) + 1e-8)

    # Mask central low frequencies
    cy, cx = h // 2, w // 2
    r = min(h, w) // 8
    magnitude[cy - r: cy + r, cx - r: cx + r] = 0
    hf_energy = float(np.mean(magnitude))

    if hf_energy > MOIRE_HIGH_FREQ_THRESHOLD:
        flags.append(f"Screen Replay / Moiré artifact detected (HF Energy: {hf_energy:.1f})")

    # 2. Specular Glare & Monitor Reflection Saturation
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    v_channel = hsv[:, :, 2]
    glare_pixels = np.count_nonzero(v_channel > 250)
    glare_ratio = float(glare_pixels / (h * w))

    if glare_ratio > GLARE_SATURATION_MAX_RATIO:
        flags.append(f"Excessive monitor glare / display reflection (Ratio: {glare_ratio:.3f})")

    # Determine spoof risk confidence
    if len(flags) > 0:
        spoof_conf = min(100.0, float(len(flags) * 50.0))
        is_live = False
    else:
        spoof_conf = 0.0
        is_live = True

    return {
        "is_live": is_live,
        "spoof_confidence": spoof_conf,
        "liveness_flags": flags
    }


def match_faces(id_card_path: str, live_photo_path: str) -> dict:
    """
    1:1 Facial Verification using Facenet512 with execution latency profiling.
    """
    t0 = time.perf_counter()
    pad_result = detect_presentation_attack(live_photo_path)

    detector_used = "opencv"
    try:
        result = DeepFace.verify(
            img1_path=id_card_path,
            img2_path=live_photo_path,
            model_name="Facenet512",
            distance_metric="cosine",
            enforce_detection=False,
            detector_backend="opencv"
        )
        distance = float(result.get("distance", 1.0))
        similarity = max(0.0, min(100.0, (1.0 - distance) * 100.0))
        is_same = bool(distance <= COSINE_MATCH_THRESHOLD)
    except Exception:
        detector_used = "fallback-ssd"
        try:
            result = DeepFace.verify(
                img1_path=id_card_path,
                img2_path=live_photo_path,
                model_name="Facenet512",
                distance_metric="cosine",
                enforce_detection=False,
                detector_backend="ssd"
            )
            distance = float(result.get("distance", 1.0))
            similarity = max(0.0, min(100.0, (1.0 - distance) * 100.0))
            is_same = bool(distance <= COSINE_MATCH_THRESHOLD)
        except Exception:
            distance = 1.0
            similarity = 0.0
            is_same = False

    latency_ms = round((time.perf_counter() - t0) * 1000.0, 2)

    return {
        "is_same_person": is_same,
        "similarity_score": round(similarity, 2),
        "cosine_distance": round(distance, 4),
        "is_live": pad_result["is_live"],
        "spoof_confidence": pad_result["spoof_confidence"],
        "liveness_flags": pad_result["liveness_flags"],
        "latency_ms": latency_ms,
        "detector_backend": detector_used
    }