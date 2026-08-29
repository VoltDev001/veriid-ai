import os
import cv2
import numpy as np
from PIL import Image, ImageChops, ImageEnhance
from typing import Tuple

def generate_ela(image_path: str, quality: int = 90, scale: int = 15) -> Tuple[Image.Image, float, bool]:
    if not os.path.exists(image_path):
        dummy_img = Image.new("RGB", (300, 200), color=(30, 30, 30))
        return dummy_img, 0.0, False

    temp_resaved = "temp_ela_resaved.jpg"
    original = Image.open(image_path).convert("RGB")
    original.save(temp_resaved, "JPEG", quality=quality)
    resaved = Image.open(temp_resaved)

    ela_image = ImageChops.difference(original, resaved)
    extrema = ela_image.getextrema()
    max_diff = max([ex[1] for ex in extrema])
    scale_factor = scale if max_diff == 0 else max(1, int(255.0 / max_diff))
    
    enhancer = ImageEnhance.Brightness(ela_image)
    ela_image = enhancer.enhance(scale_factor)

    ela_np = np.array(ela_image)
    mean_diff = float(np.mean(ela_np))
    anomaly_score = min(100.0, round(mean_diff * 4.0, 2))
    is_tampered = anomaly_score > 45.0

    if os.path.exists(temp_resaved):
        os.remove(temp_resaved)

    return ela_image, anomaly_score, is_tampered