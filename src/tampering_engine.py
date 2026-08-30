import os
import cv2
import numpy as np
from PIL import Image, ImageChops, ImageEnhance

def analyze_ela(image_path: str, quality: int = 90, scale: float = 15.0) -> dict:
    """
    Enhanced ELA pipeline designed to detect localized text forgery and spliced text boxes.
    """
    temp_filename = "temp_ela_resaved.jpg"
    
    try:
        if not os.path.exists(image_path):
            return {
                "tampering_detected": False,
                "anomaly_score": 0.0,
                "ela_image": None,
                "flagged_regions": [],
                "error": f"File not found: {image_path}"
            }

        # 1. Load original image
        original = Image.open(image_path).convert('RGB')
        
        # 2. Resave image at target JPEG quality factor (90-95)
        original.save(temp_filename, 'JPEG', quality=quality)
        resaved = Image.open(temp_filename).convert('RGB')
        
        # 3. Compute absolute difference map and enhance contrast
        ela_im = ImageChops.difference(original, resaved)
        extrema = ela_im.getextrema()
        max_diff = max([ex[1] for ex in extrema])
        if max_diff == 0:
            max_diff = 1
        
        extrema_scale = 255.0 / max_diff
        ela_im = ImageEnhance.Brightness(ela_im).enhance(extrema_scale)
        
        # 4. Convert ELA result to OpenCV grayscale format for localized contour detection
        ela_np = np.array(ela_im)
        gray = cv2.cvtColor(ela_np, cv2.COLOR_RGB2GRAY)
        
        # 5. Apply Gaussian blur and Otsu/Adaptive Thresholding to highlight anomalous clusters
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # 6. Apply Morphological Opening/Closing to consolidate localized text patch anomalies
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        morphed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        morphed = cv2.morphologyEx(morphed, cv2.MORPH_OPEN, kernel)
        
        # 7. Contour Detection for localized bounding box placement
        contours, _ = cv2.findContours(morphed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        flagged_regions = []
        image_area = original.width * original.height
        min_region_area = image_area * 0.0005  # Filter minor image noise
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > min_region_area:
                x, y, w, h = cv2.boundingRect(cnt)
                flagged_regions.append((int(x), int(y), int(w), int(h)))

        # 8. Compute normalized anomaly score (0.0 to 100.0)
        mean_anomaly = np.mean(gray)
        peak_anomaly = np.percentile(gray, 95)
        area_density = (sum([w * h for _, _, w, h in flagged_regions]) / image_area) * 100 if flagged_regions else 0
        
        # Calibrated formula targeting subtle text manipulation patch mismatches
        raw_score = (mean_anomaly * 0.4) + (peak_anomaly * 0.35) + (area_density * 2.5) + (len(flagged_regions) * 3.0)
        anomaly_score = float(np.clip(raw_score, 0.0, 100.0))
        
        # Standard forensic threshold for document text/image tampering
        tampering_detected = bool(anomaly_score >= 55.0 or len(flagged_regions) >= 2)
        
        return {
            "tampering_detected": tampering_detected,
            "anomaly_score": round(anomaly_score, 2),
            "ela_image": ela_im,
            "flagged_regions": flagged_regions,
            "error": None
        }

    except Exception as e:
        return {
            "tampering_detected": False,
            "anomaly_score": 0.0,
            "ela_image": None,
            "flagged_regions": [],
            "error": str(e)
        }
    finally:
        if os.path.exists(temp_filename):
            try:
                os.remove(temp_filename)
            except OSError:
                pass
