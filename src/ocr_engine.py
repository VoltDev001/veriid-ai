import re
from typing import Dict, Any, List
from paddleocr import PaddleOCR

# Initialize PaddleOCR engine globally to avoid re-instantiating on every function call
# use_angle_cls=True handles rotated/upside-down document scans
ocr_model = PaddleOCR(use_angle_cls=True, lang='en')


def _validate_pan(text: str) -> bool:
    """Checks for standard Indian PAN format: 5 uppercase letters, 4 digits, 1 uppercase letter."""
    pattern = r"[A-Z]{5}[0-9]{4}[A-Z]"
    return bool(re.search(pattern, text))


def _validate_passport_mrz(lines: List[str]) -> bool:
    """
    Validates Type-3 Passport MRZ structure:
    2 lines, each exactly 44 characters containing alphanumeric characters and '<'.
    """
    mrz_lines = []
    for line in lines:
        cleaned = line.replace(" ", "").upper()
        if len(cleaned) == 44 and "<" in cleaned:
            mrz_lines.append(cleaned)
    
    return len(mrz_lines) >= 2


def _validate_aadhaar_format(text: str) -> bool:
    """
    Checks for 12-digit structural format (spaced as 4-4-4 or contiguous 12 digits).
    Does not validate check-digit algorithms (Verhoeff) to avoid false negatives on raw OCR outputs.
    """
    # Pattern matches 12 digits separated by spaces/hyphens or standard 12 contiguous digits
    pattern = r"\b\d{4}[\s-]?\d{4}[\s-]?\d{4}\b"
    return bool(re.search(pattern, text))


def extract_and_validate(image_path: str) -> Dict[str, Any]:
    """
    Extracts text from an image using PaddleOCR and validates the document's format integrity.
    
    Interface Contract Output:
    {
        "raw_text": list[str],
        "doc_type_detected": str,
        "is_valid_format": bool,
        "error_flags": list[str]
    }
    """
    raw_text: List[str] = []
    error_flags: List[str] = []
    doc_type_detected = "Unknown/Unrecognized"
    is_valid_format = False

    try:
        # Perform OCR inference
        result = ocr_model.ocr(image_path, cls=True)
        
        if result and result[0]:
            # Flatten extracted text strings
            raw_text = [line[1][0].strip() for line in result[0] if line[1][0].strip()]
    except Exception as e:
        return {
            "raw_text": [],
            "doc_type_detected": "Error",
            "is_valid_format": False,
            "error_flags": [f"OCR Processing Failure: {str(e)}"]
        }

    joined_text = " ".join(raw_text).upper()

    # Rule Integrity Engine & Classification
    if "INCOME TAX DEPARTMENT" in joined_text or "PERMANENT ACCOUNT NUMBER" in joined_text or _validate_pan(joined_text):
        doc_type_detected = "PAN Card"
        if _validate_pan(joined_text):
            is_valid_format = True
        else:
            is_valid_format = False
            error_flags.append("Invalid or corrupted PAN format structure")

    elif _validate_passport_mrz(raw_text) or "PASSPORT" in joined_text:
        doc_type_detected = "Passport"
        if _validate_passport_mrz(raw_text):
            is_valid_format = True
        else:
            is_valid_format = False
            error_flags.append("Invalid Passport MRZ format (Expected 2 lines of 44 characters)")

    elif "UNIQUE IDENTIFICATION AUTHORITY OF INDIA" in joined_text or "GOVERNMENT OF INDIA" in joined_text or _validate_aadhaar_format(joined_text):
        doc_type_detected = "Aadhaar Card"
        if _validate_aadhaar_format(joined_text):
            is_valid_format = True
        else:
            is_valid_format = False
            error_flags.append("Invalid 12-digit format pattern")

    else:
        doc_type_detected = "Generic Document"
        is_valid_format = len(raw_text) > 0
        if not is_valid_format:
            error_flags.append("No readable text detected in document")

    return {
        "raw_text": raw_text,
        "doc_type_detected": doc_type_detected,
        "is_valid_format": is_valid_format,
        "error_flags": error_flags
    }


if __name__ == "__main__":
    # Internal module unit test
    import sys
    test_img = sys.argv[1] if len(sys.argv) > 1 else "data/test_samples/sample_pan.jpg"
    print(f"--- Running Local Test on: {test_img} ---")
    out = extract_and_validate(test_img)
    print("Detected Document Type:", out["doc_type_detected"])
    print("Format Validated:", out["is_valid_format"])
    print("Error Flags:", out["error_flags"])
    print("Extracted Lines Count:", len(out["raw_text"]))
