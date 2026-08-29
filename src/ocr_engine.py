import re
from typing import Dict, Any, List

def _validate_pan(text: str) -> bool:
    return bool(re.search(r"[A-Z]{5}[0-9]{4}[A-Z]", text))

def extract_and_validate(image_path: str) -> Dict[str, Any]:
    # Mock return for Member 1 contract
    sample_text = ["INCOME TAX DEPARTMENT", "PERMANENT ACCOUNT NUMBER", "ABCDE1234F"]
    joined_text = " ".join(sample_text)
    doc_type = "PAN" if _validate_pan(joined_text) else "Generic"
    
    return {
        "raw_text": sample_text,
        "doc_type_detected": doc_type,
        "is_valid_format": _validate_pan(joined_text),
        "error_flags": []
    }