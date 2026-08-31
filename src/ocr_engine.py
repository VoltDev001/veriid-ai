import os
import re
from datetime import datetime
import easyocr

# Initialize EasyOCR reader (cached globally to prevent reload delay on every call)
reader = None

def get_ocr_reader():
    global reader
    if reader is None:
        reader = easyocr.Reader(['en'], gpu=False)
    return reader

def _parse_date(date_str: str):
    """Helper to parse date string in DD/MM/YYYY format."""
    try:
        return datetime.strptime(date_str, "%d/%m/%Y")
    except (ValueError, TypeError):
        return None

def extract_and_validate(image_path: str) -> dict:
    """
    Extracts text tokens using EasyOCR, parses key fields using strict regex patterns,
    and runs cross-field integrity checks (e.g. issue date vs DOB discrepancy).
    """
    if not os.path.exists(image_path):
        return {
            "doc_type_detected": "Unknown",
            "is_valid_format": False,
            "extracted_fields": {
                "name": None,
                "dob": None,
                "gender": None,
                "id_number": None,
                "issue_date": None,
                "category": None
            },
            "error_flags": [f"File not found: {image_path}"],
            "raw_text": []
        }

    # 1. Run EasyOCR Extraction
    ocr_reader = get_ocr_reader()
    results = ocr_reader.readtext(image_path, detail=0)
    raw_text = [str(token).strip() for token in results if str(token).strip()]

    combined_text = " ".join(raw_text)

    # 2. Extract Fields using Regex Rules
    extracted_fields = {
        "name": None,
        "dob": None,
        "gender": None,
        "id_number": None,
        "issue_date": None,
        "category": None
    }
    error_flags = []

    # Regex definitions
    date_pattern = r'\b(0[1-9]|[12][0-9]|3[01])/(0[1-9]|1[012])/(19|20)\d{2}\b'
    id_pattern = r'\b[A-Z0-9]{3,4}-?\d{3,4}\b'
    
    # Locate DOB & Issue Date
    dates_found = re.findall(date_pattern, combined_text)
    full_dates = [f"{day}/{month}/{year}" for day, month, year in dates_found]
    
    # Specific targeted parsing across tokens
    for token in raw_text:
        token_upper = token.upper()
        
        # Gender check
        if token_upper in ["MALE", "FEMALE", "OTHER"]:
            extracted_fields["gender"] = token_upper

        # ID Number check
        if not extracted_fields["id_number"]:
            id_match = re.search(r'\b[A-Z]{3}-\d{3}\b|\b\d{4}-\d{4}-\d{4}\b|\b[A-Z]{5}\d{4}[A-Z]\b', token_upper)
            if id_match:
                extracted_fields["id_number"] = id_match.group(0)

    # Assign dates if found in sequence or tagged labels
    for i, token in enumerate(raw_text):
        token_lower = token.lower()
        if "dob" in token_lower or "birth" in token_lower:
            match = re.search(r'\d{2}/\d{2}/\d{4}', token)
            if match:
                extracted_fields["dob"] = match.group(0)
            elif i + 1 < len(raw_text):
                match_next = re.search(r'\d{2}/\d{2}/\d{4}', raw_text[i+1])
                if match_next:
                    extracted_fields["dob"] = match_next.group(0)

        if "issue" in token_lower or "date" in token_lower and not extracted_fields["issue_date"]:
            match = re.search(r'\d{2}/\d{2}/\d{4}', token)
            if match:
                extracted_fields["issue_date"] = match.group(0)
            elif i + 1 < len(raw_text):
                match_next = re.search(r'\d{2}/\d{2}/\d{4}', raw_text[i+1])
                if match_next:
                    extracted_fields["issue_date"] = match_next.group(0)

    # Fallback date assignment if explicit labels weren't isolated
    if not extracted_fields["dob"] and len(full_dates) > 0:
        extracted_fields["dob"] = full_dates[0]
    if not extracted_fields["issue_date"] and len(full_dates) > 1:
        extracted_fields["issue_date"] = full_dates[1]

    # Detect Document Type
    if "SYNTHETIC" in combined_text.upper() or "ID CARD" in combined_text.upper():
        doc_type = "Synthetic ID Card"
    elif "AADHAAR" in combined_text.upper():
        doc_type = "Aadhaar"
    elif "INCOME TAX" in combined_text.upper() or "PERMANENT ACCOUNT" in combined_text.upper():
        doc_type = "PAN"
    else:
        doc_type = "Unknown"

    # 3. Field Syntactic Validation
    if not extracted_fields["id_number"]:
        error_flags.append("Missing or invalid ID Number format")

    if extracted_fields["gender"] not in ["MALE", "FEMALE", "OTHER"]:
        error_flags.append("Invalid or missing Gender")

    dob_dt = _parse_date(extracted_fields["dob"]) if extracted_fields["dob"] else None
    if not dob_dt:
        error_flags.append("Invalid Date of Birth format")

    issue_dt = _parse_date(extracted_fields["issue_date"]) if extracted_fields["issue_date"] else None
    if not issue_dt:
        error_flags.append("Invalid Issue Date format")

    # 4. Cross-Field Chronological Integrity Check
    if dob_dt and issue_dt:
        age_at_issue = issue_dt.year - dob_dt.year - ((issue_dt.month, issue_dt.day) < (dob_dt.month, dob_dt.day))
        if age_at_issue < 18:
            error_flags.append("Underage cardholder or invalid issue date discrepancy")

    is_valid_format = (len(error_flags) == 0)

    return {
        "doc_type_detected": doc_type,
        "is_valid_format": is_valid_format,
        "extracted_fields": extracted_fields,
        "error_flags": error_flags,
        "raw_text": raw_text
    }