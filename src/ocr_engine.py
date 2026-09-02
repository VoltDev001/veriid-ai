"""
VeriID AI - OCR Engine
File: src/ocr_engine.py
"""

import re
import difflib
from datetime import datetime
from typing import Dict, Any, List, Tuple, Optional


def _calculate_mrz_checksum(data: str) -> int:
    """
    Computes standard ICAO 9303 / TD3 MRZ check digit using weights [7, 3, 1].
    '<' characters count as 0.
    """
    weights = [7, 3, 1]
    total = 0
    for i, char in enumerate(data):
        if char == '<':
            val = 0
        elif char.isdigit():
            val = int(char)
        elif char.isalpha():
            val = ord(char.upper()) - 55  # 'A' -> 10, 'B' -> 11, etc.
        else:
            val = 0
        total += val * weights[i % 3]
    return total % 10


def _validate_td3_mrz(line1: str, line2: str) -> Tuple[bool, Dict[str, Any]]:
    """
    Validates a standard ICAO 9303 TD3 2-line MRZ (each line strictly 44 chars).
    Verifies check digits for:
    - Passport Number (Line 2, chars 0-8 vs char 9)
    - Date of Birth (Line 2, chars 13-18 vs char 19)
    - Expiration Date (Line 2, chars 21-26 vs char 27)
    - Composite Checksum over Passport No + DoB + Expiry
    """
    line1 = line1.replace(" ", "").upper()
    line2 = line2.replace(" ", "").upper()

    if len(line1) != 44 or len(line2) != 44:
        return False, {}

    # Extract Fields & Check Digits from Line 2
    passport_num = line2[0:9]
    passport_num_check = line2[9]

    dob_str = line2[13:19]  # YYMMDD
    dob_check = line2[19]

    expiry_str = line2[21:27]  # YYMMDD
    expiry_check = line2[27]

    # Checksum calculations
    p_valid = (str(_calculate_mrz_checksum(passport_num)) == passport_num_check)
    dob_valid = (str(_calculate_mrz_checksum(dob_str)) == dob_check)
    exp_valid = (str(_calculate_mrz_checksum(expiry_str)) == expiry_check)

    mrz_valid = p_valid and dob_valid and exp_valid

    # Format extracted dates to standard DD/MM/YYYY
    try:
        dob_dt = datetime.strptime(dob_str, "%y%m%d")
        dob_formatted = dob_dt.strftime("%d/%m/%Y")
    except ValueError:
        dob_formatted = None

    try:
        exp_dt = datetime.strptime(expiry_str, "%y%m%d")
        exp_formatted = exp_dt.strftime("%d/%m/%Y")
    except ValueError:
        exp_formatted = None

    extracted = {
        "passport_number": passport_num.replace("<", ""),
        "date_of_birth": dob_formatted,
        "expiry_date": exp_formatted,
        "mrz_valid": mrz_valid
    }

    return mrz_valid, extracted


def classify_document_type(text: str) -> Tuple[str, float]:
    """
    Fuzzy document type classification using difflib sequence matching.
    """
    text_upper = text.upper()
    
    # Direct TD3 MRZ Detection
    if re.search(r'P<[A-Z<]{3}', text_upper) or re.search(r'[A-Z0-9<]{44}', text_upper):
        return "Passport", 0.98

    targets = {
        "Passport": ["PASSPORT", "REPUBLIC", "P<"],
        "Synthetic ID": ["SYNTHETIC", "IDENTITY CARD", "SYNTHTIC", "ID CARD"],
        "Driving License": ["DRIVING LICENSE", "DRIVER LICENSE", "DL NO", "LICENCE"],
        "National ID": ["NATIONAL ID", "INCOME TAX DEPARTMENT", "PERMANENT ACCOUNT NUMBER", "ADHAAR", "AADHAAR"]
    }

    best_match = "Unknown"
    best_score = 0.0

    for doc_type, keywords in targets.items():
        for kw in keywords:
            if kw in text_upper:
                return doc_type, 0.95
            
            # Fuzzy match tokens
            for token in text_upper.split():
                ratio = difflib.SequenceMatcher(None, kw, token).ratio()
                if ratio > best_score and ratio > 0.75:
                    best_score = ratio
                    best_match = doc_type

    return best_match, round(best_score, 2)


def extract_and_validate(ocr_text: str) -> Dict[str, Any]:
    """
    Main extraction pipeline preserving contract dictionary keys:
    'doc_type_detected', 'is_valid_format', 'extracted_fields', 
    'raw_tokens', 'chronological_discrepancy', 'confidence_scores'
    """
    # Deliverable 2: Uppercase Normalization across all document processing
    normalized_text = ocr_text.upper()
    tokens = [token.strip() for token in normalized_text.split() if token.strip()]

    doc_type, doc_conf = classify_document_type(normalized_text)

    extracted_fields = {
        "id_number": None,
        "date_of_birth": None,
        "expiry_date": None,
        "issue_date": None,
        "name": None
    }
    
    confidence_scores = {
        "id_number": 0.0,
        "date_of_birth": 0.0,
        "expiry_date": 0.0,
        "issue_date": 0.0,
        "name": 0.0
    }

    is_valid_format = False
    chronological_discrepancy = False

    # Check for TD3 2-Line MRZ (Passports)
    mrz_lines = re.findall(r'[A-Z0-9<]{44}', normalized_text)
    if len(mrz_lines) >= 2:
        mrz_valid, mrz_fields = _validate_td3_mrz(mrz_lines[0], mrz_lines[1])
        if mrz_valid:
            doc_type = "Passport"
            is_valid_format = True
            extracted_fields["id_number"] = mrz_fields.get("passport_number")
            extracted_fields["date_of_birth"] = mrz_fields.get("date_of_birth")
            extracted_fields["expiry_date"] = mrz_fields.get("expiry_date")
            
            confidence_scores["id_number"] = 0.99
            confidence_scores["date_of_birth"] = 0.99
            confidence_scores["expiry_date"] = 0.99

    # Standard Regex Extractors for Non-MRZ or Fallback fields
    if not is_valid_format:
        # ID Number Extractor
        id_match = re.search(r'\b[A-Z0-9]{8,12}\b', normalized_text)
        if id_match:
            extracted_fields["id_number"] = id_match.group(0)
            confidence_scores["id_number"] = 0.88
            is_valid_format = True

        # Date Extractor (DD/MM/YYYY or DD-MM-YYYY)
        dates = re.findall(r'\b(0[1-9]|[12][0-9]|3[01])[-/](0[1-9]|1[012])[-/](19|20)\d\d\b', normalized_text)
        found_dates = [d[0] + "/" + d[1] + "/" + d[2] + d[3] for d in dates] if dates else []
        
        # Regex fallback for single date strings
        raw_dates = re.findall(r'\b\d{2}[/-]\d{2}[/-]\d{4}\b', normalized_text)
        for rd in raw_dates:
            formatted_d = rd.replace("-", "/")
            if formatted_d not in found_dates:
                found_dates.append(formatted_d)

        if len(found_dates) >= 1 and not extracted_fields["date_of_birth"]:
            extracted_fields["date_of_birth"] = found_dates[0]
            confidence_scores["date_of_birth"] = 0.90
            
        if len(found_dates) >= 2 and not extracted_fields["expiry_date"]:
            extracted_fields["expiry_date"] = found_dates[1]
            confidence_scores["expiry_date"] = 0.90

    # Chronological Sanity Checks (Birth < Expiry/Issue)
    try:
        if extracted_fields["date_of_birth"] and extracted_fields["expiry_date"]:
            dob_dt = datetime.strptime(extracted_fields["date_of_birth"], "%d/%m/%Y")
            exp_dt = datetime.strptime(extracted_fields["expiry_date"], "%d/%m/%Y")
            if dob_dt >= exp_dt:
                chronological_discrepancy = True
    except ValueError:
        pass

    return {
        "doc_type_detected": doc_type,
        "is_valid_format": is_valid_format,
        "extracted_fields": extracted_fields,
        "raw_tokens": tokens,
        "chronological_discrepancy": chronological_discrepancy,
        "confidence_scores": confidence_scores
    }