import os
from typing import Dict, Any

def match_faces(id_card_path: str, live_photo_path: str) -> Dict[str, Any]:
    # Mock return for Member 3 contract
    return {
        "is_same_person": True,
        "similarity_score": 88.5,
        "face_detected_in_id": True,
        "face_detected_in_live": True
    }