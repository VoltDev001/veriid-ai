"""
VeriID AI - Demo Quick-Pack Preparation Script
File: scripts/prepare_demo_pack.py
"""

import os
import shutil

SOURCE_DIR = "data/test_samples"
TARGET_DIR = "data/demo_quickpack"

SAMPLE_PAIRS = [
    ("genuine/genuine1.jpeg", "01_Genuine_Clean.jpeg"),
    ("tampered/tempered1.jpeg", "02_Tampered_ELA_Spliced.jpeg"),
    ("impersonation/impersonation1.jpeg", "03_Impersonation_Fraud.jpeg"),
    ("stress_tests/stress_screen_spoof.jpeg", "04_Screen_Replay_Spoof.jpeg"),
    ("stress_tests/stress_underage.jpeg", "05_Underage_Discrepancy.jpeg")
]

def prepare_pack():
    os.makedirs(TARGET_DIR, exist_ok=True)
    for src_rel, target_name in SAMPLE_PAIRS:
        src_path = os.path.join(SOURCE_DIR, src_rel)
        target_path = os.path.join(TARGET_DIR, target_name)
        if os.path.exists(src_path):
            shutil.copyfile(src_path, target_path)
            print(f"[OK] Staged: {target_name}")
        else:
            print(f"[SKIP] Source missing: {src_path}")

if __name__ == "__main__":
    prepare_pack()
