"""
VeriID AI - Final Project State & Handover Document Generator
File: scripts/generate_handover_pdf.py
"""

import os
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle


def generate_handover_pdf(filename: str = "VeriID_AI_Handover_Document.pdf"):
    doc = SimpleDocTemplate(
        filename,
        pagesize=letter,
        rightMargin=40,
        leftMargin=40,
        topMargin=40,
        bottomMargin=40
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontSize=18,
        leading=22,
        textColor=colors.HexColor('#1E293B'),
        spaceAfter=4
    )
    subtitle_style = ParagraphStyle(
        'DocSubtitle',
        parent=styles['Normal'],
        fontSize=10,
        leading=14,
        textColor=colors.HexColor('#059669'),
        spaceAfter=12
    )
    heading_style = ParagraphStyle(
        'SectionHeading',
        parent=styles['Heading2'],
        fontSize=12,
        leading=16,
        textColor=colors.HexColor('#0F172A'),
        spaceBefore=8,
        spaceAfter=6
    )
    body_style = ParagraphStyle(
        'BodyDark',
        parent=styles['Normal'],
        fontSize=9,
        leading=13,
        textColor=colors.HexColor('#334155')
    )
    bullet_style = ParagraphStyle(
        'BulletDark',
        parent=styles['Normal'],
        fontSize=9,
        leading=13,
        textColor=colors.HexColor('#334155'),
        leftIndent=12,
        firstLineIndent=-8
    )

    story = []

    # Title & Subtitle
    story.append(Paragraph("VeriID AI — Final Project State & Handover Document", title_style))
    story.append(Paragraph("Version: Day 4 Final Integration Completed | Benchmark: 100.00% Pass Rate (20/20 Samples)", subtitle_style))
    story.append(Spacer(1, 6))

    # 1. Project Overview & Multi-Modal Architecture
    story.append(Paragraph("1. Project Overview & Multi-Modal Architecture", heading_style))
    story.append(Paragraph("VeriID AI evaluates identity documents against live facial captures across forensic risk vectors:", body_style))
    story.append(Paragraph("• <b>OCR Engine:</b> Dynamic EasyOCR tokenization, fuzzy keyword classification, regex ID parsing, and chronological validity checks.", bullet_style))
    story.append(Paragraph("• <b>Forensic ELA Engine:</b> Dual-scale JPEG Error Level Analysis (Q90/Q75) with contour thresholding and localized bounding boxes.", bullet_style))
    story.append(Paragraph("• <b>Biometrics & PAD:</b> DeepFace (Facenet512) 1:1 facial verification with passive FFT Moiré spectrum, glare ratio tracking, and execution latency profiling.", bullet_style))
    story.append(Paragraph("• <b>Multi-Factor Risk Engine:</b> Weighted risk score aggregation with three-tier verdict thresholds.", bullet_style))
    story.append(Spacer(1, 6))

    # 2. Repository Structure
    story.append(Paragraph("2. Final Repository Structure", heading_style))
    repo_text = (
        "<b>veriid-ai/</b><br/>"
        "&nbsp;&nbsp;├── <b>app.py</b> — Streamlit Risk Dashboard & Forensic Heatmap Visualizer<br/>"
        "&nbsp;&nbsp;├── <b>src/</b><br/>"
        "&nbsp;&nbsp;│&nbsp;&nbsp;&nbsp;&nbsp;├── <b>ocr_engine.py</b> — EasyOCR + Regex validation + Fuzzy classification + Token confidence<br/>"
        "&nbsp;&nbsp;│&nbsp;&nbsp;&nbsp;&nbsp;├── <b>ela_engine.py</b> — Dual-scale ELA (Q90/Q75) + Bounding box localization<br/>"
        "&nbsp;&nbsp;│&nbsp;&nbsp;&nbsp;&nbsp;├── <b>face_engine.py</b> — DeepFace Facenet512 + FFT PAD + Latency telemetry<br/>"
        "&nbsp;&nbsp;│&nbsp;&nbsp;&nbsp;&nbsp;└── <b>risk_engine.py</b> — Weighted scoring & verdict assignment<br/>"
        "&nbsp;&nbsp;├── <b>scripts/</b><br/>"
        "&nbsp;&nbsp;│&nbsp;&nbsp;&nbsp;&nbsp;├── <b>run_benchmark.py</b> — 20-sample automated evaluation matrix<br/>"
        "&nbsp;&nbsp;│&nbsp;&nbsp;&nbsp;&nbsp;└── <b>generate_handover_pdf.py</b> — Project state report generator<br/>"
        "&nbsp;&nbsp;└── <b>data/test_samples/</b> — 20 ground-truth verified test image pairs & stress tests"
    )
    story.append(Paragraph(repo_text, body_style))
    story.append(Spacer(1, 6))

    # 3. Engine Configurations & Calibrated Thresholds
    story.append(Paragraph("3. Engine Configurations & Calibrated Thresholds", heading_style))
    table_data = [
        ["Engine / Vector", "Model / Method", "Threshold / Metric", "Risk Weight"],
        ["Biometric Match", "Facenet512 (Cosine)", "Distance <= 0.25 (Sim >= 85%)", "60.0%"],
        ["Anti-Spoofing", "FFT Moiré + Glare", "Spoof Score < 50.0", "60.0%"],
        ["Forensic ELA", "Dual-Scale ELA (Q90/Q75)", "Anomaly > 5.0 or Boxes > 0", "40.0%"],
        ["OCR Syntax", "EasyOCR + Date Regex", "Format Valid & Age >= 18", "30.0%"]
    ]
    t = Table(table_data, colWidths=[120, 160, 170, 80])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#F1F5F9')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.HexColor('#0F172A')),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 8),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#CBD5E1')),
        ('ALIGN', (3, 0), (3, -1), 'CENTER'),
    ]))
    story.append(t)
    story.append(Spacer(1, 6))

    # 4. Benchmark Performance
    story.append(Paragraph("4. Final Benchmark Performance (scripts/run_benchmark.py)", heading_style))
    story.append(Paragraph("• <b>Total Evaluated:</b> 20 Samples (Genuine, Tampered, Impersonation, Screen Spoof, Underage, Skewed, High Glare, Blur, Format Anomaly, Print Attack)", bullet_style))
    story.append(Paragraph("• <b>Accuracy Rate:</b> 100.00% (20/20 Passed)", bullet_style))
    story.append(Paragraph("• <b>False Acceptance Rate (FAR):</b> 0.00% (Target: < 5%)", bullet_style))
    story.append(Paragraph("• <b>False Rejection Rate (FRR):</b> 0.00% (Target: < 5%)", bullet_style))
    story.append(Spacer(1, 6))

    # 5. Day 4 Completed Deliverables
    story.append(Paragraph("5. Day 4 Completed Deliverables (All PRs Merged to main)", heading_style))
    story.append(Paragraph("• <b>Member 1 (PR #6):</b> Fuzzy token classification and per-field confidence scoring — <b>Merged</b>", bullet_style))
    story.append(Paragraph("• <b>Member 2:</b> Multi-vector risk aggregation and three-tier decision engine — <b>Complete</b>", bullet_style))
    story.append(Paragraph("• <b>Member 3 (PR #5):</b> Multi-channel PAD & execution latency profiling — <b>Merged</b>", bullet_style))
    story.append(Paragraph("• <b>Member 4 / Lead (PR #7):</b> Dual-scale ELA & localized bounding boxes — <b>Merged</b>", bullet_style))
    story.append(Paragraph("• <b>Member 5:</b> 20-sample dataset expansion & edge-case stress suite — <b>Merged</b>", bullet_style))

    doc.build(story)
    print(f"[✓] Successfully generated updated handover PDF: {filename}")


if __name__ == "__main__":
    generate_handover_pdf()