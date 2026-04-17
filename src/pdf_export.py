from __future__ import annotations

from pathlib import Path
from typing import Any

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from .report_payload import build_customer_report_payload



def export_customer_report_pdf(results: list[Any], output_path: str | Path) -> Path:
    payload = build_customer_report_payload(results)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    doc = SimpleDocTemplate(str(output), pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm, topMargin=18 * mm, bottomMargin=16 * mm)
    styles = getSampleStyleSheet()
    title = ParagraphStyle("Title", parent=styles["Title"], fontName="Helvetica-Bold", fontSize=22, leading=26, textColor=colors.HexColor("#0f172a"))
    h2 = ParagraphStyle("H2", parent=styles["Heading2"], fontName="Helvetica-Bold", fontSize=14, leading=18, textColor=colors.HexColor("#0f172a"), spaceBefore=10, spaceAfter=6)
    body = ParagraphStyle("Body", parent=styles["BodyText"], fontName="Helvetica", fontSize=9.5, leading=13)
    small = ParagraphStyle("Small", parent=styles["BodyText"], fontName="Helvetica", fontSize=8.5, leading=11)

    story = [
        Paragraph(payload["scenario_name"], title),
        Spacer(1, 4 * mm),
        Paragraph(payload["executive_summary"]["headline"], body),
        Paragraph(payload["executive_summary"]["subheadline"], body),
        Spacer(1, 4 * mm),
    ]

    metric_rows = [[m["label"], m["value"]] for m in payload["executive_summary"]["top_metrics"]]
    story.extend(_table_block(metric_rows, col_widths=[55 * mm, 95 * mm]))

    story.append(Paragraph("Workload summary", h2))
    ws = payload["workload_summary"]
    story.extend(_table_block([
        ["Workload ID", ws.get("workload_id")],
        ["Class", ws.get("workload_class")],
        ["Family", ws.get("workload_family")],
        ["Arrival rate", f"{float(ws.get('configured_arrival_rate_per_sec') or 0.0):.3f} req/s"],
        ["Planning profile", ws.get("planning_profile")],
    ], col_widths=[48 * mm, 102 * mm]))

    story.append(Paragraph("Hardware recommendation", h2))
    rc = payload["recommended_configuration"]
    story.extend(_table_block([
        ["Hardware", rc.get("hardware_display_name")],
        ["GPU count", rc.get("gpu_count")],
        ["Runtime", f"{rc.get('runtime_family')} / {rc.get('runtime_mode')}"],
        ["Software stack", rc.get("software_stack")],
        ["Deployment", f"{rc.get('execution_mode')}{' / ' + str(rc.get('mig_profile')) if rc.get('mig_profile') else ''}"],
    ], col_widths=[48 * mm, 102 * mm]))

    story.append(Paragraph("24/7 safe capacity", h2))
    cap = payload["capacity_summary"]
    story.extend(_table_block([
        ["Configured arrival", f"{cap.get('configured_arrival_rate_per_sec', 0.0):.3f} req/s"],
        ["Steady-state capacity", f"{cap.get('steady_state_capacity_per_sec', 0.0):.3f} req/s"],
        ["Safe 24/7 capacity", f"{cap.get('safe_capacity_24x7_per_sec', 0.0):.3f} req/s"],
        ["Headroom ratio", f"{cap.get('capacity_headroom_ratio', 0.0):.2f}"],
        ["Capacity gap", f"{cap.get('capacity_gap_per_sec', 0.0):+.3f} req/s"],
    ], col_widths=[55 * mm, 95 * mm]))

    story.append(Paragraph("Bottleneck", h2))
    bottleneck = payload["bottleneck_analysis"]
    story.append(Paragraph(f"Primary bottleneck stage: <b>{bottleneck.get('bottleneck_stage') or '-'}</b>", body))
    if bottleneck.get("breakdown"):
        story.extend(_table_block([[stage, f"{share * 100:.1f}%"] for stage, share in bottleneck["breakdown"].items()], col_widths=[70 * mm, 35 * mm]))

    story.append(Paragraph("Baseline vs override vs safe profile", h2))
    comparison_rows = [["Scenario", "Planning", "Safe cap", "Latency p95", "GPU util p95", "Band"]]
    for row in payload["comparison_views"]:
        comparison_rows.append([
            row.get("scenario_name"),
            f"{row.get('planning_profile')}{' / override' if row.get('override_active') else ''}",
            f"{float(row.get('safe_capacity_24x7_per_sec') or 0.0):.3f}",
            f"{float(row.get('latency_p95_ms') or 0.0):.0f} ms",
            f"{100 * float(row.get('gpu_util_p95') or 0.0):.1f}%",
            row.get("procurement_band"),
        ])
    story.extend(_table_block(comparison_rows, col_widths=[36 * mm, 32 * mm, 24 * mm, 24 * mm, 24 * mm, 24 * mm], header=True, style=small))

    story.append(Paragraph("Assumptions & limits", h2))
    for note in payload["assumptions_and_limits"].get("notes") or []:
        story.append(Paragraph(f"- {note}", body))

    doc.build(story)
    return output



def _table_block(rows: list[list[Any]], *, col_widths: list[float], header: bool = False, style: ParagraphStyle | None = None):
    style = style or getSampleStyleSheet()["BodyText"]
    processed = []
    for idx, row in enumerate(rows):
        processed.append([Paragraph(str(cell), style) for cell in row])
    table = Table(processed, colWidths=col_widths, hAlign="LEFT")
    table_style = [
        ("BOX", (0, 0), (-1, -1), 0.6, colors.HexColor("#cbd5e1")),
        ("INNERGRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#cbd5e1")),
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e2e8f0") if header else colors.HexColor("#f8fafc")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]
    table.setStyle(TableStyle(table_style))
    return [table, Spacer(1, 4 * mm)]
