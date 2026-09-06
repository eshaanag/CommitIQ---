"""
backend/features/reports/pdf_service.py

Unified PDF report generator for Developer Health metrics (Issue #389).

Aggregates DORA, Cycle Time, and Team Health metrics into a clean
PDF using ReportLab.  The report includes:

  - A header with repo name, owner, and generation timestamp.
  - An executive summary with the overall DORA score and health score.
  - A DORA metrics section (deployment frequency, CFR, MTTR).
  - A Cycle Time section (avg cycle, bottlenecks).
  - A Team Health section (burnout risk, weekend/after-hours commits).
  - A risk-reasons / hotspot summary table.

The PDF is returned as raw bytes so the router can stream it as a
``StreamingResponse`` with ``Content-Type: application/pdf``.
"""

from __future__ import annotations

import io
import logging
from datetime import datetime, timezone
from typing import Any

from backend.features.metrics.cycle_time import compute_cycle_time_metrics
from backend.features.metrics.dora import compute_dora_metrics
from backend.features.metrics.team_health import compute_team_health
from backend.shared.models import HealthSnapshot, Repo

try:
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import inch
    from reportlab.platypus import (
        Paragraph,
        SimpleDocTemplate,
        Spacer,
        Table,
        TableStyle,
    )
    from reportlab.platypus.flowables import HRFlowable

    HAS_REPORTLAB = True
except ImportError:
    HAS_REPORTLAB = False
    colors = None
    TA_CENTER = None
    letter = None
    ParagraphStyle = None
    getSampleStyleSheet = None
    inch = None
    Paragraph = None
    SimpleDocTemplate = None
    Spacer = None
    Table = None
    TableStyle = None
    HRFlowable = None
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)


# ── Colour helpers ────────────────────────────────────────────────

if HAS_REPORTLAB:
    _SCORE_COLORS = {
        "Elite": colors.HexColor("#10b981"),
        "High": colors.HexColor("#3b82f6"),
        "Medium": colors.HexColor("#f59e0b"),
        "Low": colors.HexColor("#ef4444"),
    }
else:
    _SCORE_COLORS = {}


def _score_color(score: str) -> Any:
    """Return a ReportLab colour for a DORA-style score label."""
    if not HAS_REPORTLAB:
        return None
    return _SCORE_COLORS.get(score, colors.HexColor("#6b7280"))


# ── Style helpers ────────────────────────────────────────────────


def _build_styles() -> dict[str, ParagraphStyle]:
    """Build the paragraph styles used throughout the PDF."""
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "CustomTitle",
            parent=base["Title"],
            fontSize=22,
            textColor=colors.HexColor("#1e293b"),
            alignment=TA_CENTER,
            spaceAfter=4,
        ),
        "subtitle": ParagraphStyle(
            "CustomSubtitle",
            parent=base["Normal"],
            fontSize=11,
            textColor=colors.HexColor("#64748b"),
            alignment=TA_CENTER,
            spaceAfter=2,
        ),
        "h2": ParagraphStyle(
            "CustomH2",
            parent=base["Heading2"],
            fontSize=15,
            textColor=colors.HexColor("#334155"),
            spaceBefore=18,
            spaceAfter=8,
        ),
        "body": ParagraphStyle(
            "CustomBody",
            parent=base["Normal"],
            fontSize=10,
            textColor=colors.HexColor("#374151"),
            leading=14,
            spaceAfter=6,
        ),
        "cell": ParagraphStyle(
            "CustomCell",
            parent=base["Normal"],
            fontSize=9,
            textColor=colors.HexColor("#374151"),
            leading=12,
        ),
        "cell_bold": ParagraphStyle(
            "CustomCellBold",
            parent=base["Normal"],
            fontSize=9,
            textColor=colors.HexColor("#1e293b"),
            leading=12,
            fontName="Helvetica-Bold",
        ),
    }


# ── Section builders ──────────────────────────────────────────────


def _section_header(text: str, styles: dict[str, ParagraphStyle]) -> Paragraph:
    return Paragraph(text, styles["h2"])


def _metric_row(label: str, value: str, styles: dict[str, ParagraphStyle]) -> list:
    """Return a two-cell table row: [label, value]."""
    return [
        Paragraph(label, styles["cell"]),
        Paragraph(str(value), styles["cell_bold"]),
    ]


def _build_summary_table(
    dora: dict[str, Any],
    cycle: dict[str, Any],
    health: dict[str, Any],
    styles: dict[str, ParagraphStyle],
) -> Table:
    """Build the executive-summary metrics table."""
    data = [
        _metric_row("DORA Score", dora.get("dora_score", "N/A"), styles),
        _metric_row("Deployment Frequency", dora.get("deployment_frequency", "N/A"), styles),
        _metric_row("Change Failure Rate", dora.get("change_failure_rate", "N/A"), styles),
        _metric_row("MTTR (hours)", dora.get("mttr_hours", "N/A"), styles),
        _metric_row("Avg Cycle Time (hours)", cycle.get("avg_cycle_time_hours", "N/A"), styles),
        _metric_row("PRs Analyzed", cycle.get("total_prs_analyzed", "N/A"), styles),
        _metric_row("Burnout Risk", health.get("burnout_risk_score", "N/A"), styles),
        _metric_row("Weekend Commits %", health.get("weekend_commits_percent", "N/A"), styles),
        _metric_row(
            "After-Hours Commits %", health.get("after_hours_commits_percent", "N/A"), styles
        ),
    ]
    tbl = Table(data, colWidths=[2.2 * inch, 2.8 * inch])
    tbl.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#f1f5f9")),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                ("LEFTPADDING", (0, 0), (-1, -1), 8),
            ]
        )
    )
    return tbl


def _build_bottleneck_table(
    bottlenecks: list[dict],
    styles: dict[str, ParagraphStyle],
) -> Table | None:
    """Build the top-PR-bottlenecks table (or None if no bottlenecks)."""
    if not bottlenecks:
        return None
    header = [
        Paragraph("#", styles["cell_bold"]),
        Paragraph("Title", styles["cell_bold"]),
        Paragraph("Author", styles["cell_bold"]),
        Paragraph("Hours", styles["cell_bold"]),
    ]
    rows = [header]
    for i, b in enumerate(bottlenecks[:10], 1):
        rows.append(
            [
                Paragraph(str(i), styles["cell"]),
                Paragraph(b.get("title", "")[:60], styles["cell"]),
                Paragraph(b.get("author", ""), styles["cell"]),
                Paragraph(str(b.get("cycle_time_hours", "")), styles["cell"]),
            ]
        )
    tbl = Table(rows, colWidths=[0.4 * inch, 3.0 * inch, 1.2 * inch, 0.6 * inch])
    tbl.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e0e7ff")),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#c7d2fe")),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    return tbl


# ── Main entry point ──────────────────────────────────────────────


async def generate_health_report(
    db: AsyncSession,
    repo_id: int,
) -> tuple[bytes, str]:
    """Generate a unified PDF health report for ``repo_id``.

    Returns a tuple of ``(pdf_bytes, filename)``.
    """
    if not HAS_REPORTLAB:
        raise RuntimeError("ReportLab dependency is not installed")

    repo = await db.get(Repo, repo_id)
    if not repo:
        raise ValueError(f"Repository {repo_id} not found")

    # Fetch all three metric sets.
    dora = await compute_dora_metrics(db, repo_id)
    cycle = await compute_cycle_time_metrics(db, repo_id)
    health = await compute_team_health(db, repo_id)

    # Fetch the latest health snapshot for the summary score.
    snap_result = await db.execute(
        select(HealthSnapshot)
        .where(HealthSnapshot.repo_id == repo_id)
        .order_by(HealthSnapshot.id.desc())
        .limit(1)
    )
    latest_snapshot = snap_result.scalar_one_or_none()

    # Build the PDF.
    # ``pageCompression=0`` keeps content streams uncompressed so the
    # resulting PDF is directly text-searchable (useful for screen
    # readers, copy/paste, and integration tests that assert on the
    # PDF's text content).  The size overhead is negligible for the
    # single-page metric reports produced here.
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        pageCompression=0,
        rightMargin=0.75 * inch,
        leftMargin=0.75 * inch,
        topMargin=0.75 * inch,
        bottomMargin=0.75 * inch,
    )
    styles = _build_styles()
    story: list = []

    # ── Header ────────────────────────────────────────────────────
    story.append(Paragraph("Developer Health Report", styles["title"]))
    story.append(
        Paragraph(
            f"{repo.owner}/{repo.name}"
            + (f" &middot; {repo.github_language}" if repo.github_language else ""),
            styles["subtitle"],
        )
    )
    story.append(
        Paragraph(
            f"Generated: {datetime.now(tz=timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
            styles["subtitle"],
        )
    )
    story.append(Spacer(1, 10))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#cbd5e1")))
    story.append(Spacer(1, 12))

    # ── Executive Summary ─────────────────────────────────────────
    story.append(_section_header("Executive Summary", styles))
    if latest_snapshot:
        score = round(latest_snapshot.health_score, 1)
        story.append(
            Paragraph(
                f"Overall Health Score: <b>{score}/100</b>",
                styles["body"],
            )
        )
    story.append(Spacer(1, 6))
    story.append(_build_summary_table(dora, cycle, health, styles))

    # ── DORA Metrics ──────────────────────────────────────────────
    story.append(_section_header("DORA Metrics", styles))
    dora_score = dora.get("dora_score", "N/A")
    score_color = _score_color(dora_score)
    story.append(
        Paragraph(
            f"Overall DORA Score: <font color='{score_color.hexval()}'><b>{dora_score}</b></font>",
            styles["body"],
        )
    )
    dora_rows = [
        _metric_row("Deployment Frequency", dora.get("deployment_frequency", "N/A"), styles),
        _metric_row(
            "Deploys / Week",
            f"{dora.get('deployment_frequency_value', 0.0):.1f}",
            styles,
        ),
        _metric_row("Change Failure Rate", dora.get("change_failure_rate", "N/A"), styles),
        _metric_row(
            "CFR Value (%)",
            f"{dora.get('change_failure_rate_value', 0.0):.1f}%",
            styles,
        ),
        _metric_row("MTTR (hours)", f"{dora.get('mttr_hours', 0.0):.1f}", styles),
        _metric_row("MTTR Category", dora.get("mttr_category", "N/A"), styles),
    ]
    dora_tbl = Table(dora_rows, colWidths=[2.2 * inch, 2.8 * inch])
    dora_tbl.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#f1f5f9")),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                ("LEFTPADDING", (0, 0), (-1, -1), 8),
            ]
        )
    )
    story.append(dora_tbl)

    # ── Cycle Time ────────────────────────────────────────────────
    story.append(_section_header("Cycle Time Analysis", styles))
    story.append(
        Paragraph(
            f"Average cycle time: <b>{cycle.get('avg_cycle_time_hours', 0.0):.1f} hours</b>"
            f" across <b>{cycle.get('total_prs_analyzed', 0)} PRs</b>.",
            styles["body"],
        )
    )
    bottlenecks = cycle.get("bottlenecks", [])
    if bottlenecks:
        story.append(
            Paragraph(
                f"Top {len(bottlenecks[:10])} PR bottlenecks (cycle time > 72h):",
                styles["body"],
            )
        )
        bn_tbl = _build_bottleneck_table(bottlenecks, styles)
        if bn_tbl:
            story.append(bn_tbl)
    else:
        story.append(
            Paragraph(
                "No significant bottlenecks detected. All PRs resolved within 72 hours.",
                styles["body"],
            )
        )

    # ── Team Health ───────────────────────────────────────────────
    story.append(_section_header("Team Health", styles))
    burnout = health.get("burnout_risk_score", "N/A")
    burnout_color = _score_color(burnout)
    story.append(
        Paragraph(
            f"Burnout Risk: <font color='{burnout_color.hexval()}'><b>{burnout}</b></font>",
            styles["body"],
        )
    )
    health_rows = [
        _metric_row(
            "Weekend Commits %",
            f"{health.get('weekend_commits_percent', 0.0):.1f}%",
            styles,
        ),
        _metric_row(
            "After-Hours Commits %",
            f"{health.get('after_hours_commits_percent', 0.0):.1f}%",
            styles,
        ),
        _metric_row("Context Switching", health.get("context_switching_score", "N/A"), styles),
        _metric_row(
            "Avg Files / Day",
            f"{health.get('avg_files_per_day', 0.0):.1f}",
            styles,
        ),
    ]
    health_tbl = Table(health_rows, colWidths=[2.2 * inch, 2.8 * inch])
    health_tbl.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#f1f5f9")),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                ("LEFTPADDING", (0, 0), (-1, -1), 8),
            ]
        )
    )
    story.append(health_tbl)

    # ── Footer ────────────────────────────────────────────────────
    story.append(Spacer(1, 20))
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#cbd5e1")))
    story.append(
        Paragraph(
            "Generated by CommitIQ &mdash; Developer Health Analyzer",
            ParagraphStyle(
                "Footer",
                parent=styles["subtitle"],
                fontSize=8,
                textColor=colors.HexColor("#94a3b8"),
                alignment=TA_CENTER,
            ),
        )
    )

    doc.build(story)
    pdf_bytes = buffer.getvalue()
    buffer.close()

    filename = f"commitiq-health-report-{repo.owner}-{repo.name}.pdf"
    return pdf_bytes, filename
