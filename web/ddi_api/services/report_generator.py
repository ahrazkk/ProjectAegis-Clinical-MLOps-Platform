"""
Project Aegis - PDF Report Generator  (Super Report v2)

Two-tier report system:
  • Standard — data-only clinical report with risk assessment, stats, and conclusion.
  • Advanced — **super report** with visual diagrams (risk gauge, interaction heatmap,
               organ burden bars, CYP colored grid, factor radar), per-drug pharmacology
               profiles, AI narrative (Gemini), regimen sensitivity analysis, evidence
               chain, FAERS real-world evidence, food/herbal warnings, page numbers,
               and table of contents.
"""

import io
import logging
import math
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT, TA_JUSTIFY
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch, cm
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    HRFlowable,
    PageBreak,
    KeepTogether,
)
from reportlab.graphics.shapes import Drawing, Rect, Line, Circle, String, Wedge, Group
from reportlab.graphics.charts.barcharts import HorizontalBarChart
from reportlab.graphics import renderPDF

logger = logging.getLogger(__name__)

PAGE_W, PAGE_H = letter  # 612 x 792

# ---------------------------------------------------------------------------
# Brand palette
# ---------------------------------------------------------------------------
BRAND = {
    'dark': '#0f172a',
    'slate': '#1e293b',
    'body': '#334155',
    'muted': '#64748b',
    'light_muted': '#94a3b8',
    'border': '#475569',
    'border_light': '#334155',
    'bg_alt': '#1e293b',
    'bg_light': '#1e293b',
    'bg_panel': '#0f172a',
    'bg_card': '#162032',
    'cyan': '#06b6d4',
    'cyan_dark': '#0891b2',
    'green': '#22c55e',
    'yellow': '#eab308',
    'orange': '#f97316',
    'red': '#ef4444',
    'purple': '#a855f7',
    'white': '#ffffff',
    'teal': '#14b8a6',
    'indigo': '#6366f1',
    'pink': '#ec4899',
    'text_primary': '#e2e8f0',
    'text_secondary': '#cbd5e1',
    'text_muted': '#94a3b8',
}

# Risk tier colors
RISK_TIERS = [
    (0.3, BRAND['green'],  'LOW'),
    (0.6, BRAND['yellow'], 'MODERATE'),
    (0.8, BRAND['orange'], 'HIGH'),
    (1.1, BRAND['red'],    'CRITICAL'),
]


def _risk_color(score: float) -> colors.Color:
    for threshold, col, _ in RISK_TIERS:
        if score < threshold:
            return colors.HexColor(col)
    return colors.HexColor(BRAND['red'])


def _risk_label(score: float) -> str:
    for threshold, _, label in RISK_TIERS:
        if score < threshold:
            return label
    return 'CRITICAL'


def _severity_color(sev: str) -> str:
    s = (sev or '').lower().replace('_', ' ')
    if s in ('critical', 'contraindicated'):
        return BRAND['red']
    if s in ('severe', 'major'):
        return BRAND['orange']
    if s == 'moderate':
        return BRAND['yellow']
    if s in ('minor', 'low'):
        return BRAND['green']
    return BRAND['muted']


def _sanitize(text: str, max_len: int = 600) -> str:
    if not text:
        return ''
    text = re.sub(r'[<>&]', '', str(text))
    return text[:max_len]


def _safe_float(val, default=0.0):
    if isinstance(val, dict):
        return default
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


def build_filename(drug_names: List[str], report_type: str, tier: str) -> str:
    sanitized = '_'.join(
        re.sub(r'[^a-zA-Z0-9]', '', str(d))[:15] for d in (drug_names or [])[:5]
    )
    ts = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
    return f"aegis_{tier}_{sanitized or 'analysis'}_{ts}.pdf"


# ---------------------------------------------------------------------------
# Page numbering callback
# ---------------------------------------------------------------------------
def _page_footer(canvas, doc):
    canvas.saveState()
    # Dark page background
    canvas.setFillColor(colors.HexColor(BRAND['dark']))
    canvas.rect(0, 0, PAGE_W, PAGE_H, fill=1, stroke=0)
    # Footer text
    canvas.setFont('Helvetica', 7)
    canvas.setFillColor(colors.HexColor(BRAND['text_muted']))
    canvas.drawCentredString(PAGE_W / 2, 28,
                              f'Project Aegis  |  Drug-Drug Interaction Intelligence Platform  |  Page {doc.page}')
    # Top cyan accent line
    canvas.setStrokeColor(colors.HexColor(BRAND['cyan']))
    canvas.setLineWidth(2)
    canvas.line(46, PAGE_H - 34, PAGE_W - 46, PAGE_H - 34)
    # Bottom subtle line
    canvas.setStrokeColor(colors.HexColor(BRAND['border']))
    canvas.setLineWidth(0.5)
    canvas.line(46, 42, PAGE_W - 46, 42)
    canvas.restoreState()


# ---------------------------------------------------------------------------
# Styles
# ---------------------------------------------------------------------------
def _build_styles():
    ss = getSampleStyleSheet()
    s = {}
    s['title'] = ParagraphStyle('T', parent=ss['Title'], fontSize=24, leading=28,
                                 textColor=colors.HexColor(BRAND['text_primary']),
                                 spaceAfter=2, alignment=TA_CENTER)
    s['subtitle'] = ParagraphStyle('ST', parent=ss['Normal'], fontSize=11, leading=15,
                                    textColor=colors.HexColor(BRAND['text_muted']),
                                    alignment=TA_CENTER, spaceAfter=4)
    s['section'] = ParagraphStyle('Sec', parent=ss['Heading2'], fontSize=13, leading=17,
                                   textColor=colors.HexColor(BRAND['cyan']),
                                   spaceBefore=14, spaceAfter=5)
    s['subsection'] = ParagraphStyle('Sub', parent=ss['Heading3'], fontSize=11, leading=14,
                                      textColor=colors.HexColor(BRAND['teal']),
                                      spaceBefore=8, spaceAfter=3)
    s['body'] = ParagraphStyle('B', parent=ss['Normal'], fontSize=10, leading=14,
                                textColor=colors.HexColor(BRAND['text_secondary']),
                                spaceAfter=5, alignment=TA_JUSTIFY)
    s['body_small'] = ParagraphStyle('BS', parent=ss['Normal'], fontSize=9, leading=12,
                                      textColor=colors.HexColor(BRAND['text_muted']), spaceAfter=3)
    s['narrative'] = ParagraphStyle('N', parent=ss['Normal'], fontSize=10, leading=15,
                                     textColor=colors.HexColor(BRAND['text_secondary']),
                                     spaceAfter=6, alignment=TA_JUSTIFY,
                                     leftIndent=8, rightIndent=8,
                                     borderColor=colors.HexColor(BRAND['cyan']),
                                     borderWidth=1, borderPadding=8,
                                     backColor=colors.HexColor(BRAND['bg_card']))
    s['label'] = ParagraphStyle('L', parent=ss['Normal'], fontSize=9, leading=12,
                                 textColor=colors.HexColor(BRAND['text_muted']))
    s['stat_value'] = ParagraphStyle('SV', parent=ss['Normal'], fontSize=18, leading=22,
                                      alignment=TA_CENTER, spaceAfter=2)
    s['stat_label'] = ParagraphStyle('SL', parent=ss['Normal'], fontSize=8, leading=10,
                                      textColor=colors.HexColor(BRAND['text_muted']),
                                      alignment=TA_CENTER)
    s['disclaimer'] = ParagraphStyle('D', parent=ss['Normal'], fontSize=8, leading=11,
                                      textColor=colors.HexColor(BRAND['text_muted']),
                                      spaceBefore=16, alignment=TA_CENTER)
    s['toc'] = ParagraphStyle('TOC', parent=ss['Normal'], fontSize=10, leading=15,
                               textColor=colors.HexColor(BRAND['text_secondary']),
                               leftIndent=12, spaceAfter=1)
    s['drug_header'] = ParagraphStyle('DH', parent=ss['Heading3'], fontSize=12, leading=15,
                                       textColor=colors.HexColor(BRAND['cyan']),
                                       spaceBefore=6, spaceAfter=3,
                                       borderColor=colors.HexColor(BRAND['cyan']),
                                       borderWidth=0.5, borderPadding=(4, 4, 4, 4))
    return s


DISCLAIMER_TEXT = (
    "DISCLAIMER: This report is generated by an AI-assisted system for informational "
    "and research purposes only. It does not constitute medical advice. Always consult "
    "qualified healthcare professionals before making clinical decisions. Drug interaction "
    "predictions are probabilistic and should be verified against primary literature."
)


# ═══════════════════════════════════════════════════════════════════════════
# VISUAL DIAGRAM BUILDERS  (ReportLab Drawing objects)
# ═══════════════════════════════════════════════════════════════════════════

def _draw_risk_gauge(risk_score: float, width: float = 220, height: float = 130) -> Drawing:
    """Compact semi-circular risk gauge (speedometer) with needle."""
    d = Drawing(width, height)
    cx, cy = width / 2, 30
    radius = 75
    segments = [
        (0.0, 0.3, BRAND['green']),
        (0.3, 0.6, BRAND['yellow']),
        (0.6, 0.8, BRAND['orange']),
        (0.8, 1.0, BRAND['red']),
    ]
    for lo, hi, col in segments:
        start_angle = 180 - (hi * 180)
        extent = (hi - lo) * 180
        w = Wedge(cx, cy, radius, start_angle, start_angle + extent,
                  fillColor=colors.HexColor(col), strokeColor=colors.HexColor(BRAND['dark']), strokeWidth=1)
        d.add(w)
    # Inner dark circle for donut
    d.add(Circle(cx, cy, radius * 0.60, fillColor=colors.HexColor(BRAND['dark']), strokeColor=None))
    # Needle
    angle_rad = math.radians(180 - (min(max(risk_score, 0), 1.0) * 180))
    needle_len = radius * 0.52
    nx = cx + needle_len * math.cos(angle_rad)
    ny = cy + needle_len * math.sin(angle_rad)
    d.add(Line(cx, cy, nx, ny, strokeColor=colors.HexColor(BRAND['text_primary']), strokeWidth=2))
    d.add(Circle(cx, cy, 4, fillColor=colors.HexColor(BRAND['cyan']), strokeColor=None))
    # Score text
    d.add(String(cx, cy + 16, f'{risk_score:.2f}', fontSize=16,
                 fillColor=colors.HexColor(BRAND['text_primary']),
                 textAnchor='middle', fontName='Helvetica-Bold'))
    d.add(String(cx, cy - 6, _risk_label(risk_score), fontSize=8,
                 fillColor=colors.HexColor(BRAND['text_muted']), textAnchor='middle'))
    # Tier labels
    d.add(String(cx - radius - 4, cy - 4, '0', fontSize=6, fillColor=colors.HexColor(BRAND['text_muted'])))
    d.add(String(cx + radius + 1, cy - 4, '1.0', fontSize=6, fillColor=colors.HexColor(BRAND['text_muted'])))
    return d


def _draw_interaction_heatmap(drug_names: List[str], interactions: list,
                               width: float = 380) -> Optional[Drawing]:
    """NxN colored grid showing pairwise risk scores — dark theme."""
    n = len(drug_names)
    if n < 2 or n > 8:
        return None
    cell = min(45, 300 // n)
    label_w = 75
    total_w = label_w + n * cell + 10
    total_h = 22 + n * cell + 10
    d = Drawing(total_w, total_h)

    risk_map = {}
    for edge in interactions:
        src = (edge.get('source') or edge.get('drug_a', '')).lower()
        tgt = (edge.get('target') or edge.get('drug_b', '')).lower()
        score = _safe_float(edge.get('risk_score'), 0)
        risk_map[(src, tgt)] = score
        risk_map[(tgt, src)] = score

    names_lower = [str(dn).lower() for dn in drug_names]

    for j, name in enumerate(drug_names):
        short = str(name)[:6]
        x = label_w + j * cell + cell / 2
        d.add(String(x, total_h - 8, short, fontSize=6,
                     fillColor=colors.HexColor(BRAND['text_muted']),
                     textAnchor='middle', fontName='Helvetica-Bold'))

    for i, name_i in enumerate(drug_names):
        y = total_h - 20 - i * cell
        short_i = str(name_i)[:9]
        d.add(String(label_w - 3, y + cell / 2 - 3, short_i, fontSize=7,
                     fillColor=colors.HexColor(BRAND['text_secondary']),
                     textAnchor='end', fontName='Helvetica-Bold'))
        for j, name_j in enumerate(drug_names):
            x = label_w + j * cell
            if i == j:
                fill = colors.HexColor(BRAND['slate'])
                label = '—'
            else:
                score = risk_map.get((names_lower[i], names_lower[j]), 0)
                fill = _risk_color(score) if score > 0 else colors.HexColor(BRAND['bg_card'])
                label = f'{score:.1f}' if score > 0 else '—'
            d.add(Rect(x, y, cell, cell, fillColor=fill,
                       strokeColor=colors.HexColor(BRAND['dark']), strokeWidth=1))
            tc = colors.white if score > 0 and i != j else colors.HexColor(BRAND['text_muted'])
            d.add(String(x + cell / 2, y + cell / 2 - 3, label, fontSize=7,
                         fillColor=tc, textAnchor='middle', fontName='Helvetica-Bold'))
    return d


def _draw_organ_bars(organ_data: dict, width: float = 460, max_bars: int = 8) -> Optional[Drawing]:
    """Horizontal bar chart for organ system severity."""
    if not organ_data:
        return None
    items = []
    for sys_name, val in organ_data.items():
        score = _safe_float(val if not isinstance(val, dict) else (val.get('severity', 0)), 0)
        if score > 0:
            items.append((sys_name.replace('_', ' ').title(), score))
    items.sort(key=lambda x: x[1], reverse=True)
    items = items[:max_bars]
    if not items:
        return None

    bar_h = 18
    gap = 6
    label_w = 130
    chart_w = width - label_w - 40
    total_h = len(items) * (bar_h + gap) + 20
    d = Drawing(width, total_h)

    for idx, (name, score) in enumerate(items):
        y = total_h - 20 - idx * (bar_h + gap)
        # Label
        d.add(String(label_w - 4, y + 4, name[:18], fontSize=8,
                     fillColor=colors.HexColor(BRAND['text_secondary']),
                     textAnchor='end'))
        # Background bar
        d.add(Rect(label_w, y, chart_w, bar_h, fillColor=colors.HexColor(BRAND['bg_card']),
                   strokeColor=colors.HexColor(BRAND['border']), strokeWidth=0.5))
        # Filled bar
        fill_w = max(2, chart_w * min(score, 1.0))
        d.add(Rect(label_w, y, fill_w, bar_h, fillColor=_risk_color(score), strokeColor=None))
        # Score label
        d.add(String(label_w + fill_w + 4, y + 4, f'{score:.2f}', fontSize=8,
                     fillColor=colors.HexColor(BRAND['text_muted'])))
    return d


def _draw_factor_bars(factors: dict, width: float = 460) -> Optional[Drawing]:
    """Horizontal stacked bar chart for polypharmacy risk factor breakdown."""
    factor_defs = [
        ('pairwise_baseline', 'Pairwise Baseline', 0.40, BRAND['red']),
        ('enzyme_competition', 'CYP Enzyme Load', 0.25, BRAND['purple']),
        ('target_overlap', 'Target Overlap', 0.15, BRAND['indigo']),
        ('organ_burden', 'Organ Burden', 0.10, BRAND['orange']),
        ('network_stress', 'Network Stress', 0.10, BRAND['teal']),
    ]
    items = []
    for key, label, weight, col in factor_defs:
        raw = factors.get(key, 0)
        if isinstance(raw, dict):
            raw = raw.get('score', raw.get('value', 0))
        score = _safe_float(raw, 0)
        items.append((label, weight, score, col))

    bar_h = 22
    gap = 8
    label_w = 120
    chart_w = width - label_w - 60
    total_h = len(items) * (bar_h + gap) + 30
    d = Drawing(width, total_h)

    # Title
    d.add(String(width / 2, total_h - 5, 'Risk Factor Contributions', fontSize=9,
                 fillColor=colors.HexColor(BRAND['cyan']), textAnchor='middle',
                 fontName='Helvetica-Bold'))

    for idx, (label, weight, score, col) in enumerate(items):
        y = total_h - 25 - idx * (bar_h + gap)
        # Label + weight
        d.add(String(label_w - 4, y + 6, f'{label} ({weight:.0%})', fontSize=8,
                     fillColor=colors.HexColor(BRAND['text_secondary']), textAnchor='end'))
        # Background
        d.add(Rect(label_w, y, chart_w, bar_h, fillColor=colors.HexColor(BRAND['bg_card']),
                   strokeColor=colors.HexColor(BRAND['border']), strokeWidth=0.5))
        # Fill
        fill_w = max(2, chart_w * min(score, 1.0))
        d.add(Rect(label_w, y, fill_w, bar_h, fillColor=colors.HexColor(col), strokeColor=None))
        # Score
        d.add(String(label_w + chart_w + 4, y + 6, f'{score:.3f}' if score else '—',
                     fontSize=8, fillColor=colors.HexColor(BRAND['text_muted'])))
    return d


def _draw_cyp_heatmap(drug_names: List[str], profiles: dict,
                       width: float = 460) -> Optional[Drawing]:
    """Colored CYP450 grid — green=substrate, red=inhibitor, blue=inducer."""
    enzymes = ['CYP1A2', 'CYP2C9', 'CYP2C19', 'CYP2D6', 'CYP3A4']
    if not profiles:
        return None
    n_drugs = len(drug_names)
    cell_w = 65
    cell_h = 24
    label_w = 90
    total_w = label_w + len(enzymes) * cell_w + 5
    total_h = 25 + (n_drugs + 1) * cell_h + 10
    d = Drawing(total_w, total_h)

    role_colors = {
        'Sub': '#2dd4bf',       # teal
        'Inh(S)': '#ef4444',    # red
        'Inh(M)': '#f97316',    # orange
        'Inh(W)': '#eab308',    # yellow
        'Ind': '#6366f1',       # indigo
    }

    # Header row
    for j, enz in enumerate(enzymes):
        x = label_w + j * cell_w
        d.add(Rect(x, total_h - 25, cell_w, cell_h,
                   fillColor=colors.HexColor(BRAND['slate']), strokeColor=colors.HexColor(BRAND['border']), strokeWidth=0.5))
        d.add(String(x + cell_w / 2, total_h - 19, enz, fontSize=8,
                     fillColor=colors.white, textAnchor='middle', fontName='Helvetica-Bold'))

    # Drug rows
    for i, drug in enumerate(drug_names):
        y = total_h - 25 - (i + 1) * cell_h
        # Drug label
        d.add(String(label_w - 4, y + 7, str(drug)[:12], fontSize=8,
                     fillColor=colors.HexColor(BRAND['text_primary']),
                     textAnchor='end', fontName='Helvetica-Bold'))
        profile = profiles.get(str(drug).lower(), {})
        for j, enz in enumerate(enzymes):
            x = label_w + j * cell_w
            roles = profile.get(enz, [])
            if roles:
                short_roles = []
                for r in roles:
                    if 'substrate' in r: short_roles.append('Sub')
                    elif 'strong' in r: short_roles.append('Inh(S)')
                    elif 'moderate' in r: short_roles.append('Inh(M)')
                    elif 'weak' in r: short_roles.append('Inh(W)')
                    elif 'inducer' in r: short_roles.append('Ind')
                role_text = ','.join(short_roles) if short_roles else '—'
                first_role = short_roles[0] if short_roles else None
                bg = colors.HexColor(role_colors.get(first_role, BRAND['bg_light']))
            else:
                role_text = '—'
                bg = colors.HexColor(BRAND['bg_alt'])

            d.add(Rect(x, y, cell_w, cell_h, fillColor=bg,
                       strokeColor=colors.HexColor(BRAND['border']), strokeWidth=0.5))
            text_col = colors.white if role_text != '—' else colors.HexColor(BRAND['text_muted'])
            d.add(String(x + cell_w / 2, y + 7, role_text, fontSize=7,
                         fillColor=text_col, textAnchor='middle',
                         fontName='Helvetica-Bold' if role_text != '—' else 'Helvetica'))

    # Legend
    legend_y = total_h - 25 - (n_drugs + 1) * cell_h - 5
    leg_x = label_w
    for abbr, col in role_colors.items():
        d.add(Rect(leg_x, legend_y, 10, 10, fillColor=colors.HexColor(col), strokeColor=None))
        d.add(String(leg_x + 13, legend_y + 1, abbr, fontSize=6,
                     fillColor=colors.HexColor(BRAND['text_muted'])))
        leg_x += 55

    return d


def _draw_evidence_bars(evidence_sources: dict, width: float = 460) -> Optional[Drawing]:
    """Evidence source strength visualization."""
    source_weights = {
        'knowledge_graph': ('Knowledge Graph', 1.0, BRAND['cyan']),
        'ddi_corpus': ('DDI Corpus', 0.9, BRAND['teal']),
        'twosides': ('TWOSIDES', 0.78, BRAND['indigo']),
        'openfda_faers': ('OpenFDA FAERS', 0.62, BRAND['purple']),
        'categorical_rules': ('Categorical Rules', 0.85, BRAND['green']),
        'gnn_model': ('GNN Model', 0.7, BRAND['orange']),
    }
    # Determine which sources were used
    used_sources = set()
    if isinstance(evidence_sources, list):
        used_sources = set(evidence_sources)
    elif isinstance(evidence_sources, str):
        used_sources = {s.strip() for s in evidence_sources.split(',')}
    elif isinstance(evidence_sources, dict):
        used_sources = set(evidence_sources.keys())

    items = []
    for key, (label, weight, col) in source_weights.items():
        active = any(key.replace('_', '') in s.replace('_', '').lower() for s in used_sources) if used_sources else False
        items.append((label, weight, col, active))

    bar_h = 16
    gap = 5
    label_w = 110
    chart_w = width - label_w - 50
    total_h = len(items) * (bar_h + gap) + 15
    d = Drawing(width, total_h)

    for idx, (label, weight, col, active) in enumerate(items):
        y = total_h - 12 - idx * (bar_h + gap)
        d.add(String(label_w - 4, y + 3, label, fontSize=7,
                     fillColor=colors.HexColor(BRAND['text_secondary'] if active else BRAND['text_muted']),
                     textAnchor='end'))
        d.add(Rect(label_w, y, chart_w, bar_h,
                   fillColor=colors.HexColor(BRAND['bg_card']), strokeColor=colors.HexColor(BRAND['border']), strokeWidth=0.5))
        fill_w = chart_w * weight
        fill_col = colors.HexColor(col) if active else colors.HexColor(BRAND['border'])
        d.add(Rect(label_w, y, fill_w, bar_h, fillColor=fill_col, strokeColor=None))
        status = '●' if active else '○'
        d.add(String(label_w + chart_w + 4, y + 3,
                     f'{status} {weight:.0%}', fontSize=7,
                     fillColor=colors.HexColor(BRAND['text_secondary'] if active else BRAND['text_muted'])))
    return d


# ═══════════════════════════════════════════════════════════════════════════
# REUSABLE ELEMENT BUILDERS
# ═══════════════════════════════════════════════════════════════════════════

def _header(styles, report_title: str, drug_names: List[str], tier: str):
    elements = []
    elements.append(Paragraph('PROJECT AEGIS', styles['title']))
    elements.append(Paragraph(report_title, styles['subtitle']))
    if drug_names:
        drug_str = '  •  '.join(str(d) for d in drug_names)
        elements.append(Paragraph(drug_str, styles['subtitle']))
    ts = datetime.now(timezone.utc).strftime('%Y-%m-%d  %H:%M UTC')
    tier_label = 'Advanced Clinical Report' if tier == 'advanced' else 'Standard Report'
    elements.append(Paragraph(f'{tier_label}  |  Generated: {ts}', styles['label']))
    elements.append(Spacer(1, 6))
    elements.append(HRFlowable(width='100%', thickness=2,
                                color=colors.HexColor(BRAND['cyan']), spaceAfter=12))
    return elements


def _disclaimer(styles):
    return [
        HRFlowable(width='100%', thickness=0.5, color=colors.HexColor(BRAND['border_light'])),
        Paragraph(DISCLAIMER_TEXT, styles['disclaimer']),
    ]


def _section_line(styles, title: str):
    return [
        Spacer(1, 3),
        HRFlowable(width='100%', thickness=0.5, color=colors.HexColor(BRAND['border'])),
        Paragraph(title, styles['section']),
    ]


def _kv_table(pairs, styles, col_widths=None):
    data = []
    for label, value in pairs:
        data.append([
            Paragraph(f'<b>{_sanitize(str(label))}</b>', styles['label']),
            Paragraph(_sanitize(str(value)), styles['body']),
        ])
    if not data:
        return Spacer(1, 0)
    widths = col_widths or [2.0 * inch, 4.5 * inch]
    t = Table(data, colWidths=widths)
    t.setStyle(TableStyle([
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ('LEFTPADDING', (0, 0), (0, -1), 0),
        ('LINEBELOW', (0, 0), (-1, -2), 0.5, colors.HexColor(BRAND['border'])),
    ]))
    return t


def _stats_strip(styles, stats: List[Tuple[str, str, str]]):
    row_vals, row_labels = [], []
    for label, value, color in stats:
        vs = ParagraphStyle(f'SV_{label}', parent=styles['stat_value'],
                            textColor=colors.HexColor(color))
        row_vals.append(Paragraph(str(value), vs))
        row_labels.append(Paragraph(label.upper(), styles['stat_label']))
    if not row_vals:
        return Spacer(1, 0)
    col_w = 6.5 * inch / len(row_vals)
    t = Table([row_vals, row_labels], colWidths=[col_w] * len(row_vals))
    t.setStyle(TableStyle([
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING', (0, 0), (-1, -1), 8),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor(BRAND['bg_card'])),
        ('BOX', (0, 0), (-1, -1), 1, colors.HexColor(BRAND['border'])),
        ('LINEBEFORE', (1, 0), (-1, -1), 0.5, colors.HexColor(BRAND['border'])),
    ]))
    return t


def _data_table(headers, rows, styles, col_widths=None):
    if not rows:
        return Spacer(1, 0)
    wrapped_h = [Paragraph(f'<b>{h}</b>', styles['label']) for h in headers]
    wrapped_r = [[Paragraph(_sanitize(str(c)), styles['body_small']) for c in row] for row in rows]
    data = [wrapped_h] + wrapped_r
    n = len(headers)
    if col_widths is None:
        col_widths = [6.5 * inch / n] * n
    t = Table(data, colWidths=col_widths)
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor(BRAND['slate'])),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.HexColor(BRAND['cyan'])),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1),
         [colors.HexColor(BRAND['bg_panel']), colors.HexColor(BRAND['bg_card'])]),
        ('TEXTCOLOR', (0, 1), (-1, -1), colors.HexColor(BRAND['text_secondary'])),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING', (0, 0), (-1, -1), 5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
        ('LEFTPADDING', (0, 0), (-1, -1), 6),
        ('RIGHTPADDING', (0, 0), (-1, -1), 6),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor(BRAND['border'])),
    ]))
    return t


# ═══════════════════════════════════════════════════════════════════════════
# SECTION BUILDERS — SHARED
# ═══════════════════════════════════════════════════════════════════════════

def _build_risk_assessment(styles, data: dict):
    elements = []
    elements.extend(_section_line(styles, 'Risk Assessment'))
    risk_score = _safe_float(
        data.get('risk_score') or data.get('regimen_risk_score')
        or (data.get('risk_metrics', {}) or {}).get('regimen_composite_score') or 0)
    severity = str(
        data.get('severity') or data.get('overall_risk_level')
        or data.get('regimen_risk_level') or 'unknown').replace('_', ' ').title()
    confidence = data.get('confidence')
    risk_level = data.get('risk_level') or data.get('regimen_risk_level') or _risk_label(risk_score)
    stats = [
        ('Risk Score', f'{risk_score:.2f}', BRAND['red'] if risk_score >= 0.6 else BRAND['cyan_dark']),
        ('Severity', severity, _severity_color(severity)),
        ('Risk Level', str(risk_level).upper(), BRAND['red'] if risk_score >= 0.6 else BRAND['cyan_dark']),
    ]
    if confidence is not None:
        stats.append(('Confidence', f'{_safe_float(confidence):.0%}', BRAND['teal']))
    elements.append(_stats_strip(styles, stats))
    elements.append(Spacer(1, 8))
    return elements


def _build_mechanism(styles, mechanism: str):
    if not mechanism:
        return []
    e = _section_line(styles, 'Mechanism of Interaction')
    e.append(Paragraph(_sanitize(mechanism, 1200), styles['body']))
    return e


def _build_affected_systems(styles, systems: list):
    if not systems:
        return []
    e = _section_line(styles, 'Affected Body Systems')
    rows = []
    for s in systems:
        if isinstance(s, dict):
            rows.append([s.get('system', '?'), str(s.get('severity', '—')),
                         ', '.join(s.get('symptoms', [])) or '—'])
        else:
            rows.append([str(s), '—', '—'])
    e.append(_data_table(['System', 'Severity', 'Symptoms'], rows, styles,
                          [1.5 * inch, 1.2 * inch, 3.8 * inch]))
    return e


def _build_provenance(styles, prov: dict):
    if not prov:
        return []
    e = _section_line(styles, 'Model Provenance')
    pairs = []
    for key, label in [('model_used', 'Model'), ('prediction_path', 'Prediction Path'),
                       ('calibration_method', 'Calibration'), ('model_version', 'Model Version'),
                       ('calibration_version', 'Cal. Version'), ('fallback_reason', 'Fallback'),
                       ('evidence_source', 'Evidence Sources'), ('known_interaction_source', 'KG Source'),
                       ('known_interaction_severity', 'KG Severity')]:
        v = prov.get(key)
        if v:
            pairs.append((label, str(v)))
    if pairs:
        e.append(_kv_table(pairs, styles))
    return e


def _build_class_warnings(styles, warnings: list):
    if not warnings:
        return []
    e = _section_line(styles, 'Drug Class Warnings')
    for w in warnings:
        w_type = 'DUPLICATE THERAPY' if w.get('type') == 'duplicate_therapy' else 'CLASS INTERACTION'
        sc = BRAND['red'] if w.get('severity') == 'high' else BRAND['orange']
        e.append(Paragraph(
            f'<font color="{sc}"><b>[{w_type}]</b></font>  {_sanitize(w.get("message", ""))}',
            styles['body']))
    return e


def _build_interactions_table(styles, interactions: list):
    if not interactions:
        return []
    e = _section_line(styles, 'Pairwise Interaction Analysis')
    rows = []
    for edge in interactions:
        src = edge.get('source', edge.get('drug_a', ''))
        tgt = edge.get('target', edge.get('drug_b', ''))
        risk = _safe_float(edge.get('risk_score'))
        sev = str(edge.get('severity', '?')).replace('_', ' ').title()
        conf = edge.get('confidence', '')
        mech = (edge.get('mechanism', '') or '')[:60]
        rows.append([src, tgt, f'{risk:.2f}', sev,
                     f'{_safe_float(conf):.0%}' if conf else '—', mech or '—'])
    e.append(_data_table(
        ['Drug A', 'Drug B', 'Risk', 'Severity', 'Conf.', 'Mechanism'], rows, styles,
        [1.05 * inch, 1.05 * inch, 0.55 * inch, 0.85 * inch, 0.55 * inch, 2.45 * inch]))
    return e


def _build_hub_drug(styles, hub: str, count: int):
    if not hub:
        return []
    e = _section_line(styles, 'Hub Drug Analysis')
    e.append(Paragraph(
        f'<b>{_sanitize(hub)}</b> is the most connected drug, involved in <b>{count}</b> '
        f'pairwise interaction(s). This drug requires closest monitoring and potential dose adjustment.',
        styles['body']))
    return e


def _build_conclusion(styles, data: dict, report_type: str):
    e = _section_line(styles, 'Conclusion')
    risk_score = _safe_float(data.get('risk_score') or data.get('regimen_risk_score') or 0)
    severity = str(data.get('severity') or data.get('overall_risk_level') or 'unknown').replace('_', ' ').lower()

    if report_type == 'poly':
        drugs = data.get('drugs', [])
        interactions = data.get('interactions', [])
        overall = data.get('overall_risk_level', 'unknown')
        hub = data.get('hub_drug', '')
        sig = sum(1 for x in interactions if _safe_float(x.get('risk_score')) >= 0.5)
        text = (f'Analysis of {len(drugs)} drugs identified {len(interactions)} pairwise interactions, '
                f'of which {sig} are clinically significant (risk ≥ 0.50). '
                f'The overall regimen risk is classified as <b>{overall.upper()}</b>.')
        if hub:
            text += f'  <b>{hub}</b> is the primary hub drug requiring close monitoring.'
    else:
        da, db = data.get('drug_a', 'Drug A'), data.get('drug_b', 'Drug B')
        text = (f'The interaction between <b>{da}</b> and <b>{db}</b> has been assessed as '
                f'<b>{severity}</b> severity with a composite risk score of <b>{risk_score:.2f}</b> '
                f'({_risk_label(risk_score)}).')
        conf = data.get('confidence')
        if conf:
            text += f'  Model confidence: <b>{_safe_float(conf):.0%}</b>.'
        if risk_score >= 0.7:
            text += ('  This combination warrants clinical attention. Consider alternative '
                     'therapies, dose adjustment, or enhanced monitoring.')
        elif risk_score >= 0.4:
            text += '  Moderate risk detected. Standard therapeutic monitoring is recommended.'
        else:
            text += '  Low risk interaction. No immediate clinical concern.'
    e.append(Paragraph(text, styles['body']))
    return e


# ═══════════════════════════════════════════════════════════════════════════
# SECTION BUILDERS — ADVANCED ONLY
# ═══════════════════════════════════════════════════════════════════════════

def _build_toc(styles, sections: List[str]):
    """Table of contents."""
    e = _section_line(styles, 'Report Contents')
    for i, sec in enumerate(sections, 1):
        e.append(Paragraph(f'{i}.  {sec}', styles['toc']))
    e.append(Spacer(1, 8))
    return e


def _build_llm_section(styles, title: str, text: str):
    if not text:
        return []
    e = _section_line(styles, title)
    e.append(Paragraph(
        f'<i><font color="{BRAND["cyan_dark"]}">AI-Generated Analysis (Gemini)</font></i>',
        styles['label']))
    e.append(Spacer(1, 4))
    for para in text.strip().split('\n\n'):
        clean = para.strip().replace('\n', ' ')
        if clean:
            e.append(Paragraph(_sanitize(clean, 2000), styles['narrative']))
    return e


def _build_drug_profiles(styles, drug_names: List[str]):
    """Per-drug pharmacology profile cards with data from offline DB + CYP."""
    e = _section_line(styles, 'Drug Pharmacology Profiles')
    e.append(Paragraph(
        'Individual pharmacological profiles for each drug in the analyzed regimen.',
        styles['body_small']))

    # Fetch from offline database
    drug_db = {}
    try:
        from .offline_training_data import DRUG_DATABASE
        for di in DRUG_DATABASE:
            drug_db[di.name.lower()] = di
    except Exception:
        pass

    # Fetch CYP profiles
    cyp_profiles = {}
    try:
        from .cyp450_database import get_cyp450_database
        db = get_cyp450_database()
        for dn in drug_names:
            cyp_profiles[str(dn).lower()] = db.get_drug_cyp_profile(str(dn).lower())
    except Exception:
        pass

    for drug in drug_names:
        dl = str(drug).lower()
        info = drug_db.get(dl)

        e.append(Paragraph(f'<b>{str(drug).title()}</b>', styles['drug_header']))
        pairs = []

        if info:
            if info.category:
                pairs.append(('Category', info.category))
            if info.description:
                pairs.append(('Description', info.description[:200]))
            if info.targets:
                pairs.append(('Targets', ', '.join(info.targets[:6])))
            if info.smiles:
                pairs.append(('SMILES', info.smiles[:80]))
            if info.drugbank_id:
                pairs.append(('DrugBank ID', info.drugbank_id))
        else:
            pairs.append(('Note', 'Extended pharmacology data not available in offline database.'))

        # CYP roles
        cyp = cyp_profiles.get(dl, {})
        if cyp:
            roles = []
            for enz, role_list in cyp.items():
                for r in role_list:
                    short = 'substrate' if 'substrate' in r else ('inhibitor' if 'inhibitor' in r else 'inducer')
                    roles.append(f'{enz}: {short}')
            if roles:
                pairs.append(('CYP Metabolism', ', '.join(roles[:6])))

        if pairs:
            e.append(_kv_table(pairs, styles, [1.5 * inch, 5.0 * inch]))
        e.append(Spacer(1, 6))

    return e


def _build_cyp450_visual(styles, drug_names: List[str]):
    """CYP450 section with colored heatmap grid."""
    e = _section_line(styles, 'CYP450 Enzyme Metabolism')
    e.append(Paragraph(
        'Cytochrome P450 enzyme profiles determine drug metabolism. '
        '<font color="#2dd4bf"><b>Teal</b></font> = substrate, '
        '<font color="#ef4444"><b>Red</b></font> = strong inhibitor, '
        '<font color="#f97316"><b>Orange</b></font> = moderate inhibitor, '
        '<font color="#6366f1"><b>Indigo</b></font> = inducer.',
        styles['body_small']))
    e.append(Spacer(1, 4))

    try:
        from .cyp450_database import get_cyp450_database
        db = get_cyp450_database()
    except Exception:
        e.append(Paragraph('CYP450 database unavailable.', styles['body_small']))
        return e

    profiles = {}
    for dn in drug_names:
        profiles[str(dn).lower()] = db.get_drug_cyp_profile(str(dn).lower())

    heatmap = _draw_cyp_heatmap(drug_names, profiles)
    if heatmap:
        e.append(heatmap)

    # Interaction conflicts
    if len(drug_names) >= 2:
        e.append(Spacer(1, 6))
        e.append(Paragraph('CYP Interaction Conflicts', styles['subsection']))
        rows = []
        checked = set()
        for i, d1 in enumerate(drug_names):
            for d2 in drug_names[i + 1:]:
                pair = tuple(sorted([str(d1).lower(), str(d2).lower()]))
                if pair in checked:
                    continue
                checked.add(pair)
                try:
                    ixs = db.check_cyp_interaction(pair[0], pair[1])
                    for ix in ixs:
                        rows.append([str(d1), str(d2), str(getattr(ix, 'enzyme', '')),
                                     getattr(ix, 'mechanism', ''), getattr(ix, 'severity', ''),
                                     getattr(ix, 'clinical_effect', '')[:50],
                                     getattr(ix, 'management', '')[:50]])
                except Exception:
                    pass
        if rows:
            e.append(_data_table(
                ['Drug 1', 'Drug 2', 'Enzyme', 'Mechanism', 'Severity', 'Effect', 'Management'],
                rows, styles,
                [0.75*inch, 0.75*inch, 0.65*inch, 1.1*inch, 0.6*inch, 1.3*inch, 1.35*inch]))
        else:
            e.append(Paragraph('No direct CYP-mediated conflicts detected.', styles['body_small']))
    return e


def _build_body_map_visual(styles, body_map_data: dict, data: dict):
    """Organ impact with horizontal bar chart."""
    bm = body_map_data or data.get('body_map', {}) or {}
    affected = data.get('affected_systems', [])
    if not bm and not affected:
        return []

    e = _section_line(styles, 'Organ System Impact Analysis')
    e.append(Paragraph(
        'Predicted organ-level impact based on pharmacological effects, adverse events, and interaction mechanisms.',
        styles['body_small']))

    # Combine sources
    organ_scores = {}
    if bm:
        for sys_name, val in bm.items():
            score = _safe_float(val if not isinstance(val, dict) else (val.get('severity', 0)), 0)
            organ_scores[sys_name] = score
    if affected:
        for s in affected:
            if isinstance(s, dict):
                organ_scores.setdefault(s.get('system', ''), _safe_float(s.get('severity', 0)))

    # Draw bar chart
    bars = _draw_organ_bars(organ_scores)
    if bars:
        e.append(bars)
    e.append(Spacer(1, 4))

    # Detail table
    if bm:
        rows = []
        for sys_name, info in sorted(bm.items()):
            if isinstance(info, dict):
                sev = _safe_float(info.get('severity', 0))
                se = ', '.join(
                    (se.get('name', str(se)) if isinstance(se, dict) else str(se))
                    for se in (info.get('sideEffects', info.get('side_effects', [])) or [])[:4]
                ) or '—'
                contrib = ', '.join(
                    (f"{c.get('drug', '')} ({c.get('contribution', '')})" if isinstance(c, dict) else str(c))
                    for c in (info.get('drugContributions', info.get('drug_contributions', [])) or [])[:3]
                ) or '—'
            else:
                sev = _safe_float(info)
                se, contrib = '—', '—'
            rows.append([sys_name.replace('_', ' ').title(), f'{sev:.2f}', se, contrib])
        if rows:
            e.append(_data_table(
                ['Organ System', 'Severity', 'Side Effects', 'Drug Contributions'],
                rows, styles, [1.3*inch, 0.7*inch, 2.4*inch, 2.1*inch]))
    return e


def _build_polypharmacy_factors_visual(styles, data: dict):
    """N-order factor breakdown with visual bar chart."""
    dt = data.get('digital_twin_result', {}) or {}
    risk_metrics = data.get('risk_metrics', {}) or {}
    if not dt and not risk_metrics:
        return []

    e = _section_line(styles, 'Polypharmacy Risk Factor Breakdown')

    factors = dt.get('factors', dt.get('factor_scores', {}))
    if factors and isinstance(factors, dict):
        chart = _draw_factor_bars(factors)
        if chart:
            e.append(chart)
        e.append(Spacer(1, 6))

        # Additional factor details
        for key in ['enzyme_competition', 'target_overlap', 'network_stress']:
            fdata = factors.get(key, {})
            if isinstance(fdata, dict):
                conflicts = fdata.get('conflicts', [])
                if conflicts:
                    e.append(Paragraph(f'<b>{key.replace("_", " ").title()} Details:</b>', styles['body_small']))
                    for c in conflicts[:4]:
                        if isinstance(c, dict):
                            parts = [f"{k}: {v}" for k, v in c.items()
                                     if k not in ('score',) and not isinstance(v, (list, dict))]
                            e.append(Paragraph(f'  •  {", ".join(parts[:5])}', styles['body_small']))

    elif risk_metrics and isinstance(risk_metrics, dict):
        kv = []
        for key in ['total_pairs', 'significant_pair_count', 'max_pair_risk',
                     'average_pair_risk', 'regimen_composite_score', 'regimen_risk_level']:
            val = risk_metrics.get(key)
            if val is not None and not isinstance(val, (dict, list)):
                kv.append((key.replace('_', ' ').title(),
                           f'{val:.3f}' if isinstance(val, float) else str(val)))
        if kv:
            e.append(_kv_table(kv, styles))

    # Recommendations
    recs = dt.get('recommendations', [])
    if recs:
        e.append(Spacer(1, 4))
        e.append(Paragraph('<b>Clinical Recommendations:</b>', styles['body_small']))
        for r in recs[:5]:
            e.append(Paragraph(f'  •  {_sanitize(str(r))}', styles['body_small']))

    return e


def _build_regimen_sensitivity(styles, data: dict):
    """Regimen sensitivity analysis — what happens if each drug is removed."""
    interactions = data.get('interactions', [])
    drugs = data.get('drugs', data.get('drug_names', []))
    if len(drugs) < 3 or not interactions:
        return []

    e = _section_line(styles, 'Regimen Sensitivity Analysis')
    e.append(Paragraph(
        'Simulates removing each drug from the regimen to identify which drug contributes most to overall risk. '
        'This is analogous to mutation testing — systematically eliminating one variable to measure its impact.',
        styles['body_small']))
    e.append(Spacer(1, 4))

    # Calculate baseline
    all_scores = [_safe_float(ix.get('risk_score')) for ix in interactions]
    baseline_max = max(all_scores) if all_scores else 0
    baseline_avg = sum(all_scores) / len(all_scores) if all_scores else 0

    rows = []
    for drug in drugs:
        dl = str(drug).lower()
        # Filter out interactions involving this drug
        remaining = [ix for ix in interactions
                     if dl not in (ix.get('source', '').lower(), ix.get('target', '').lower(),
                                   ix.get('drug_a', '').lower(), ix.get('drug_b', '').lower())]
        rem_scores = [_safe_float(ix.get('risk_score')) for ix in remaining]
        new_max = max(rem_scores) if rem_scores else 0
        new_avg = sum(rem_scores) / len(rem_scores) if rem_scores else 0
        delta_max = baseline_max - new_max
        delta_avg = baseline_avg - new_avg
        impact = 'HIGH' if delta_max > 0.2 else ('MEDIUM' if delta_max > 0.05 else 'LOW')
        rows.append([
            str(drug), str(len(remaining)), f'{new_max:.2f}',
            f'{delta_max:+.2f}', f'{delta_avg:+.2f}', impact
        ])

    rows.sort(key=lambda r: -abs(float(r[3])))
    e.append(Paragraph(
        f'<b>Baseline:</b> {len(interactions)} interactions, max risk {baseline_max:.2f}, '
        f'avg risk {baseline_avg:.2f}', styles['body_small']))
    e.append(Spacer(1, 4))
    e.append(_data_table(
        ['Drug Removed', 'Remaining Pairs', 'New Max Risk', 'Δ Max', 'Δ Avg', 'Impact'],
        rows, styles,
        [1.2*inch, 1.0*inch, 1.0*inch, 0.8*inch, 0.8*inch, 1.7*inch]))
    return e


def _build_food_herbal(styles, drug_names: List[str]):
    try:
        from .enhanced_data_ingestion import EnhancedDataIngestion
        svc = EnhancedDataIngestion()
    except Exception:
        return []

    food_rows, herbal_rows = [], []
    for drug in drug_names:
        name = str(drug).strip()
        try:
            for fi in svc.get_food_drug_interactions(drug_name=name):
                food_rows.append([name, fi.food, fi.severity.title(),
                                  fi.mechanism[:80], fi.recommendation[:80]])
        except Exception:
            pass
        try:
            for hi in svc.get_herbal_drug_interactions(drug_name=name):
                herbal_rows.append([name, hi.herb, hi.severity.title(),
                                    hi.mechanism[:80], hi.recommendation[:80]])
        except Exception:
            pass

    if not food_rows and not herbal_rows:
        return []

    e = _section_line(styles, 'Food &amp; Herbal Interaction Warnings')
    if food_rows:
        e.append(Paragraph('Food-Drug Interactions', styles['subsection']))
        e.append(_data_table(
            ['Drug', 'Food', 'Severity', 'Mechanism', 'Recommendation'],
            food_rows[:12], styles,
            [0.9*inch, 1.0*inch, 0.7*inch, 2.0*inch, 1.9*inch]))
    if herbal_rows:
        e.append(Paragraph('Herbal-Drug Interactions', styles['subsection']))
        e.append(_data_table(
            ['Drug', 'Herb', 'Severity', 'Mechanism', 'Recommendation'],
            herbal_rows[:12], styles,
            [0.9*inch, 1.0*inch, 0.7*inch, 2.0*inch, 1.9*inch]))
    return e


def _build_faers_section(styles, drug_names: List[str], report_type: str):
    try:
        from .faers_service import FaersService
        svc = FaersService()
    except Exception:
        return []

    results = []
    if report_type == 'pair' and len(drug_names) >= 2:
        try:
            r = svc.query_adverse_events(drug_names[0], drug_names[1])
            if r and r.total_reports > 0:
                results.append(r)
        except Exception:
            pass
    elif report_type == 'poly':
        checked = set()
        for i, d1 in enumerate(drug_names):
            for d2 in drug_names[i + 1:]:
                pair = tuple(sorted([str(d1).lower(), str(d2).lower()]))
                if pair in checked:
                    continue
                checked.add(pair)
                try:
                    r = svc.query_adverse_events(d1, d2)
                    if r and r.total_reports > 0:
                        results.append(r)
                except Exception:
                    pass
                if len(results) >= 5:
                    break
            if len(results) >= 5:
                break

    if not results:
        return []

    e = _section_line(styles, 'FAERS Real-World Evidence')
    e.append(Paragraph(
        'Post-market adverse event data from the FDA Adverse Event Reporting System (openFDA).',
        styles['body_small']))

    for r in results:
        pair_label = f'{r.drug_pair[0]}  +  {r.drug_pair[1]}' if hasattr(r, 'drug_pair') else '?'
        kv = [('Drug Pair', pair_label), ('Total Reports', f'{r.total_reports:,}'),
              ('Signal Score', f'{r.signal_score:.2f}')]
        if r.top_reactions:
            top_rx = ', '.join(
                f"{rx.get('term', rx) if isinstance(rx, dict) else rx} "
                f"({rx.get('count', '') if isinstance(rx, dict) else ''})"
                for rx in r.top_reactions[:5])
            kv.append(('Top Reactions', top_rx))
        if r.seriousness_stats:
            parts = [f'{k.replace("_", " ").title()}: {v}'
                     for k, v in r.seriousness_stats.items() if v]
            if parts:
                kv.append(('Serious Outcomes', ', '.join(parts[:4])))
        e.append(_kv_table(kv, styles))
        e.append(Spacer(1, 6))
    return e


def _build_evidence_chain_visual(styles, data: dict):
    """Evidence chain with source strength visualization."""
    prov = data.get('provenance', {}) or {}
    evidence = data.get('interaction_evidence', {}) or {}
    evidence_sources = prov.get('evidence_source', '')

    e = _section_line(styles, 'Evidence Chain &amp; Source Analysis')
    e.append(Paragraph(
        'Traceability of data sources and their relative strengths in the prediction pipeline.',
        styles['body_small']))

    # Draw evidence bars
    chart = _draw_evidence_bars(evidence_sources)
    if chart:
        e.append(chart)
        e.append(Spacer(1, 6))

    # Full provenance KV
    if prov:
        kv = []
        for key, label in [('prediction_path', 'Prediction Path'),
                           ('model_used', 'Model'), ('calibration_method', 'Calibration'),
                           ('known_interaction_detected', 'Known Interaction'),
                           ('known_interaction_source', 'KG Source'),
                           ('model_estimate_score', 'Raw Model Score'),
                           ('evidence_prior_score', 'Evidence Prior'),
                           ('uncertainty_fusion_cap', 'Uncertainty Cap')]:
            v = prov.get(key)
            if v is not None and v != '' and v is not False:
                kv.append((label, f'{v:.4f}' if isinstance(v, float) else str(v)))
        if kv:
            e.append(_kv_table(kv, styles))

    # Evidence summary
    ev_summary = evidence.get('evidence_summary', {})
    if ev_summary and isinstance(ev_summary, dict):
        e.append(Spacer(1, 4))
        e.append(Paragraph('Evidence Summary', styles['subsection']))
        sum_kv = []
        for k in ['total_items', 'supporting_items', 'weighted_support_score',
                   'confidence_band', 'unique_sources']:
            v = ev_summary.get(k)
            if v is not None:
                if isinstance(v, list):
                    v = ', '.join(str(x) for x in v)
                elif isinstance(v, float):
                    v = f'{v:.3f}'
                sum_kv.append((k.replace('_', ' ').title(), str(v)))
        if sum_kv:
            e.append(_kv_table(sum_kv, styles))

    return e


def _build_gnn_section(styles, data: dict):
    """GNN model analysis section."""
    explanation = data.get('explanation', {}) or {}
    prov = data.get('provenance', {}) or {}
    raw = prov.get('model_estimate_score') or data.get('raw_score')
    cal = prov.get('evidence_prior_score') or data.get('calibrated_score')

    if not explanation and raw is None and cal is None:
        return []

    e = _section_line(styles, 'GNN Model Analysis')
    e.append(Paragraph(
        'Graph Neural Network (MacroscopicDDI-GraphSAGE) model metadata and score decomposition.',
        styles['body_small']))

    kv = []
    if explanation:
        if explanation.get('model_version'):
            kv.append(('Model Version', explanation['model_version']))
        if explanation.get('data_source'):
            kv.append(('Data Source', explanation['data_source']))
        cal_info = explanation.get('calibration', {})
        if cal_info:
            kv.append(('Calibration', f"{cal_info.get('method', '')} v{cal_info.get('version', '')}"))
    if raw is not None:
        kv.append(('Raw Model Score', f'{_safe_float(raw):.4f}'))
    if cal is not None:
        kv.append(('Calibrated Score', f'{_safe_float(cal):.4f}'))
    cap = prov.get('uncertainty_fusion_cap')
    if cap is not None:
        kv.append(('Uncertainty Cap', f'{_safe_float(cap):.4f}'))
    if kv:
        e.append(_kv_table(kv, styles))
    return e


# ═══════════════════════════════════════════════════════════════════════════
# GEMINI LLM NARRATIVES
# ═══════════════════════════════════════════════════════════════════════════

def _generate_llm_narratives(data: dict, report_type: str, drug_names: List[str]) -> Dict[str, str]:
    try:
        from .gemini_client import get_gemini_client
        client = get_gemini_client()
        if not client.is_available():
            return {}
    except Exception:
        return {}

    results = {}
    drug_str = ', '.join(str(d) for d in drug_names)
    risk = data.get('risk_score') or data.get('regimen_risk_score', 0)
    sev = data.get('severity') or data.get('overall_risk_level', '?')
    mech = data.get('mechanism_hypothesis', 'Not available')

    # Executive Summary
    prompt = (
        f"Write a concise executive summary (150-200 words) for a clinical drug interaction report.\n"
        f"Drugs: {drug_str}\nRisk Score: {risk}\nSeverity: {sev}\nMechanism: {mech}\n")
    if report_type == 'poly':
        prompt += f"Total interactions: {len(data.get('interactions', []))}\n"
        prompt += f"Hub drug: {data.get('hub_drug', 'N/A')}\n"
    prompt += "Write professionally. Paragraph form, no headers or bullets."
    try:
        resp = client.generate(prompt, f"Drug interaction report for: {drug_str}")
        if resp and resp.text and 'not available' not in resp.text.lower():
            results['executive_summary'] = resp.text
    except Exception:
        pass

    # Clinical Assessment
    prompt2 = (
        f"Provide a clinical assessment (200-300 words) for: {drug_str}\n"
        f"Risk: {risk}, Severity: {sev}, Mechanism: {mech}\n"
        f"Cover: pharmacokinetic implications, pharmacodynamic considerations, "
        f"patient populations at risk, monitoring recommendations, relevant literature. "
        f"Professional clinical prose.")
    try:
        resp = client.generate(prompt2, f"Clinical assessment for: {drug_str}")
        if resp and resp.text and 'not available' not in resp.text.lower():
            results['clinical_assessment'] = resp.text
    except Exception:
        pass

    return results


# ═══════════════════════════════════════════════════════════════════════════
# PUBLIC API — STANDARD REPORT
# ═══════════════════════════════════════════════════════════════════════════

def generate_standard_report(prediction_data: dict, report_type: str = 'pair') -> Tuple[bytes, str]:
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=letter,
                            topMargin=0.6*inch, bottomMargin=0.5*inch,
                            leftMargin=0.7*inch, rightMargin=0.7*inch)
    styles = _build_styles()
    el = []

    if report_type == 'poly':
        drug_names = prediction_data.get('drugs', prediction_data.get('drug_names', []))
    else:
        drug_names = prediction_data.get('drug_names', [
            prediction_data.get('drug_a', 'Unknown'),
            prediction_data.get('drug_b', 'Unknown')])

    title = 'Polypharmacy Analysis Report' if report_type == 'poly' else 'Drug Interaction Report'
    el.extend(_header(styles, title, drug_names, 'standard'))
    el.extend(_build_risk_assessment(styles, prediction_data))
    el.extend(_build_mechanism(styles, prediction_data.get('mechanism_hypothesis', '')))
    el.extend(_build_affected_systems(styles, prediction_data.get('affected_systems', [])))

    if report_type == 'poly':
        el.extend(_build_hub_drug(styles, prediction_data.get('hub_drug', ''),
                                   prediction_data.get('hub_interaction_count', 0)))
        el.extend(_build_interactions_table(styles, prediction_data.get('interactions', [])))
        bm = prediction_data.get('body_map', {})
        if bm:
            el.extend(_section_line(styles, 'Affected Body Systems'))
            rows = [[s.replace('_', ' ').title(),
                     f'{_safe_float(v):.2f}' if not isinstance(v, dict) else f'{_safe_float(v.get("severity", 0)):.2f}']
                    for s, v in sorted(bm.items())]
            el.append(_data_table(['System', 'Score'], rows, styles, [3.25*inch, 3.25*inch]))

    el.extend(_build_provenance(styles, prediction_data.get('provenance', {})))
    el.extend(_build_class_warnings(styles, prediction_data.get('class_warnings', [])))
    el.extend(_build_conclusion(styles, prediction_data, report_type))
    el.extend(_disclaimer(styles))

    doc.build(el, onFirstPage=_page_footer, onLaterPages=_page_footer)
    return buf.getvalue(), build_filename(drug_names, report_type, 'standard')


# ═══════════════════════════════════════════════════════════════════════════
# PUBLIC API — ADVANCED (SUPER) REPORT
# ═══════════════════════════════════════════════════════════════════════════

def generate_advanced_report(prediction_data: dict, report_type: str = 'pair') -> Tuple[bytes, str]:
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=letter,
                            topMargin=0.55*inch, bottomMargin=0.5*inch,
                            leftMargin=0.65*inch, rightMargin=0.65*inch)
    styles = _build_styles()
    el = []

    if report_type == 'poly':
        drug_names = prediction_data.get('drugs', prediction_data.get('drug_names', []))
    else:
        drug_names = prediction_data.get('drug_names', [
            prediction_data.get('drug_a', 'Unknown'),
            prediction_data.get('drug_b', 'Unknown')])

    title = 'Polypharmacy Analysis Report' if report_type == 'poly' else 'Drug Interaction Report'

    # ─── Header ───
    el.extend(_header(styles, title, drug_names, 'advanced'))

    # ─── Table of Contents ───
    toc_sections = ['Executive Summary', 'Risk Assessment', 'Drug Pharmacology Profiles',
                    'CYP450 Enzyme Metabolism', 'Organ System Impact Analysis']
    if report_type == 'poly':
        toc_sections.extend(['Hub Drug Analysis', 'Pairwise Interaction Analysis',
                             'Polypharmacy Risk Factor Breakdown', 'Regimen Sensitivity Analysis'])
    toc_sections.extend(['FAERS Real-World Evidence', 'Food & Herbal Warnings',
                         'Evidence Chain & Source Analysis', 'GNN Model Analysis',
                         'Clinical Assessment', 'Conclusion'])
    el.extend(_build_toc(styles, toc_sections))

    # ─── LLM Narratives ───
    llm = _generate_llm_narratives(prediction_data, report_type, drug_names)
    if llm.get('executive_summary'):
        el.extend(_build_llm_section(styles, 'Executive Summary', llm['executive_summary']))
    else:
        el.extend(_section_line(styles, 'Executive Summary'))
        el.append(Paragraph(
            '<i>AI narrative unavailable — Gemini API not configured or request failed.</i>',
            styles['body_small']))

    # ─── Risk Assessment with Gauge ───
    el.extend(_build_risk_assessment(styles, prediction_data))
    risk_score = _safe_float(
        prediction_data.get('risk_score') or prediction_data.get('regimen_risk_score') or 0)
    gauge = _draw_risk_gauge(risk_score)
    el.append(gauge)
    el.append(Spacer(1, 8))

    # ─── Mechanism ───
    el.extend(_build_mechanism(styles, prediction_data.get('mechanism_hypothesis', '')))

    el.append(PageBreak())

    # ─── Drug Profiles ───
    el.extend(_build_drug_profiles(styles, drug_names))

    # ─── CYP450 Visual Heatmap ───
    cyp_section = _build_cyp450_visual(styles, drug_names)
    el.append(KeepTogether(cyp_section))

    # ─── Organ Impact with Bars ───
    body_map_data = prediction_data.get('body_map_data', {})
    organ_section = _build_body_map_visual(styles, body_map_data, prediction_data)
    el.append(KeepTogether(organ_section))

    # ─── Poly-specific sections ───
    if report_type == 'poly':
        el.append(PageBreak())

        hub_section = _build_hub_drug(styles, prediction_data.get('hub_drug', ''),
                                       prediction_data.get('hub_interaction_count', 0))
        el.extend(hub_section)
        el.extend(_build_interactions_table(styles, prediction_data.get('interactions', [])))

        # Interaction Heatmap
        interactions = prediction_data.get('interactions', [])
        if interactions and len(drug_names) >= 2:
            heatmap_section = _section_line(styles, 'Interaction Risk Heatmap')
            heatmap_section.append(Paragraph(
                'Visual matrix showing pairwise risk scores between all drugs. '
                'Color intensity corresponds to risk level.',
                styles['body_small']))
            heatmap = _draw_interaction_heatmap(drug_names, interactions)
            if heatmap:
                heatmap_section.append(heatmap)
            heatmap_section.append(Spacer(1, 8))
            el.append(KeepTogether(heatmap_section))

        # Factor breakdown
        factor_section = _build_polypharmacy_factors_visual(styles, prediction_data)
        el.append(KeepTogether(factor_section))

        # Regimen Sensitivity
        sensitivity_section = _build_regimen_sensitivity(styles, prediction_data)
        el.append(KeepTogether(sensitivity_section))

    el.append(PageBreak())

    # ─── FAERS ───
    faers_section = _build_faers_section(styles, drug_names, report_type)
    el.append(KeepTogether(faers_section))

    # ─── Food & Herbal ───
    food_section = _build_food_herbal(styles, drug_names)
    el.append(KeepTogether(food_section))

    # ─── Evidence Chain ───
    evidence_section = _build_evidence_chain_visual(styles, prediction_data)
    el.append(KeepTogether(evidence_section))

    # ─── GNN Model Analysis ───
    gnn_section = _build_gnn_section(styles, prediction_data)
    el.append(KeepTogether(gnn_section))

    # ─── Clinical Assessment (LLM) ───
    if llm.get('clinical_assessment'):
        el.extend(_build_llm_section(styles, 'Clinical Assessment', llm['clinical_assessment']))

    # ─── Class Warnings ───
    el.extend(_build_class_warnings(styles, prediction_data.get('class_warnings', [])))

    # ─── Provenance ───
    el.extend(_build_provenance(styles, prediction_data.get('provenance', {})))

    # ─── Conclusion ───
    el.extend(_build_conclusion(styles, prediction_data, report_type))

    # ─── Disclaimer ───
    el.extend(_disclaimer(styles))

    doc.build(el, onFirstPage=_page_footer, onLaterPages=_page_footer)
    return buf.getvalue(), build_filename(drug_names, report_type, 'advanced')


# ═══════════════════════════════════════════════════════════════════════════
# BACKWARD-COMPATIBLE ALIASES
# ═══════════════════════════════════════════════════════════════════════════

def generate_pair_report(prediction_data: dict) -> bytes:
    pdf_bytes, _ = generate_standard_report(prediction_data, 'pair')
    return pdf_bytes

def generate_poly_report(polypharmacy_data: dict) -> bytes:
    pdf_bytes, _ = generate_standard_report(polypharmacy_data, 'poly')
    return pdf_bytes
