"""
UI styles for Portfolio Analyzer.

Modern dark-mode design inspired by AI company aesthetics.
Slate backgrounds, blue-purple gradients, glass-morphism cards.
"""

import streamlit as st


# ── Color tokens (used by formula_card and importable by app.py) ──────────
BG_PRIMARY   = "#0f172a"   # slate-900
BG_SURFACE   = "#1e293b"   # slate-800
BG_CARD      = "rgba(30,41,59,0.65)"  # glass card
BORDER       = "rgba(148,163,184,0.12)"
ACCENT       = "#818cf8"   # indigo-400
ACCENT_HOVER = "#6366f1"   # indigo-500
GRADIENT     = "linear-gradient(135deg, #6366f1 0%, #a855f7 100%)"
TEXT_PRIMARY = "#f1f5f9"   # slate-100
TEXT_SECONDARY = "#94a3b8" # slate-400
TEXT_MUTED   = "#64748b"   # slate-500
GREEN        = "#34d399"   # emerald-400
RED          = "#f87171"   # red-400


# ── Formula hover-card ────────────────────────────────────────────────────
_fc_counter = 0


def formula_card(label: str, value: str, formula_html: str, delta: str = "") -> str:
    """
    Self-contained HTML metric card with a hover formula overlay.
    Each card gets a unique ID + its own <style> block.
    """
    global _fc_counter
    _fc_counter += 1
    uid = f"fc{_fc_counter}"

    delta_block = ""
    if delta:
        color = GREEN if not delta.lstrip().startswith("-") else RED
        delta_block = (
            f'<p style="font-size:13px;margin:4px 0 0 0;color:{color} !important;font-weight:500;">'
            f"{delta}</p>"
        )

    return f"""
<style>
#{uid} {{
    position: relative;
    background: {BG_CARD};
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    padding: 22px 24px;
    border-radius: 12px;
    border: 1px solid {BORDER};
    cursor: default;
    transition: box-shadow .25s ease, border-color .25s ease;
}}
#{uid}:hover {{
    box-shadow: 0 0 24px rgba(99,102,241,0.15);
    border-color: rgba(129,140,248,0.35);
}}
#{uid} .fc-tip {{
    display: none;
    position: absolute;
    top: 0; left: 0; right: 0;
    z-index: 999999;
}}
#{uid}:hover .fc-tip {{
    display: block;
}}
</style>
<div id="{uid}">
  <p style="font-size:12px;font-weight:600;color:{TEXT_SECONDARY} !important;margin:0 0 6px 0;
            text-transform:uppercase;letter-spacing:.5px;
            font-family:'Inter',system-ui,-apple-system,sans-serif;">
    {label}
  </p>
  <p style="font-size:28px;font-weight:700;color:{TEXT_PRIMARY} !important;margin:0;line-height:1.2;
            font-family:'Inter',system-ui,-apple-system,sans-serif;">
    {value}
  </p>
  {delta_block}

  <div class="fc-tip">
    <div style="background:rgba(15,23,42,0.97);color:#e0e7ff;border-radius:12px;
                padding:20px 22px;box-shadow:0 16px 48px rgba(0,0,0,0.45);
                border:1px solid rgba(129,140,248,0.2);
                font-size:13px;line-height:1.7;min-width:260px;
                backdrop-filter:blur(16px);-webkit-backdrop-filter:blur(16px);
                font-family:'Inter',system-ui,-apple-system,sans-serif;">
      <p style="font-size:11px;font-weight:700;text-transform:uppercase;
                letter-spacing:.8px;color:#818cf8 !important;margin:0 0 12px 0;">
        {label}
      </p>
      {formula_html}
    </div>
  </div>
</div>
"""


# ── Global stylesheet ─────────────────────────────────────────────────────
def apply_global_styles() -> None:
    """Inject the global CSS for the dark AI-company aesthetic."""
    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

        /* ── Dark backdrop ── */
        .main, .stApp {{
            background-color: {BG_PRIMARY} !important;
        }}

        /* Let formula tooltips escape column/container boundaries */
        [data-testid="column"],
        [data-testid="stVerticalBlock"],
        [data-testid="stHorizontalBlock"],
        .stColumn,
        .element-container {{
            overflow: visible !important;
        }}

        .block-container {{
            padding-top: 2rem;
            padding-bottom: 2rem;
            max-width: 1400px;
        }}

        /* ── Metrics (glass cards) ── */
        [data-testid="stMetric"],
        .stMetric {{
            background: {BG_CARD} !important;
            backdrop-filter: blur(12px);
            -webkit-backdrop-filter: blur(12px);
            padding: 24px;
            border-radius: 12px;
            border: 1px solid {BORDER} !important;
            transition: all 0.25s ease;
        }}
        [data-testid="stMetric"]:hover,
        .stMetric:hover {{
            border-color: rgba(129,140,248,0.3) !important;
            box-shadow: 0 0 20px rgba(99,102,241,0.1);
        }}
        [data-testid="stMetricLabel"],
        [data-testid="stMetricLabel"] label,
        [data-testid="stMetricLabel"] p {{
            color: {TEXT_SECONDARY} !important;
            font-family: 'Inter', system-ui, sans-serif !important;
            font-weight: 500 !important;
            font-size: 13px !important;
        }}
        [data-testid="stMetricValue"] {{
            color: {TEXT_PRIMARY} !important;
            font-family: 'Inter', system-ui, sans-serif !important;
            font-weight: 700 !important;
        }}
        [data-testid="stMetricDelta"] {{
            font-family: 'Inter', system-ui, sans-serif !important;
        }}

        /* ── Typography ── */
        h1 {{
            color: {TEXT_PRIMARY} !important;
            font-weight: 700;
            letter-spacing: -0.8px;
            font-size: 2.5rem;
            font-family: 'Inter', system-ui, sans-serif;
        }}
        h2 {{
            color: {TEXT_PRIMARY} !important;
            font-weight: 600;
            font-size: 1.75rem;
            margin-top: 2rem;
            font-family: 'Inter', system-ui, sans-serif;
        }}
        h3, h4 {{
            color: {TEXT_PRIMARY} !important;
            font-weight: 500;
            font-family: 'Inter', system-ui, sans-serif;
        }}
        p, span, div, label {{
            color: {TEXT_SECONDARY} !important;
            font-family: 'Inter', system-ui, sans-serif;
        }}

        /* ── Force light text on ALL Streamlit internals ── */
        .stMarkdown, .stMarkdown p, .stMarkdown span, .stMarkdown div,
        .stText, .stText p,
        [data-testid="stText"],
        [data-testid="stMarkdownContainer"],
        [data-testid="stMarkdownContainer"] p,
        [data-testid="stMarkdownContainer"] span,
        [data-testid="stMarkdownContainer"] div,
        [data-testid="stMarkdownContainer"] li,
        [data-testid="stMarkdownContainer"] strong,
        [data-testid="stCaptionContainer"],
        [data-testid="stCaptionContainer"] p,
        .element-container, .element-container p, .element-container span {{
            color: {TEXT_SECONDARY} !important;
        }}
        strong, b {{
            color: {TEXT_PRIMARY} !important;
        }}
        [data-testid="stMetricValue"],
        [data-testid="stMetricValue"] div {{
            color: {TEXT_PRIMARY} !important;
        }}
        [data-testid="stMetricDelta"] svg {{
            fill: currentColor;
        }}

        /* info / success / warning / error boxes */
        .stAlert p, .stAlert span, .stAlert div {{
            color: {TEXT_SECONDARY} !important;
        }}

        /* file uploader text */
        [data-testid="stFileUploader"] p,
        [data-testid="stFileUploader"] span,
        [data-testid="stFileUploader"] div,
        [data-testid="stFileUploader"] label {{
            color: {TEXT_SECONDARY} !important;
        }}

        /* data editor / table text */
        .stDataFrame th, .stDataFrame td {{
            color: {TEXT_SECONDARY} !important;
        }}

        /* column config / header text in data editor */
        [data-testid="stDataFrameResizable"] {{
            color: {TEXT_SECONDARY} !important;
        }}

        /* tooltip / help icons */
        [data-testid="stTooltipIcon"] svg {{
            fill: {TEXT_MUTED} !important;
        }}

        /* selectbox / dropdown text */
        .stSelectbox label, .stSelectbox p, .stSelectbox span,
        .stMultiSelect label, .stMultiSelect p, .stMultiSelect span {{
            color: {TEXT_SECONDARY} !important;
        }}

        /* checkbox / radio label text */
        .stCheckbox label span,
        .stRadio label span {{
            color: {TEXT_SECONDARY} !important;
        }}

        /* ── Tabs ── */
        .stTabs [data-baseweb="tab-list"] {{
            gap: 4px;
            background: {BG_SURFACE};
            padding: 6px;
            border-radius: 10px;
            border: 1px solid {BORDER};
        }}
        .stTabs [data-baseweb="tab"] {{
            padding: 10px 20px;
            background: transparent;
            border-radius: 8px;
            color: {TEXT_MUTED} !important;
            font-weight: 500;
            font-size: 14px;
        }}
        .stTabs [aria-selected="true"] {{
            background: rgba(99,102,241,0.15) !important;
            color: {ACCENT} !important;
            font-weight: 600;
            border-bottom: 2px solid {ACCENT};
        }}

        /* ── Buttons (gradient accent) ── */
        .stButton > button {{
            background: {GRADIENT} !important;
            color: white !important;
            border: none;
            border-radius: 8px;
            padding: 12px 28px;
            font-weight: 600;
            font-size: 15px;
            box-shadow: 0 2px 12px rgba(99,102,241,0.3);
            transition: all 0.25s ease;
            font-family: 'Inter', system-ui, sans-serif;
        }}
        .stButton > button:hover {{
            box-shadow: 0 4px 20px rgba(99,102,241,0.45);
            transform: translateY(-1px);
        }}

        /* ── Data tables ── */
        .stDataFrame {{
            background: {BG_CARD} !important;
            border-radius: 10px;
            border: 1px solid {BORDER};
        }}

        /* ── Alerts ── */
        .stAlert {{
            background: {BG_SURFACE} !important;
            border-radius: 10px;
            border: 1px solid {BORDER};
        }}

        /* ── Inputs ── */
        .stSelectbox > div > div,
        .stTextInput > div > div,
        input {{
            background: {BG_SURFACE} !important;
            border-radius: 8px;
            border: 1px solid {BORDER} !important;
            color: {TEXT_PRIMARY} !important;
        }}

        /* ── Radio as pill buttons ── */
        div[data-testid="stRadio"] > div[role="radiogroup"] {{
            display: flex;
            flex-direction: row;
            gap: 0;
            background: {BG_SURFACE};
            padding: 4px;
            border-radius: 10px;
            border: 1px solid {BORDER};
            width: fit-content;
        }}
        div[data-testid="stRadio"] > div[role="radiogroup"] > label {{
            background: transparent;
            padding: 8px 16px;
            border-radius: 8px;
            color: {TEXT_MUTED} !important;
            font-weight: 500;
            font-size: 13px;
            cursor: pointer;
            transition: all 0.15s ease;
            margin: 0 2px;
            border: none;
            white-space: nowrap;
        }}
        div[data-testid="stRadio"] > div[role="radiogroup"] > label:hover {{
            background: rgba(99,102,241,0.1);
            color: {TEXT_PRIMARY} !important;
        }}
        div[data-testid="stRadio"] > div[role="radiogroup"] > label[data-checked="true"],
        div[data-testid="stRadio"] > div[role="radiogroup"] > label:has(input:checked) {{
            background: {ACCENT} !important;
            color: white !important;
            box-shadow: 0 2px 8px rgba(99,102,241,0.35);
        }}
        div[data-testid="stRadio"] input[type="radio"] {{
            position: absolute;
            opacity: 0;
            width: 0;
            height: 0;
        }}

        /* ── Dividers ── */
        hr {{
            border: none;
            border-top: 1px solid {BORDER};
            margin: 2rem 0;
        }}

        /* ── Streamlit chrome overrides ── */
        header[data-testid="stHeader"] {{
            background: {BG_PRIMARY} !important;
        }}
        .stDeployButton {{
            color: {TEXT_SECONDARY} !important;
        }}

        /* file uploader */
        [data-testid="stFileUploader"] {{
            background: {BG_SURFACE} !important;
            border-radius: 10px;
            border: 1px solid {BORDER} !important;
        }}

        /* expanders */
        .streamlit-expanderHeader {{
            color: {TEXT_PRIMARY} !important;
            background: {BG_SURFACE} !important;
        }}
        .streamlit-expanderContent {{
            background: {BG_SURFACE} !important;
        }}

        /* toast / spinner */
        .stSpinner > div {{
            border-top-color: {ACCENT} !important;
        }}

        /* scrollbar */
        ::-webkit-scrollbar {{
            width: 6px;
        }}
        ::-webkit-scrollbar-track {{
            background: {BG_PRIMARY};
        }}
        ::-webkit-scrollbar-thumb {{
            background: {TEXT_MUTED};
            border-radius: 3px;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )
