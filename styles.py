"""
UI styles for Portfolio Analyzer.

Centralizes all custom CSS so Streamlit layout and colors stay
consistent and easy to tweak.
"""

import streamlit as st


_fc_counter = 0


def formula_card(label: str, value: str, formula_html: str, delta: str = "") -> str:
    """
    Return a **fully self-contained** HTML metric card with hover overlay.

    Every card gets a unique ID so its embedded ``<style>`` block only
    affects itself.  All visual properties use inline styles so nothing
    depends on the global stylesheet.

    Args:
        label:        metric title  (e.g. "Portfolio Beta")
        value:        display value (e.g. "1.24")
        formula_html: rich HTML rows for the formula breakdown
        delta:        optional delta string shown below the value
    """
    global _fc_counter
    _fc_counter += 1
    uid = f"fc{_fc_counter}"

    delta_block = ""
    if delta:
        color = "#22c55e" if not delta.lstrip().startswith("-") else "#ef4444"
        delta_block = (
            f'<p style="font-size:13px;margin:4px 0 0 0;color:{color};font-weight:500;">'
            f"{delta}</p>"
        )

    return f"""
<style>
#{uid} {{
    position: relative;
    background: #ffffff;
    padding: 22px 24px;
    border-radius: 10px;
    border: 1px solid #f0f0f0;
    box-shadow: 0 1px 3px rgba(0,0,0,0.08);
    cursor: default;
    transition: box-shadow .2s ease, border-color .2s ease;
}}
#{uid}:hover {{
    box-shadow: 0 4px 16px rgba(109,40,217,0.15);
    border-color: #d8b4fe;
}}
#{uid} .fc-tip {{
    display: none;
    position: absolute;
    top: 0; left: 0; right: 0;
    z-index: 9999;
}}
#{uid}:hover .fc-tip {{
    display: block;
}}
</style>
<div id="{uid}">
  <p style="font-size:13px;font-weight:600;color:#6b7280;margin:0 0 6px 0;
            text-transform:uppercase;letter-spacing:.3px;
            font-family:system-ui,-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;">
    {label}
  </p>
  <p style="font-size:28px;font-weight:700;color:#111827;margin:0;line-height:1.2;
            font-family:system-ui,-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;">
    {value}
  </p>
  {delta_block}

  <div class="fc-tip">
    <div style="background:#1e1b4b;color:#e0e7ff;border-radius:12px;
                padding:20px 22px;box-shadow:0 12px 40px rgba(0,0,0,0.3);
                font-size:13px;line-height:1.7;min-width:260px;
                font-family:system-ui,-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;">
      <p style="font-size:11px;font-weight:700;text-transform:uppercase;
                letter-spacing:.8px;color:#a5b4fc;margin:0 0 12px 0;">
        {label}
      </p>
      {formula_html}
    </div>
  </div>
</div>
"""


def apply_global_styles() -> None:
    """
    Inject global CSS for:
    - Pure white background
    - Purple accents (#6D28D9) for primary actions
    - Clean cards with minimal shadows
    - Sans-serif typography and high whitespace
    - Wide layout visuals
    """
    st.markdown(
        """
        <style>
        /* Pure white backdrop */
        .main {
            background-color: #ffffff;
        }

        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
            max-width: 1400px;
        }

        /* Metrics styling - minimal shadows */
        .stMetric {
            background: #ffffff;
            padding: 24px;
            border-radius: 8px;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.10);
            border: 1px solid #f0f0f0;
            transition: all 0.2s ease;
        }

        .stMetric:hover {
            box-shadow: 0 2px 6px rgba(0, 0, 0, 0.12);
        }

        /* Clean typography with strong contrast */
        h1 {
            color: #000000;
            font-weight: 600;
            letter-spacing: -0.8px;
            font-size: 2.5rem;
            font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI",
                         sans-serif;
        }

        h2 {
            color: #000000;
            font-weight: 600;
            font-size: 1.75rem;
            margin-top: 2rem;
            font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI",
                         sans-serif;
        }

        h3, h4 {
            color: #1a1a1a;
            font-weight: 500;
            font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI",
                         sans-serif;
        }

        /* All body text */
        p, span, div, label {
            color: #333333;
            font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI",
                         sans-serif;
        }

        /* Minimal tabs */
        .stTabs [data-baseweb="tab-list"] {
            gap: 4px;
            background: #ffffff;
            padding: 8px;
            border-radius: 8px;
            border-bottom: 1px solid #f0f0f0;
        }

        .stTabs [data-baseweb="tab"] {
            padding: 10px 20px;
            background: transparent;
            border-radius: 6px;
            color: #4a4a4a;
            font-weight: 500;
            font-size: 15px;
        }

        .stTabs [aria-selected="true"] {
            background-color: #f8f9fa;
            color: #000000;
            font-weight: 600;
            border-bottom: 2px solid #6D28D9;
        }

        /* Primary buttons - solid purple accent */
        .stButton > button {
            background-color: #6D28D9;
            color: white;
            border: none;
            border-radius: 6px;
            padding: 12px 28px;
            font-weight: 500;
            font-size: 15px;
            box-shadow: 0 1px 3px rgba(109, 40, 217, 0.25);
            transition: all 0.2s ease;
        }

        .stButton > button:hover {
            background-color: #5B21B6;
            box-shadow: 0 2px 6px rgba(109, 40, 217, 0.35);
        }

        /* Data Editor / Tables as white cards */
        .stDataFrame {
            background: #ffffff;
            border-radius: 8px;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.10);
            border: 1px solid #f0f0f0;
        }

        /* Clean alert boxes */
        .stAlert {
            background: #ffffff;
            border-radius: 8px;
            border: 1px solid #f0f0f0;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.10);
        }

        /* Input fields */
        .stSelectbox > div > div,
        .stTextInput > div > div,
        input {
            background: #ffffff;
            border-radius: 6px;
            border: 1px solid #e0e0e0;
        }

        /* Time period radio buttons styled as sleek buttons */
        div[data-testid="stRadio"] > div[role="radiogroup"] {
            display: flex;
            flex-direction: row;
            gap: 0;
            background: #f8f9fa;
            padding: 4px;
            border-radius: 8px;
            border: 1px solid #e5e7eb;
            width: fit-content;
        }

        div[data-testid="stRadio"] > div[role="radiogroup"] > label {
            background: transparent;
            padding: 8px 16px;
            border-radius: 6px;
            color: #4b5563 !important;
            font-weight: 500;
            font-size: 13px;
            cursor: pointer;
            transition: all 0.15s ease;
            margin: 0 2px;
            border: none;
            white-space: nowrap;
        }

        div[data-testid="stRadio"] > div[role="radiogroup"] > label:hover {
            background: #e5e7eb;
            color: #1f2937 !important;
        }

        div[data-testid="stRadio"] > div[role="radiogroup"] > label[data-checked="true"],
        div[data-testid="stRadio"] > div[role="radiogroup"] > label:has(input:checked) {
            background: #6D28D9 !important;
            color: white !important;
            box-shadow: 0 1px 3px rgba(109, 40, 217, 0.35);
        }

        /* Hide the radio circles */
        div[data-testid="stRadio"] input[type="radio"] {
            position: absolute;
            opacity: 0;
            width: 0;
            height: 0;
        }

        /* Dividers */
        hr {
            border: none;
            border-top: 1px solid #f0f0f0;
            margin: 2rem 0;
        }

        /* Remove default Streamlit branding colors */
        .stApp {
            background-color: #ffffff;
        }

        /* Formula cards use self-contained inline styles + scoped IDs */
        </style>
        """,
        unsafe_allow_html=True,
    )

