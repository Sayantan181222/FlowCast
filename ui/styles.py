"""
Custom CSS Styles for the FlowCast Streamlit Dashboard.

Implements a clean theme with readable dark text, gradient accents,
glassmorphism card effects, and smooth hover transitions.
"""


def get_main_css():
    """Return the main CSS stylesheet for the dashboard."""
    return """
    <style>
    /* ========== Google Fonts ========== */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    /* ========== Global Styles ========== */
    .stApp {
        font-family: 'Inter', sans-serif;
    }

    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1200px;
    }

    /* ========== Sidebar Styles ========== */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #f5f3ff 0%, #ede9fe 60%, #f5f3ff 100%);
        border-right: 2px solid rgba(99, 102, 241, 0.2);
    }

    [data-testid="stSidebar"] .stMarkdown h1,
    [data-testid="stSidebar"] .stMarkdown h2,
    [data-testid="stSidebar"] .stMarkdown h3 {
        color: #4f46e5 !important;
    }

    [data-testid="stSidebar"] .stMarkdown p,
    [data-testid="stSidebar"] .stMarkdown span {
        color: #475569 !important;
    }

    [data-testid="stSidebar"] label {
        color: #374151 !important;
        font-weight: 500;
    }

    [data-testid="stSidebar"] .stCaption {
        color: #64748b !important;
    }

    [data-testid="stSidebar"] hr {
        border-color: rgba(99, 102, 241, 0.25) !important;
    }

    /* ========== Metric Cards ========== */
    .metric-card {
        background: linear-gradient(135deg, #f0f0ff, #e8e6ff);
        border: 1px solid rgba(99, 102, 241, 0.25);
        border-radius: 16px;
        padding: 1.5rem;
        text-align: center;
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }

    .metric-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(99, 102, 241, 0.08), transparent);
        transition: left 0.5s ease;
    }

    .metric-card:hover::before {
        left: 100%;
    }

    .metric-card:hover {
        border-color: rgba(99, 102, 241, 0.5);
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(99, 102, 241, 0.15);
    }

    .metric-value {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(135deg, #4f46e5, #7c3aed);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0.5rem 0;
    }

    .metric-label {
        font-size: 0.85rem;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 1px;
        font-weight: 600;
    }

    .metric-icon {
        font-size: 1.5rem;
        margin-bottom: 0.3rem;
    }

    /* ========== Section Headers ========== */
    .section-header {
        background: linear-gradient(135deg, #f0f0ff, #e8e6ff);
        border-left: 4px solid #6366f1;
        padding: 1rem 1.5rem;
        border-radius: 0 12px 12px 0;
        margin: 1.5rem 0 1rem 0;
    }

    .section-header h2 {
        margin: 0;
        font-weight: 700;
        font-size: 1.3rem;
        color: #1e293b;
    }

    .section-header p {
        margin: 0.3rem 0 0 0;
        font-size: 0.85rem;
        color: #64748b;
    }

    /* ========== Glass Cards ========== */
    .glass-card {
        background: linear-gradient(135deg, #f5f3ff, #ede9fe);
        border: 1px solid rgba(99, 102, 241, 0.2);
        border-radius: 16px;
        padding: 1.5rem;
        margin-bottom: 1rem;
    }

    .glass-card h3, .glass-card h4 {
        color: #4f46e5 !important;
    }

    .glass-card p {
        color: #475569 !important;
    }

    /* ========== Status Badge ========== */
    .status-badge {
        display: inline-block;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    .status-success {
        background: rgba(34, 197, 94, 0.15);
        color: #16a34a;
        border: 1px solid rgba(34, 197, 94, 0.3);
    }

    .status-warning {
        background: rgba(234, 179, 8, 0.15);
        color: #ca8a04;
        border: 1px solid rgba(234, 179, 8, 0.3);
    }

    .status-error {
        background: rgba(239, 68, 68, 0.15);
        color: #dc2626;
        border: 1px solid rgba(239, 68, 68, 0.3);
    }

    .status-info {
        background: rgba(99, 102, 241, 0.15);
        color: #4f46e5;
        border: 1px solid rgba(99, 102, 241, 0.3);
    }

    /* ========== Algorithm Card ========== */
    .algo-card {
        background: linear-gradient(135deg, #f5f3ff, #f0f0ff);
        border: 1px solid rgba(99, 102, 241, 0.2);
        border-radius: 12px;
        padding: 1.2rem;
        margin-bottom: 0.8rem;
        transition: all 0.3s ease;
    }

    .algo-card:hover {
        border-color: rgba(99, 102, 241, 0.4);
        box-shadow: 0 4px 15px rgba(99, 102, 241, 0.1);
    }

    .algo-name {
        font-size: 1.1rem;
        font-weight: 700;
        color: #4f46e5;
        margin-bottom: 0.5rem;
    }

    .algo-metric {
        display: inline-block;
        margin-right: 1rem;
        font-size: 0.85rem;
        color: #475569;
    }

    .algo-metric strong {
        color: #1e293b;
    }

    /* ========== Streamlit Overrides ========== */
    .stButton > button {
        background: linear-gradient(135deg, #6366f1, #8b5cf6);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.6rem 1.5rem;
        font-weight: 600;
        letter-spacing: 0.3px;
        transition: all 0.3s ease;
    }

    .stButton > button:hover {
        background: linear-gradient(135deg, #4f46e5, #7c3aed);
        box-shadow: 0 4px 15px rgba(99, 102, 241, 0.3);
        transform: translateY(-1px);
    }

    /* Tab styles */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
        background: #f0f0ff;
        border-radius: 12px;
        padding: 4px;
    }

    .stTabs [data-baseweb="tab"] {
        border-radius: 10px;
        padding: 8px 20px;
        font-weight: 500;
        color: #475569;
    }

    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.2), rgba(139, 92, 246, 0.15));
        color: #4f46e5;
    }

    /* ========== Footer ========== */
    .footer {
        text-align: center;
        padding: 2rem 0 1rem 0;
        color: #94a3b8;
        font-size: 0.8rem;
        border-top: 1px solid rgba(99, 102, 241, 0.15);
        margin-top: 3rem;
    }

    /* ========== Animations ========== */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }

    .animate-in {
        animation: fadeIn 0.5s ease-out forwards;
    }
    </style>
    """


def metric_card(icon, value, label, color="#4f46e5"):
    """
    Generate HTML for a metric card.

    Args:
        icon: Emoji or icon character.
        value: Display value.
        label: Metric label text.
        color: Accent color (hex).

    Returns:
        HTML string.
    """
    return f"""
    <div class="metric-card">
        <div class="metric-icon">{icon}</div>
        <div class="metric-value" style="background: linear-gradient(135deg, {color}, {color}99);
            -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
            {value}
        </div>
        <div class="metric-label">{label}</div>
    </div>
    """


def section_header(title, subtitle=""):
    """Generate HTML for a section header."""
    sub_html = f"<p>{subtitle}</p>" if subtitle else ""
    return f"""
    <div class="section-header">
        <h2>{title}</h2>
        {sub_html}
    </div>
    """


def status_badge(text, status="info"):
    """Generate HTML for a status badge."""
    return f'<span class="status-badge status-{status}">{text}</span>'


def glass_card(content):
    """Wrap content in a glass card."""
    return f'<div class="glass-card">{content}</div>'
