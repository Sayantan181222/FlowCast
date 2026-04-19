"""
Reusable Streamlit UI Components.

Provides common UI patterns used across dashboard pages.
"""

import streamlit as st


def render_header(title, icon="🚀", subtitle=None):
    """Render a page header with icon and optional subtitle."""
    st.markdown(f"# {icon} {title}")
    if subtitle:
        st.caption(subtitle)
    st.markdown("---")


def render_metric_row(metrics):
    """
    Render a row of metric cards.

    Args:
        metrics: List of (icon, value, label) tuples.
    """
    from ui.styles import metric_card
    cols = st.columns(len(metrics))
    for col, (icon, value, label) in zip(cols, metrics):
        with col:
            st.markdown(metric_card(icon, value, label), unsafe_allow_html=True)


def render_status(label, is_ready):
    """Render a status indicator."""
    from ui.styles import status_badge
    status = "success" if is_ready else "warning"
    text = f"{label}: {'Ready' if is_ready else 'Not Ready'}"
    st.markdown(status_badge(text, status), unsafe_allow_html=True)


def render_info_box(title, content, icon="ℹ️"):
    """Render an info callout box."""
    st.info(f"{icon} **{title}**\n\n{content}")


def render_error_box(title, content):
    """Render an error callout box."""
    st.error(f"❌ **{title}**\n\n{content}")


def render_success_box(title, content):
    """Render a success callout box."""
    st.success(f"✅ **{title}**\n\n{content}")
