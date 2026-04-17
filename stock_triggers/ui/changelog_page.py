from __future__ import annotations

from pathlib import Path

import streamlit as st


_CHANGELOG_PATH = Path(__file__).resolve().parent.parent / "docs" / "CHANGELOG.md"


def handle_changelog_query_param() -> None:
    """Call once at top of app — intercepts ?page=changelog links."""
    page = str(st.query_params.get("page", "") or "").strip().lower()
    if page != "changelog":
        return
    params = dict(st.query_params)
    params.pop("page", None)
    st.query_params.from_dict(params)
    st.session_state["mode"] = "Release History"
    st.rerun()


def render_changelog_page() -> None:
    st.markdown(
        (
            "<div class='hero'>"
            "<div class='hero-title'>Release History</div>"
            "<div class='hero-sub'>"
            "Full release history from repo inception. What's New stays short and recent; this page keeps the longer timeline in one place."
            "</div>"
            "</div>"
        ),
        unsafe_allow_html=True,
    )

    try:
        changelog_text = _CHANGELOG_PATH.read_text(encoding="utf-8").strip()
    except OSError:
        st.info("The complete release history file is not available yet.")
        return

    if not changelog_text:
        st.info("The complete release history is empty right now.")
        return

    st.markdown(changelog_text)