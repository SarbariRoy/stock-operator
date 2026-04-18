from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
import html
from pathlib import Path
import re

import streamlit as st


_CHANGELOG_PATH = Path(__file__).resolve().parent.parent / "docs" / "CHANGELOG.md"
_DATE_HEADING_RE = re.compile(r"^##\s+(.+?)\s*$")
_RELEASE_TITLE_RE = re.compile(r"^###\s+(.+?)\s*$")
_ISO_DATE_RE = re.compile(r"(\d{4}-\d{2}-\d{2})")


@dataclass(frozen=True, slots=True)
class ChangelogMilestone:
        date_label: str
        title: str
        body_lines: list[str]
        summary: str
        start_date: date | None
        end_date: date | None


def _parse_date_span(label: str) -> tuple[date | None, date | None]:
        matches = _ISO_DATE_RE.findall(label)
        if not matches:
                return None, None

        parsed_dates: list[date] = []
        for value in matches[:2]:
                try:
                        parsed_dates.append(datetime.strptime(value, "%Y-%m-%d").date())
                except ValueError:
                        continue

        if not parsed_dates:
                return None, None
        if len(parsed_dates) == 1:
                return parsed_dates[0], parsed_dates[0]
        return parsed_dates[0], parsed_dates[1]


def _split_changelog_sections(text: str) -> tuple[list[str], list[tuple[str, list[str]]]]:
        preamble_lines: list[str] = []
        sections: list[tuple[str, list[str]]] = []
        current_heading: str | None = None
        current_lines: list[str] = []

        for line in text.splitlines():
                match = _DATE_HEADING_RE.match(line)
                if not match:
                        current_lines.append(line)
                        continue

                if current_heading is None:
                        preamble_lines = current_lines[:]
                else:
                        sections.append((current_heading, current_lines[:]))

                current_heading = match.group(1).strip()
                current_lines = []

        if current_heading is None:
                return current_lines, []

        sections.append((current_heading, current_lines[:]))
        return preamble_lines, sections


def _clean_summary_line(line: str) -> str:
        stripped = line.strip()
        stripped = re.sub(r"^[-*+]\s+", "", stripped)
        stripped = re.sub(r"^\d+\.\s+", "", stripped)
        return stripped.strip()


def _extract_summary(lines: list[str]) -> str:
        for line in lines:
                cleaned = _clean_summary_line(line)
                if cleaned:
                        return cleaned
        return ""


def _parse_changelog(text: str) -> tuple[list[str], list[ChangelogMilestone], list[tuple[str, list[str]]]]:
        preamble_lines, sections = _split_changelog_sections(text)
        milestones: list[ChangelogMilestone] = []
        supplemental_sections: list[tuple[str, list[str]]] = []

        for heading, content_lines in sections:
                if not _ISO_DATE_RE.search(heading):
                        supplemental_sections.append((heading, content_lines))
                        continue

                title_index: int | None = None
                title = ""
                for idx, line in enumerate(content_lines):
                        match = _RELEASE_TITLE_RE.match(line)
                        if match:
                                title_index = idx
                                title = match.group(1).strip()
                                break

                if title_index is None or not title:
                        continue

                body_lines = content_lines[:title_index] + content_lines[title_index + 1 :]
                start_date, end_date = _parse_date_span(heading)
                milestones.append(
                        ChangelogMilestone(
                                date_label=heading,
                                title=title,
                                body_lines=body_lines,
                                summary=_extract_summary(body_lines),
                                start_date=start_date,
                                end_date=end_date,
                        )
                )

        return preamble_lines, milestones, supplemental_sections


def _format_display_date(value: date | None) -> str:
        if value is None:
                return "Unknown"
        return value.strftime("%d %b %Y")


def _format_span_label(milestones: list[ChangelogMilestone]) -> str:
        dated = [item for item in milestones if item.start_date or item.end_date]
        if not dated:
                return "Milestone-based"

        earliest = min((item.start_date or item.end_date) for item in dated)
        latest = max((item.end_date or item.start_date) for item in dated)
        if earliest == latest:
                return _format_display_date(latest)
        return f"{_format_display_date(earliest)} to {_format_display_date(latest)}"


def _count_highlights(lines: list[str]) -> int:
        count = sum(1 for line in lines if line.strip().startswith("- "))
        return count or (1 if _extract_summary(lines) else 0)


def _apply_inline_formatting(text: str) -> str:
        escaped = html.escape(text, quote=False)
        escaped = re.sub(r"`([^`]+)`", r"<code>\1</code>", escaped)
        escaped = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", escaped)
        escaped = re.sub(r"\*([^*]+)\*", r"<em>\1</em>", escaped)
        return escaped


def _render_simple_markdown_html(lines: list[str]) -> str:
        blocks: list[str] = []
        paragraph_lines: list[str] = []
        bullet_items: list[str] = []

        def flush_paragraph() -> None:
                if not paragraph_lines:
                        return
                text = " ".join(part.strip() for part in paragraph_lines if part.strip())
                if text:
                        blocks.append(f"<p>{_apply_inline_formatting(text)}</p>")
                paragraph_lines.clear()

        def flush_bullets() -> None:
                if not bullet_items:
                        return
                items = "".join(f"<li>{_apply_inline_formatting(item)}</li>" for item in bullet_items)
                blocks.append(f"<ul>{items}</ul>")
                bullet_items.clear()

        for raw_line in lines:
                stripped = raw_line.strip()
                if not stripped:
                        flush_paragraph()
                        flush_bullets()
                        continue

                if stripped.startswith("- "):
                        flush_paragraph()
                        bullet_items.append(stripped[2:].strip())
                        continue

                flush_bullets()
                paragraph_lines.append(stripped)

        flush_paragraph()
        flush_bullets()
        return "".join(blocks)


def _milestone_option_label(milestone: ChangelogMilestone) -> str:
        return f"{milestone.date_label} - {milestone.title}"


def _render_release_history_styles() -> None:
        st.markdown(
                """
                <style>
                .release-shell {
                    border:1px solid rgba(251,191,36,0.34);
                    border-radius:24px;
                    background:linear-gradient(135deg,#fff7ed 0%,#fefce8 28%,#ecfeff 100%);
                    padding:1.1rem;
                    margin:0.2rem 0 1rem 0;
                    position:relative;
                    overflow:hidden;
                    box-shadow:0 18px 44px rgba(124,45,18,0.12), 0 10px 30px rgba(13,148,136,0.10);
                }
                .release-shell::before {
                    content:'';
                    position:absolute;
                    top:-84px;
                    left:-42px;
                    width:250px;
                    height:250px;
                    background:radial-gradient(circle, rgba(251,191,36,0.26) 0%, rgba(251,191,36,0) 72%);
                    pointer-events:none;
                }
                .release-shell::after {
                    content:'';
                    position:absolute;
                    right:-58px;
                    bottom:-120px;
                    width:300px;
                    height:300px;
                    background:radial-gradient(circle, rgba(20,184,166,0.20) 0%, rgba(20,184,166,0) 72%);
                    pointer-events:none;
                }
                .release-hero,
                .release-panel,
                .release-featured,
                .release-timeline-card,
                .release-supplemental {
                    position:relative;
                    z-index:1;
                }
                .release-badge {
                    display:inline-flex;
                    align-items:center;
                    gap:0.35rem;
                    padding:0.3rem 0.68rem;
                    border-radius:999px;
                    background:#0f172a;
                    color:#f8fafc;
                    font-size:0.72rem;
                    font-weight:800;
                    letter-spacing:0.05em;
                    text-transform:uppercase;
                }
                .release-title {
                    font-size:1.6rem;
                    font-weight:900;
                    line-height:1.05;
                    color:#7c2d12;
                    margin-top:0.65rem;
                }
                .release-sub {
                    max-width:860px;
                    margin-top:0.45rem;
                    font-size:0.92rem;
                    line-height:1.55;
                    color:#7c2d12;
                }
                .release-intro {
                    margin-top:0.95rem;
                    padding:0.9rem 1rem;
                    border-radius:18px;
                    border:1px solid rgba(249,115,22,0.18);
                    background:rgba(255,255,255,0.65);
                    color:#5b341f;
                    box-shadow:0 10px 24px rgba(124,45,18,0.08);
                }
                .release-intro p,
                .release-featured-copy p,
                .release-featured-copy ul,
                .release-timeline-copy p,
                .release-timeline-copy ul,
                .release-supplemental-copy p,
                .release-supplemental-copy ul {
                    margin:0.45rem 0 0 0;
                    padding-left:1rem;
                }
                .release-intro p,
                .release-featured-copy p,
                .release-timeline-copy p,
                .release-supplemental-copy p {
                    padding-left:0;
                }
                .release-intro ul,
                .release-featured-copy ul,
                .release-timeline-copy ul,
                .release-supplemental-copy ul {
                    line-height:1.55;
                }
                .release-panel {
                    border:1px solid rgba(14,165,233,0.18);
                    border-radius:20px;
                    padding:0.95rem;
                    background:rgba(255,255,255,0.76);
                    backdrop-filter:blur(3px);
                    box-shadow:0 12px 28px rgba(15,23,42,0.06);
                    height:100%;
                }
                .release-panel-title {
                    font-size:0.78rem;
                    font-weight:800;
                    letter-spacing:0.05em;
                    text-transform:uppercase;
                    color:#0f766e;
                }
                .release-stats-grid {
                    display:grid;
                    grid-template-columns:repeat(2, minmax(0, 1fr));
                    gap:0.7rem;
                    margin-top:0.8rem;
                }
                .release-stat {
                    border-radius:16px;
                    padding:0.8rem 0.85rem;
                    background:linear-gradient(180deg, rgba(255,255,255,0.94) 0%, rgba(240,253,250,0.92) 100%);
                    border:1px solid rgba(13,148,136,0.16);
                }
                .release-stat-label {
                    font-size:0.7rem;
                    font-weight:800;
                    letter-spacing:0.05em;
                    text-transform:uppercase;
                    color:#64748b;
                }
                .release-stat-value {
                    margin-top:0.28rem;
                    font-size:1rem;
                    font-weight:900;
                    line-height:1.2;
                    color:#0f172a;
                }
                .release-featured {
                    margin-top:0.2rem;
                    border:1px solid rgba(249,115,22,0.28);
                    border-radius:22px;
                    padding:1rem 1.05rem;
                    background:linear-gradient(135deg,#7c2d12 0%,#c2410c 52%,#0f766e 100%);
                    color:#fff7ed;
                    box-shadow:0 16px 34px rgba(124,45,18,0.18);
                }
                .release-featured-top,
                .release-timeline-top {
                    display:flex;
                    align-items:baseline;
                    justify-content:space-between;
                    gap:0.7rem;
                    flex-wrap:wrap;
                }
                .release-featured-kicker,
                .release-timeline-kicker,
                .release-supplemental-kicker {
                    font-size:0.72rem;
                    font-weight:800;
                    letter-spacing:0.05em;
                    text-transform:uppercase;
                }
                .release-featured-kicker {
                    color:#fde68a;
                }
                .release-featured-date,
                .release-timeline-date {
                    font-size:0.76rem;
                    color:#fed7aa;
                }
                .release-featured-title {
                    margin-top:0.6rem;
                    font-size:1.35rem;
                    font-weight:900;
                    line-height:1.15;
                    color:#fff7ed;
                }
                .release-featured-summary {
                    margin-top:0.55rem;
                    font-size:0.96rem;
                    line-height:1.5;
                    color:#fff7ed;
                }
                .release-featured-copy {
                    margin-top:0.8rem;
                    color:#ffedd5;
                    font-size:0.88rem;
                    line-height:1.55;
                }
                .release-browse-label {
                    margin:0.2rem 0 0.35rem 0;
                    font-size:0.75rem;
                    font-weight:800;
                    letter-spacing:0.05em;
                    text-transform:uppercase;
                    color:#0f766e;
                }
                .release-timeline-card {
                    border-radius:20px;
                    border:1px solid rgba(226,232,240,0.95);
                    background:rgba(255,255,255,0.90);
                    padding:0.95rem 1rem;
                    box-shadow:0 12px 28px rgba(15,23,42,0.06);
                    margin-top:0.85rem;
                }
                .release-timeline-card.is-warm {
                    border-color:rgba(249,115,22,0.22);
                    box-shadow:0 12px 28px rgba(124,45,18,0.08);
                }
                .release-timeline-card.is-cool {
                    border-color:rgba(13,148,136,0.20);
                    box-shadow:0 12px 28px rgba(15,118,110,0.08);
                }
                .release-timeline-kicker {
                    color:#0f766e;
                }
                .release-timeline-date {
                    color:#64748b;
                }
                .release-timeline-title {
                    margin-top:0.45rem;
                    font-size:1.08rem;
                    font-weight:900;
                    line-height:1.25;
                    color:#0f172a;
                }
                .release-timeline-copy {
                    margin-top:0.7rem;
                    color:#334155;
                    font-size:0.88rem;
                    line-height:1.55;
                }
                .release-chip {
                    display:inline-flex;
                    align-items:center;
                    border-radius:999px;
                    padding:0.26rem 0.6rem;
                    background:rgba(15,118,110,0.10);
                    border:1px solid rgba(15,118,110,0.16);
                    color:#115e59;
                    font-size:0.72rem;
                    font-weight:800;
                }
                .release-supplemental {
                    margin-top:0.9rem;
                    border-radius:18px;
                    border:1px solid rgba(148,163,184,0.18);
                    background:rgba(255,255,255,0.82);
                    padding:0.9rem 1rem;
                }
                .release-supplemental-kicker {
                    color:#475569;
                }
                .release-supplemental-title {
                    margin-top:0.4rem;
                    font-size:1rem;
                    font-weight:900;
                    color:#0f172a;
                }
                .release-supplemental-copy {
                    margin-top:0.55rem;
                    color:#334155;
                    font-size:0.86rem;
                    line-height:1.55;
                }
                @media (max-width: 900px) {
                    .release-title {
                        font-size:1.3rem;
                    }
                    .release-featured-title {
                        font-size:1.15rem;
                    }
                    .release-stats-grid {
                        grid-template-columns:1fr;
                    }
                }
                </style>
                """,
                unsafe_allow_html=True,
        )


def _render_top_cards(
        *,
        preamble_lines: list[str],
        milestones: list[ChangelogMilestone],
) -> None:
        latest = milestones[0]
        total_highlights = sum(_count_highlights(item.body_lines) for item in milestones)
        stats_html = "".join(
                (
                        "<div class='release-stat'><div class='release-stat-label'>Milestones</div>"
                        f"<div class='release-stat-value'>{len(milestones)}</div></div>",
                        "<div class='release-stat'><div class='release-stat-label'>Highlights</div>"
                        f"<div class='release-stat-value'>{total_highlights}</div></div>",
                        "<div class='release-stat'><div class='release-stat-label'>Coverage</div>"
                        f"<div class='release-stat-value'>{html.escape(_format_span_label(milestones), quote=True)}</div></div>",
                        "<div class='release-stat'><div class='release-stat-label'>Latest Window</div>"
                        f"<div class='release-stat-value'>{html.escape(latest.date_label, quote=True)}</div></div>",
                )
        )

        intro_html = _render_simple_markdown_html(preamble_lines)
        if intro_html:
                st.markdown(f"<div class='release-intro'>{intro_html}</div>", unsafe_allow_html=True)

        left_col, right_col = st.columns([1.4, 1.0], gap="large")
        with left_col:
                st.markdown(
                        (
                                "<article class='release-featured'>"
                                "<div class='release-featured-top'>"
                                "<div class='release-featured-kicker'>Latest release</div>"
                                f"<div class='release-chip'>{_count_highlights(latest.body_lines)} highlights</div>"
                                "</div>"
                                f"<div class='release-featured-title'>{html.escape(latest.title, quote=True)}</div>"
                                f"<div class='release-featured-date'>{html.escape(latest.date_label, quote=True)}</div>"
                                f"<div class='release-featured-summary'>{html.escape(latest.summary or 'Milestone update', quote=True)}</div>"
                                f"<div class='release-featured-copy'>{_render_simple_markdown_html(latest.body_lines)}</div>"
                                "</article>"
                        ),
                        unsafe_allow_html=True,
                )
        with right_col:
                st.markdown(
                        (
                                "<section class='release-panel'>"
                                "<div class='release-panel-title'>Release snapshot</div>"
                                f"<div class='release-stats-grid'>{stats_html}</div>"
                                "</section>"
                        ),
                        unsafe_allow_html=True,
                )


def _render_milestone_timeline(milestones: list[ChangelogMilestone]) -> None:
        for idx, milestone in enumerate(milestones, start=1):
                accent_class = "is-warm" if idx % 2 else "is-cool"
                body_html = _render_simple_markdown_html(milestone.body_lines)
                st.markdown(
                        (
                                f"<article class='release-timeline-card {accent_class}'>"
                                "<div class='release-timeline-top'>"
                                f"<div class='release-timeline-kicker'>Milestone {idx}</div>"
                                f"<div class='release-timeline-date'>{html.escape(milestone.date_label, quote=True)}</div>"
                                "</div>"
                                f"<div class='release-timeline-title'>{html.escape(milestone.title, quote=True)}</div>"
                                f"<div class='release-timeline-copy'>{body_html}</div>"
                                "</article>"
                        ),
                        unsafe_allow_html=True,
                )


def _render_supplemental_sections(sections: list[tuple[str, list[str]]]) -> None:
        for heading, lines in sections:
                body_html = _render_simple_markdown_html(lines)
                if not body_html:
                        continue
                st.markdown(
                        (
                                "<section class='release-supplemental'>"
                                "<div class='release-supplemental-kicker'>Additional context</div>"
                                f"<div class='release-supplemental-title'>{html.escape(heading, quote=True)}</div>"
                                f"<div class='release-supplemental-copy'>{body_html}</div>"
                                "</section>"
                        ),
                        unsafe_allow_html=True,
                )


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
    _render_release_history_styles()
    st.markdown(
        (
            "<div class='release-shell'>"
            "<section class='release-hero'>"
            "<div class='release-badge'>Release History</div>"
            "<div class='release-title'>Milestone timeline from repo inception</div>"
            "<div class='release-sub'>"
            "What's New stays short and recent. This page keeps the broader release story in one place, using the changelog markdown as the single source of truth."
            "</div>"
            "</section>"
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

    preamble_lines, milestones, supplemental_sections = _parse_changelog(changelog_text)
    if not milestones:
        st.markdown(changelog_text)
        return

    option_labels = ["All milestones", *[_milestone_option_label(item) for item in milestones]]
    selected_label = st.selectbox(
        "Browse milestone",
        options=option_labels,
        index=0,
        help="Focus a specific milestone or keep the full timeline view.",
    )

    _render_top_cards(
        preamble_lines=preamble_lines,
        milestones=milestones,
    )

    st.markdown("<div class='release-browse-label'>Timeline</div>", unsafe_allow_html=True)
    milestones_to_render = milestones
    if selected_label != "All milestones":
        milestones_to_render = [item for item in milestones if _milestone_option_label(item) == selected_label]

    _render_milestone_timeline(milestones_to_render)
    _render_supplemental_sections(supplemental_sections)