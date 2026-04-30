from __future__ import annotations

import json
from datetime import date
from pathlib import Path
import re


WHATS_NEW_PATH = Path("stock_triggers/data/whats_new.json")
CHANGELOG_PATH = Path("stock_triggers/docs/CHANGELOG.md")
_DATE_HEADING_RE = re.compile(r"^##\s+.+$", re.MULTILINE)
_AUTO_MARKER_LINE_RE = re.compile(r"^\s*<!--\s*auto-release-(?:source-commits|generated):.*?-->\s*$", re.MULTILINE)
_AUTO_PUSH_PREFIX_RE = re.compile(r"^auto\s+push\s+summary\s*:\s*", re.IGNORECASE)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _normalize_text(value: object) -> str:
    return " ".join(str(value or "").strip().split())


def _clean_push_headline(value: object) -> str:
    title = _normalize_text(value)
    if not title:
        return ""
    return _AUTO_PUSH_PREFIX_RE.sub("", title).strip()


def _load_whats_new(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}
    if isinstance(payload, dict):
        return payload
    return {}


def _get_latest_auto_entry(payload: dict) -> dict[str, object] | None:
    entries = payload.get("entries")
    if not isinstance(entries, list):
        return None
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        if entry.get("auto_generated") is True:
            return entry
    return None


def _extract_source_commits(entry: dict[str, object]) -> list[str]:
    values = entry.get("source_commits")
    if not isinstance(values, list):
        return []
    commits: list[str] = []
    for value in values:
        sha = _normalize_text(value)
        if sha:
            commits.append(sha)
    return commits


def _strip_auto_markers(text: str) -> str:
    cleaned = _AUTO_MARKER_LINE_RE.sub("", text)
    # Keep section spacing stable after removing marker lines.
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned


def _render_auto_section(entry: dict[str, object]) -> str:
    entry_date = _normalize_text(entry.get("date")) or date.today().isoformat()
    title = _clean_push_headline(entry.get("title")) or "Push summary"
    summary = _normalize_text(entry.get("summary"))
    details = _normalize_text(entry.get("details"))
    impact = _normalize_text(entry.get("impact"))
    source_ref = _normalize_text(entry.get("source_ref")) or "refs/heads/master"

    lines = [
        f"## {entry_date}",
        "",
        f"### {title}",
        f"- Auto-generated from commits pushed to `{source_ref}`.",
    ]
    if summary:
        lines.append(f"- Summary: {summary}")
    if details:
        lines.append(f"- Details: {details}")
    if impact:
        lines.append(f"- Impact: {impact}")
    lines.append("")
    return "\n".join(lines)


def _top_section_range(text: str) -> tuple[int, int] | None:
    matches = list(_DATE_HEADING_RE.finditer(text))
    if not matches:
        return None
    start = matches[0].start()
    end = matches[1].start() if len(matches) > 1 else len(text)
    return start, end


def update_release_history(repo_root: Path) -> bool:
    whats_new_payload = _load_whats_new(repo_root / WHATS_NEW_PATH)
    latest_entry = _get_latest_auto_entry(whats_new_payload)
    if latest_entry is None:
        return False

    changelog_path = repo_root / CHANGELOG_PATH
    if not changelog_path.exists():
        return False

    original_text = changelog_path.read_text(encoding="utf-8")
    text = _strip_auto_markers(original_text)

    new_section = _render_auto_section(latest_entry)
    section_range = _top_section_range(text)
    if section_range is None:
        next_text = text.rstrip() + "\n\n" + new_section
    else:
        start, end = section_range
        top_section = text[start:end].strip()
        if top_section == new_section.strip():
            next_text = text
        else:
            next_text = text[:start] + new_section + text[start:]

    if next_text == original_text:
        return False

    changelog_path.write_text(next_text, encoding="utf-8")
    return True


def main() -> int:
    repo_root = _repo_root()
    changed = update_release_history(repo_root)
    if changed:
        print(f"Updated {CHANGELOG_PATH}")
    else:
        print("No Release History update needed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())