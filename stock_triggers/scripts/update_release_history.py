from __future__ import annotations

import json
from datetime import date
from pathlib import Path
import re


WHATS_NEW_PATH = Path("stock_triggers/data/whats_new.json")
CHANGELOG_PATH = Path("stock_triggers/docs/CHANGELOG.md")
_DATE_HEADING_RE = re.compile(r"^##\s+.+$", re.MULTILINE)
_AUTO_COMMITS_RE = re.compile(r"<!--\s*auto-release-source-commits:\s*([^>]+?)\s*-->")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _normalize_text(value: object) -> str:
    return " ".join(str(value or "").strip().split())


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


def _render_auto_section(entry: dict[str, object]) -> str:
    entry_date = _normalize_text(entry.get("date")) or date.today().isoformat()
    title = _normalize_text(entry.get("title")) or "Auto push summary"
    summary = _normalize_text(entry.get("summary"))
    details = _normalize_text(entry.get("details"))
    impact = _normalize_text(entry.get("impact"))
    source_ref = _normalize_text(entry.get("source_ref")) or "refs/heads/master"
    source_commits = _extract_source_commits(entry)
    commit_list = ", ".join(source_commits)

    lines = [
        f"## {entry_date}",
        "",
        f"### Auto push summary: {title}",
        f"- Auto-generated from commits pushed to `{source_ref}`.",
    ]
    if summary:
        lines.append(f"- Summary: {summary}")
    if details:
        lines.append(f"- Details: {details}")
    if impact:
        lines.append(f"- Impact: {impact}")
    if commit_list:
        lines.append(f"<!-- auto-release-source-commits: {commit_list} -->")
    lines.append("<!-- auto-release-generated: true -->")
    lines.append("")
    return "\n".join(lines)


def _top_section_range(text: str) -> tuple[int, int] | None:
    matches = list(_DATE_HEADING_RE.finditer(text))
    if not matches:
        return None
    start = matches[0].start()
    end = matches[1].start() if len(matches) > 1 else len(text)
    return start, end


def _extract_top_auto_commits(text: str) -> list[str]:
    section_range = _top_section_range(text)
    if section_range is None:
        return []
    start, end = section_range
    top_section = text[start:end]
    match = _AUTO_COMMITS_RE.search(top_section)
    if not match:
        return []
    values = [item.strip() for item in match.group(1).split(",")]
    return [item for item in values if item]


def update_release_history(repo_root: Path) -> bool:
    whats_new_payload = _load_whats_new(repo_root / WHATS_NEW_PATH)
    latest_entry = _get_latest_auto_entry(whats_new_payload)
    if latest_entry is None:
        return False

    changelog_path = repo_root / CHANGELOG_PATH
    if not changelog_path.exists():
        return False

    text = changelog_path.read_text(encoding="utf-8")
    source_commits = _extract_source_commits(latest_entry)
    top_auto_commits = _extract_top_auto_commits(text)

    if source_commits and top_auto_commits == source_commits:
        return False

    new_section = _render_auto_section(latest_entry)
    section_range = _top_section_range(text)
    if section_range is None:
        next_text = text.rstrip() + "\n\n" + new_section
    else:
        start, _ = section_range
        next_text = text[:start] + new_section + text[start:]

    if next_text == text:
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