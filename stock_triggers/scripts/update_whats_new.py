from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from datetime import date
from pathlib import Path


AUTO_COMMIT_SUBJECT = "Update What's New for master push"
ZERO_OID = "0" * 40
WHATS_NEW_PATH = Path("stock_triggers/data/whats_new.json")
MAX_SUBJECTS_IN_SUMMARY = 4
MAX_SUBJECTS_IN_DETAILS = 8


@dataclass(frozen=True)
class CommitRecord:
    sha: str
    subject: str
    body: str
    files: tuple[str, ...]


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _git(*args: str, repo_root: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo_root), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout


def _git_optional(*args: str, repo_root: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo_root), *args],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return ""
    return result.stdout


def _coerce_subject(subject: str) -> str:
    clean = " ".join(str(subject).strip().split())
    return clean.rstrip(".")


def _list_commit_shas(repo_root: Path, *, local_oid: str, remote_oid: str) -> list[str]:
    if not local_oid or local_oid == ZERO_OID:
        return []
    if not remote_oid or remote_oid == ZERO_OID:
        output = _git("rev-list", "--reverse", local_oid, repo_root=repo_root)
    else:
        output = _git("rev-list", "--reverse", f"{remote_oid}..{local_oid}", repo_root=repo_root)
    return [line.strip() for line in output.splitlines() if line.strip()]


def _read_commit(repo_root: Path, sha: str) -> CommitRecord:
    subject = _coerce_subject(
        _git("show", "-s", "--format=%s", sha, repo_root=repo_root).strip()
    )
    body = _git_optional("show", "-s", "--format=%b", sha, repo_root=repo_root).strip()
    files_output = _git_optional(
        "diff-tree", "--no-commit-id", "--name-only", "-r", sha, repo_root=repo_root
    )
    files = tuple(line.strip() for line in files_output.splitlines() if line.strip())
    return CommitRecord(sha=sha, subject=subject, body=body, files=files)


def _is_auto_commit(commit: CommitRecord) -> bool:
    if commit.subject != AUTO_COMMIT_SUBJECT:
        return False
    tracked_files = set(commit.files)
    return tracked_files == {str(WHATS_NEW_PATH)}


def _load_payload(path: Path) -> dict:
    if not path.exists():
        return {"updated_at": "", "entries": []}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {"updated_at": "", "entries": []}
    if isinstance(data, list):
        return {"updated_at": "", "entries": data}
    if not isinstance(data, dict):
        return {"updated_at": "", "entries": []}
    entries = data.get("entries")
    if not isinstance(entries, list):
        entries = []
    return {"updated_at": str(data.get("updated_at", "")).strip(), "entries": entries}


def _infer_tag(subjects: list[str]) -> str:
    lowered = [subject.lower() for subject in subjects]
    if len(subjects) == 1:
        subject = lowered[0]
        if subject.startswith(("fix", "bugfix", "hotfix")):
            return "Fix"
        if subject.startswith(("add", "feat", "feature", "introduce", "create")):
            return "Feature"
        if subject.startswith(("doc", "docs")):
            return "Docs"
        if subject.startswith(("refactor", "cleanup")):
            return "Refactor"
    return "Auto push summary"


def _format_subject_list(subjects: list[str], limit: int) -> str:
    visible = subjects[:limit]
    joined = "; ".join(visible)
    hidden = len(subjects) - len(visible)
    if hidden > 0:
        joined = f"{joined}; plus {hidden} more"
    return joined


def _categorize_paths(paths: list[str]) -> list[str]:
    categories: list[str] = []
    seen: set[str] = set()
    for path in paths:
        if path.startswith("stock_triggers/ui/"):
            label = "UI"
        elif path.startswith("stock_triggers/scripts/"):
            label = "trigger scripts"
        elif path.startswith("stock_triggers/data/"):
            label = "trigger data/config"
        elif path.startswith("stock_triggers/docs/"):
            label = "trigger docs"
        elif path.startswith("stock_selector/"):
            label = "stock selector"
        else:
            label = "repo root"
        if label not in seen:
            categories.append(label)
            seen.add(label)
    return categories


def _build_title(subjects: list[str]) -> str:
    if len(subjects) == 1:
        return subjects[0]
    return f"Master updated from {len(subjects)} unpushed commits"


def _build_summary(subjects: list[str]) -> str:
    if len(subjects) == 1:
        return f"Auto-captured from the commit being pushed to master: {subjects[0]}."
    return (
        f"Auto-captured from {len(subjects)} commits being pushed to master. "
        f"Highlights: {_format_subject_list(subjects, MAX_SUBJECTS_IN_SUMMARY)}."
    )


def _build_details(subjects: list[str], categories: list[str]) -> str:
    details = f"Commit list: {_format_subject_list(subjects, MAX_SUBJECTS_IN_DETAILS)}."
    if categories:
        details += f" Touched areas: {', '.join(categories)}."
    return details


def _build_impact(categories: list[str], subject_count: int) -> str:
    if not categories:
        return "Keeps the in-app What's New panel aligned with what is actually being pushed to master."
    areas = ", ".join(categories)
    noun = "commit" if subject_count == 1 else "commits"
    return f"Keeps the in-app What's New panel aligned with the {subject_count} {noun} being pushed across {areas}."


def _build_entry(commits: list[CommitRecord], remote_ref: str) -> dict[str, object]:
    subjects = [_coerce_subject(commit.subject) for commit in commits if _coerce_subject(commit.subject)]
    changed_paths = [path for commit in commits for path in commit.files if path != str(WHATS_NEW_PATH)]
    categories = _categorize_paths(changed_paths)
    return {
        "date": date.today().isoformat(),
        "tag": _infer_tag(subjects),
        "title": _build_title(subjects),
        "summary": _build_summary(subjects),
        "details": _build_details(subjects, categories),
        "impact": _build_impact(categories, len(subjects)),
        "auto_generated": True,
        "source_commits": [commit.sha for commit in commits],
        "source_ref": remote_ref,
    }


def _entries_equal(left: dict, right: dict) -> bool:
    return json.dumps(left, sort_keys=True) == json.dumps(right, sort_keys=True)


def update_whats_new(repo_root: Path, *, local_oid: str, remote_oid: str, remote_ref: str) -> bool:
    commit_shas = _list_commit_shas(repo_root, local_oid=local_oid, remote_oid=remote_oid)
    commits = [_read_commit(repo_root, sha) for sha in commit_shas]
    commits = [commit for commit in commits if not _is_auto_commit(commit)]
    if not commits:
        return False

    payload = _load_payload(repo_root / WHATS_NEW_PATH)
    entries = list(payload.get("entries", []))
    next_entry = _build_entry(commits, remote_ref)

    if entries and isinstance(entries[0], dict) and entries[0].get("auto_generated") is True:
        if _entries_equal(entries[0], next_entry):
            return False
        entries[0] = next_entry
    else:
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            if entry.get("auto_generated") is True and entry.get("source_commits") == next_entry["source_commits"]:
                return False
        entries.insert(0, next_entry)

    payload["updated_at"] = date.today().isoformat()
    payload["entries"] = entries
    (repo_root / WHATS_NEW_PATH).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return True


def _resolve_upstream(repo_root: Path) -> tuple[str, str]:
    remote_ref = _git(
        "rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{upstream}", repo_root=repo_root
    ).strip()
    remote_name, remote_branch = remote_ref.split("/", 1)
    remote_oid = _git("rev-parse", remote_ref, repo_root=repo_root).strip()
    local_oid = _git("rev-parse", "HEAD", repo_root=repo_root).strip()
    return local_oid, remote_oid, f"refs/heads/{remote_branch}"


def main() -> int:
    parser = argparse.ArgumentParser(description="Update stock_triggers/data/whats_new.json from pending push commits.")
    parser.add_argument("--local-oid", default="")
    parser.add_argument("--remote-oid", default="")
    parser.add_argument("--remote-ref", default="refs/heads/master")
    args = parser.parse_args()

    repo_root = _repo_root()
    local_oid = args.local_oid.strip()
    remote_oid = args.remote_oid.strip()
    remote_ref = args.remote_ref.strip() or "refs/heads/master"

    if not local_oid:
        try:
            local_oid, remote_oid, remote_ref = _resolve_upstream(repo_root)
        except (subprocess.CalledProcessError, ValueError) as exc:
            print(f"Unable to resolve upstream branch: {exc}", file=sys.stderr)
            return 1

    try:
        changed = update_whats_new(
            repo_root,
            local_oid=local_oid,
            remote_oid=remote_oid,
            remote_ref=remote_ref,
        )
    except subprocess.CalledProcessError as exc:
        print(exc.stderr.strip() or str(exc), file=sys.stderr)
        return exc.returncode or 1

    if changed:
        print(f"Updated {WHATS_NEW_PATH}")
    else:
        print("No What's New update needed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())