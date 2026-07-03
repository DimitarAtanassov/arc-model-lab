#!/usr/bin/env python3
"""Validate the Copilot stack for schema and integrity errors.

Checks every agent, prompt, instruction, and skill for valid frontmatter,
resolvable tool ids, absence of the legacy ``mode:`` key, and working internal
links. Exits non-zero when any error is found so it can gate CI.

No third-party dependencies: the frontmatter here is simple enough to parse
without a YAML library, which keeps the validator runnable anywhere Python is.
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

# Tool sets and ids known to the VS Code Copilot customization surface. Update
# this set when the surface adds tools; an unknown token is treated as an error
# so typos are caught rather than silently accepted.
TOOL_SETS = {"agent", "browser", "edit", "execute", "read", "search", "web", "vscode"}
TOOL_IDS = {
    "read/readFile", "read/problems", "read/terminalLastCommand",
    "read/terminalSelection", "read/getNotebookSummary", "read/readNotebookCellOutput",
    "search/codebase", "search/fileSearch", "search/textSearch",
    "search/listDirectory", "search/usages", "search/changes",
    "edit/createFile", "edit/editFiles", "edit/createDirectory", "edit/editNotebook",
    "execute/runInTerminal", "execute/getTerminalOutput", "execute/createAndRunTask",
    "execute/runNotebookCell", "execute/testFailure",
    "web/fetch",
    "vscode/askQuestions", "vscode/runCommand", "vscode/extensions",
    "vscode/installExtension", "vscode/VSCodeAPI",
    "githubRepo", "githubTextSearch", "newWorkspace", "selection", "todos",
}
VALID_PROMPT_AGENTS = {"ask", "agent", "plan"}

LINK_RE = re.compile(r"\[[^\]]+\]\(([^)]+)\)")


@dataclass
class Report:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def error(self, path: Path, msg: str) -> None:
        self.errors.append(f"{path}: {msg}")

    def warn(self, path: Path, msg: str) -> None:
        self.warnings.append(f"{path}: {msg}")


def split_frontmatter(text: str) -> tuple[dict[str, str], str] | None:
    """Return (frontmatter map, body) or None when no frontmatter block exists."""
    if not text.startswith("---"):
        return None
    end = text.find("\n---", 3)
    if end == -1:
        return None
    block = text[3:end].strip("\n")
    body = text[end + 4 :]
    fm: dict[str, str] = {}
    current: str | None = None
    for line in block.splitlines():
        if re.match(r"^\s*-\s", line) and current is not None:
            fm[current] += " " + line.strip()[1:].strip()
            continue
        match = re.match(r"^([A-Za-z][\w-]*):\s*(.*)$", line)
        if match:
            current = match.group(1)
            fm[current] = match.group(2).strip()
    return fm, body


def extract_tokens(value: str) -> list[str]:
    """Pull tool or agent tokens out of a flow or block frontmatter value."""
    inner = value.strip().strip("[]")
    tokens: list[str] = []
    for part in inner.split(","):
        part = part.strip().strip("\"'")
        if part:
            tokens.append(part)
    return tokens


def check_tools(path: Path, value: str, report: Report) -> None:
    for token in extract_tokens(value):
        if token in TOOL_SETS or token in TOOL_IDS:
            continue
        report.error(path, f"unknown tool id '{token}'")


def check_links(path: Path, body: str, report: Report) -> None:
    for target in LINK_RE.findall(body):
        if target.startswith(("http://", "https://", "#", "mailto:")):
            continue
        rel = target.split("#", 1)[0]
        if not rel:
            continue
        if not (path.parent / rel).resolve().exists():
            report.error(path, f"broken link to '{target}'")


def check_no_em_dash(path: Path, text: str, report: Report) -> None:
    if "\u2014" in text:
        report.warn(path, "contains an em dash (house style forbids it)")


def check_model_format(path: Path, value: str, report: Report) -> None:
    for token in extract_tokens(value):
        if "(" not in token or not token.rstrip().endswith(")"):
            report.warn(path, f"model '{token}' is not in 'Model Name (vendor)' form")


def check_handoff_targets(path: Path, text: str, report: Report) -> None:
    end = text.find("\n---", 3)
    block = text[3:end] if end != -1 else ""
    for line in block.splitlines():
        stripped = line.strip()
        if stripped.startswith("agent:"):
            target = stripped.partition("agent:")[2].strip()
            if target and not (path.parent / f"{target}.agent.md").exists():
                report.error(path, f"handoff to unknown agent '{target}'")


def validate_agent(path: Path, report: Report) -> None:
    text = path.read_text(encoding="utf-8")
    parsed = split_frontmatter(text)
    if parsed is None:
        report.error(path, "missing frontmatter")
        return
    fm, body = parsed
    if "description" not in fm:
        report.error(path, "agent missing 'description'")
    if "tools" in fm:
        check_tools(path, fm["tools"], report)
    if "model" in fm:
        check_model_format(path, fm["model"], report)
    else:
        report.warn(path, "agent has no pinned 'model'")
    check_handoff_targets(path, text, report)
    if "agents" in fm:
        for token in extract_tokens(fm["agents"]):
            if token != "*" and not (path.parent / f"{token}.agent.md").exists():
                report.error(path, f"subagent '{token}' has no matching agent file")
    check_links(path, body, report)
    check_no_em_dash(path, body, report)


def validate_prompt(path: Path, report: Report) -> None:
    parsed = split_frontmatter(path.read_text(encoding="utf-8"))
    if parsed is None:
        report.error(path, "missing frontmatter")
        return
    fm, body = parsed
    if "description" not in fm:
        report.error(path, "prompt missing 'description'")
    if "mode" in fm:
        report.error(path, "prompt uses legacy 'mode:'; use 'agent:'")
    if "tools" in fm:
        check_tools(path, fm["tools"], report)
    agent = fm.get("agent")
    if agent and agent not in VALID_PROMPT_AGENTS:
        # A custom agent name is valid; confirm the agent file exists.
        agent_file = path.parents[1] / "agents" / f"{agent}.agent.md"
        if not agent_file.exists():
            report.error(path, f"agent '{agent}' has no matching agent file")
    check_links(path, body, report)
    check_no_em_dash(path, body, report)


def validate_instruction(path: Path, report: Report) -> None:
    parsed = split_frontmatter(path.read_text(encoding="utf-8"))
    if parsed is None:
        report.error(path, "missing frontmatter")
        return
    fm, body = parsed
    if "applyTo" not in fm:
        report.error(path, "instruction missing 'applyTo'")
    check_links(path, body, report)
    check_no_em_dash(path, body, report)


def validate_skill(path: Path, report: Report) -> None:
    parsed = split_frontmatter(path.read_text(encoding="utf-8"))
    if parsed is None:
        report.error(path, "missing frontmatter")
        return
    fm, body = parsed
    for key in ("name", "description"):
        if key not in fm:
            report.error(path, f"skill missing '{key}'")
    check_links(path, body, report)
    check_no_em_dash(path, body, report)


def main() -> int:
    root = Path(__file__).resolve().parents[2]
    github = root / ".github"
    report = Report()

    for base in (root / "AGENTS.md", github / "copilot-instructions.md"):
        if not base.exists():
            report.error(base, "required file is missing")
        else:
            check_no_em_dash(base, base.read_text(encoding="utf-8"), report)

    for path in sorted((github / "agents").glob("*.agent.md")):
        validate_agent(path, report)
    for path in sorted((github / "prompts").glob("*.prompt.md")):
        validate_prompt(path, report)
    for path in sorted((github / "instructions").glob("*.instructions.md")):
        validate_instruction(path, report)
    for path in sorted((github / "skills").glob("*/SKILL.md")):
        validate_skill(path, report)

    for warning in report.warnings:
        print(f"warning: {warning}")
    for error in report.errors:
        print(f"error: {error}")

    counts = (
        f"{len(report.errors)} error(s), {len(report.warnings)} warning(s)"
    )
    if report.errors:
        print(f"\nStack validation FAILED: {counts}")
        return 1
    print(f"\nStack validation passed: {counts}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
