"""Rendering helpers for diagnostics output.

Kept separate from the CLI commands so the formatting can be unit-tested
without invoking a command, and so both `setup` and `doctor` present the same
check results in the way that suits them.
"""

from __future__ import annotations

import sys

from ..diagnostics.base import Check, DiagnosticReport, Status

#: Symbols used in checklist output. Plain ASCII fallback is handled by
#: encoding below, so this also renders in a non-UTF-8 terminal.
SYMBOLS = {
    Status.OK: "✓",
    Status.WARN: "!",
    Status.FAIL: "✗",
    Status.INFO: "-",
}


def _safe(text: str) -> str:
    """Best-effort safe printing for terminals that cannot encode the symbols."""
    encoding = getattr(sys.stdout, "encoding", None) or "utf-8"
    try:
        text.encode(encoding)
        return text
    except (UnicodeEncodeError, LookupError):
        return (
            text.replace("✓", "OK  ")
            .replace("✗", "FAIL")
            .replace("→", "->")
        )


def render_table(report: DiagnosticReport) -> str:
    """Aligned ``name  STATUS  detail`` lines, for `micdrop doctor`."""
    if not report.checks:
        return "(no checks ran)"

    width = max(len(c.name) for c in report.checks)
    lines = []
    for check in report.checks:
        status = check.status.value.ljust(width and 5 or 5)
        detail = f"  {check.detail}" if check.detail else ""
        lines.append(f"{check.name.ljust(width)}  {status}{detail}")
    return "\n".join(lines)


def render_checklist(report: DiagnosticReport, indent: str = "") -> str:
    """``✓ name`` lines with indented fix instructions, for `micdrop setup`."""
    lines = []
    for check in report.checks:
        symbol = SYMBOLS[check.status]
        line = f"{indent}{symbol} {check.name}"
        if check.status is Status.OK and check.detail:
            line += f" — {check.detail}"
        lines.append(_safe(line))
        if check.fix and check.status in (Status.FAIL, Status.WARN):
            for fix_line in check.fix.splitlines():
                lines.append(_safe(f"{indent}    {fix_line}"))
    return "\n".join(lines)


def fixes_for(report: DiagnosticReport) -> str:
    """Just the actionable fixes, grouped for display."""
    actionable = [c for c in report.checks if c.fix and c.status in (Status.FAIL, Status.WARN)]
    if not actionable:
        return ""
    blocks = []
    for check in actionable:
        blocks.append(f"{check.name}\n    " + check.fix.replace("\n", "\n    "))
    return "\n\n".join(blocks)


def overall_line(report: DiagnosticReport) -> str:
    """The one-word verdict `micdrop doctor` ends on."""
    if report.ready:
        if report.warnings:
            return f"Overall: READY ({len(report.warnings)} warning(s))"
        return "Overall: READY"
    return f"Overall: NOT READY ({len(report.failures)} problem(s))"


def print_report(report: DiagnosticReport, style: str = "table") -> None:
    """Print a report in the requested style."""
    if style == "checklist":
        print(render_checklist(report))
    else:
        print(render_table(report))


def print_overall(report: DiagnosticReport) -> None:
    print(overall_line(report))


__all__ = [
    "Check",
    "DiagnosticReport",
    "Status",
    "render_table",
    "render_checklist",
    "fixes_for",
    "overall_line",
    "print_report",
    "print_overall",
]