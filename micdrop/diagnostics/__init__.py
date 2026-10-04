"""Diagnostics: independent, actionable checks of everything MicDrop needs.

One engine, several views:

* ``micdrop setup``    — friendly checklist + fix instructions, creates a config
* ``micdrop doctor``   — the same checks, terse table, non-interactive
* ``micdrop status``   — what the app *would* use right now
* ``micdrop models``   — which models and providers are available

The checks are split in two:

* :mod:`micdrop.diagnostics.base` — platform-neutral checks (OS, display,
  microphone, Python packages, models, API keys, config file). Imports nothing
  platform specific.
* the selected platform's own ``diagnostics`` module — Linux evdev/uinput/group/
  hotkey/clipboard probes live in ``micdrop/platform/linux/diagnostics.py``.

That split is what keeps the core free of Linux assumptions, and it is enforced
by ``micdrop/tests/test_architecture.py``.
"""

from __future__ import annotations

import logging
from typing import Callable, Optional

from ..app.config import AppConfig
from .base import (
    BLOCKING,
    Check,
    DiagnosticReport,
    Status,
    check_config_file,
    check_desktop,
    check_display,
    check_groq,
    check_local_models,
    check_microphone,
    check_platform,
    check_python_packages,
    check_whisper,
    _huggingface_cached,
    _has_module,
    _run,
)

logger = logging.getLogger("micdrop")

__all__ = [
    "Status",
    "Check",
    "DiagnosticReport",
    "run_full_diagnostics",
    "platform_checks",
    "provider_summary",
    "check_platform",
    "check_display",
    "check_desktop",
    "check_microphone",
    "check_python_packages",
    "check_whisper",
    "check_local_models",
    "check_groq",
    "check_config_file",
    "BLOCKING",
]


def platform_checks(config: AppConfig) -> list[Check]:
    """Ask the selected platform for its own checks.

    Returns a single explanatory check when there is no backend for this
    platform, so `doctor` still explains itself on an unsupported system
    instead of crashing.
    """
    from ..platform import UnsupportedPlatformError, select_platform

    try:
        selection = select_platform()
    except UnsupportedPlatformError as exc:
        return [Check("Platform backend", Status.FAIL, str(exc))]

    hooks: Optional[Callable[[AppConfig], list[Check]]] = getattr(
        selection, "diagnostics", None
    )
    if hooks is None:
        return []
    return list(hooks(config) or [])


def run_full_diagnostics(config: AppConfig) -> DiagnosticReport:
    """Every check, each computed independently and never raising."""
    report = DiagnosticReport()

    def safe(fn, *args) -> list[Check]:
        # Even the failure path must not raise: a diagnostic that crashes while
        # reporting a crash is worse than useless.
        name = getattr(fn, "__name__", "check")
        pretty = name.removeprefix("check_").replace("_", " ").title()
        try:
            result = fn(*args)
        except Exception as exc:  # noqa: BLE001 - a broken probe is a WARN
            return [Check(pretty, Status.WARN, f"check failed: {exc}")]
        try:
            if isinstance(result, Check):
                return [result]
            return list(result or [])
        except Exception as exc:  # noqa: BLE001 - a probe returning junk
            return [Check(pretty, Status.WARN, f"check returned unusable data: {exc}")]

    core_checks = (
        (check_platform, ()),
        (check_desktop, ()),
        (check_display, ()),
        (check_microphone, ()),
        (check_python_packages, ()),
        (check_whisper, (config,)),
        (check_local_models, (config,)),
        (check_groq, (config,)),
    )
    for fn, fn_args in core_checks:
        for check in safe(fn, *fn_args):
            report.add(check)

    for check in safe(platform_checks, config):
        report.add(check)

    for check in safe(check_config_file):
        report.add(check)

    return report


def provider_summary(config: AppConfig) -> dict:
    """mode -> provider, for `micdrop status` / `micdrop models`."""
    return config.providers()