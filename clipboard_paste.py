"""
clipboard_paste.py

Put transcribed text on the Linux clipboard and paste it at the current
cursor position, without the user touching the keyboard.

Target platform: Fedora Linux, GNOME on Wayland.

Strategy:
  - Copy:   wl-copy (wl-clipboard) -- the standard Wayland clipboard tool.
  - Paste:  ydotool types Ctrl+V as a virtual uinput keyboard. Unlike
            wtype/xdotool, it does not depend on the focused application's
            Wayland/X11 text-input protocol, so it works everywhere.

Failure handling: every step prints a clear, actionable error and returns
False instead of raising, so a clipboard/paste problem can never break the
recording loop.

The public entry point is copy_and_paste(text). This module is intentionally
self-contained so a future LLM prompt-refinement stage can be dropped between
Whisper and copy_and_paste() without touching the rest of the pipeline.
"""

import os
import pathlib
import shutil
import subprocess
import sys
import tempfile
import time


# -----------------------------
# Configuration
# -----------------------------

# Delay (seconds) between putting text on the clipboard and pressing Ctrl+V.
# Clipboard managers (GNOME's, CopyQ, etc.) can grab the new selection
# asynchronously; a short grace period avoids pasting the previous contents.
PASTE_DELAY = 0.15

# Max seconds to wait on subprocesses before giving up.
TOOL_TIMEOUT = 5

# ydotoold's socket location varies between builds: newer upstream uses
# /run/ydotoold/socket, Fedora's package defaults to /tmp/.ydotool_socket.
# The client honors $YDOTOOL_SOCKET if set, so check that first.
def _socket_candidates():
    candidates = [
        os.environ.get("YDOTOOL_SOCKET"),
        # 1.0.4 client default: $XDG_RUNTIME_DIR/.ydotool_socket (user daemon)
        os.path.join(
            os.environ.get("XDG_RUNTIME_DIR", f"/run/user/{os.getuid()}"),
            ".ydotool_socket",
        ),
        "/run/ydotoold/socket",
        "/tmp/.ydotool_socket",
    ]
    return [pathlib.Path(c) for c in candidates if c]

# Tools we need, with the Fedora package / setup that provides them.
REQUIRED_TOOLS = {
    "wl-copy": {
        "package": "wl-clipboard",
        "install": "sudo dnf install wl-clipboard",
    },
    "ydotool": {
        "package": "ydotool",
        "install": (
            "sudo dnf install ydotool, then enable the daemon:\n"
            "    sudo systemctl enable --now ydotool.service\n"
            "    (runs /usr/libexec/ydotoold with access to /dev/uinput)"
        ),
    },
}


# -----------------------------
# Dependency checks
# -----------------------------

def check_dependencies():
    """
    Verify the required system utilities are installed.

    Returns a list of problems; empty list means everything is available.
    Never raises.
    """
    problems = []

    for tool, info in REQUIRED_TOOLS.items():
        if shutil.which(tool) is None:
            problems.append(
                f"Missing '{tool}' ({info['package']}). Install with:\n"
                f"    {info['install']}"
            )
        elif tool == "ydotool":
            problems.extend(_check_ydotoold())

    return problems


def _check_ydotoold():
    """Check that the ydotoold daemon socket exists in a known location."""
    candidates = _socket_candidates()

    for sock in candidates:
        if sock.exists():
            if sock.is_socket():
                return []
            return [
                f"ydotool daemon socket exists but is not a socket: {sock}"
            ]

    searched = ", ".join(str(sock) for sock in candidates)
    return [
        "ydotool is installed, but its daemon socket was not found.\n"
        f"    Searched: {searched}\n"
        f"Start the daemon with:\n"
        f"    sudo systemctl enable --now ydotool.service"
    ]


def report_dependencies(stream=sys.stderr):
    """Print a dependency report. Returns True if everything is OK."""
    problems = check_dependencies()

    if not problems:
        print("Clipboard/paste dependencies: OK (wl-copy, ydotoold).")
        return True

    print("Clipboard/paste dependency problems:", file=stream)
    for problem in problems:
        print("  - " + problem, file=stream)

    return False


# -----------------------------
# Clipboard (Wayland)
# -----------------------------

def copy_to_clipboard(text):
    """
    Copy text to the Wayland clipboard via wl-copy.

    Returns True on success, False on failure (details printed to stderr).
    """
    # NOTE: wl-copy forks to the background to keep serving the clipboard,
    # and the forked child inherits our stderr fd. If we used a pipe for
    # stderr, run() would block until EOF on that pipe (i.e. until the
    # background child dies) and "time out" even though the copy worked.
    # So we capture stderr into a temp file instead of a pipe.
    try:
        with tempfile.TemporaryFile(mode="w+") as errfile:
            proc = subprocess.run(
                ["wl-copy", text],
                timeout=TOOL_TIMEOUT,
                stdout=subprocess.DEVNULL,
                stderr=errfile,
            )
            errfile.seek(0)
            stderr = errfile.read().strip()
    except FileNotFoundError:
        print(
            "ERROR: wl-copy not found. Install it with:\n"
            "    sudo dnf install wl-clipboard",
            file=sys.stderr,
        )
        return False
    except subprocess.TimeoutExpired:
        print("ERROR: wl-copy timed out.", file=sys.stderr)
        return False

    if proc.returncode != 0:
        print(
            f"ERROR: wl-copy failed (exit {proc.returncode}): "
            f"{stderr or 'no error output'}\n"
            f"Hint: check that WAYLAND_DISPLAY is set (GNOME Wayland session).",
            file=sys.stderr,
        )
        return False

    return True


# -----------------------------
# Paste at cursor (virtual Ctrl+V)
# -----------------------------

def paste_at_cursor():
    """
    Paste by emitting Ctrl+V through ydotool (uinput virtual keyboard).

    This works on GNOME Wayland regardless of which application has focus.

    Returns True on success, False on failure (details printed to stderr).
    """
    # Same tempfile trick as wl-copy: never capture tool stderr in a pipe,
    # so a background child inheriting the fd can't make run() hang.
    try:
        with tempfile.TemporaryFile(mode="w+") as errfile:
            proc = subprocess.run(
                [
                    "ydotool", "key",
                    "29:1",   # left Ctrl down
                    "47:1",   # V down
                    "47:0",   # V up
                    "29:0",   # left Ctrl up
                ],
                timeout=TOOL_TIMEOUT,
                stdout=subprocess.DEVNULL,
                stderr=errfile,
            )
            errfile.seek(0)
            stderr = errfile.read().strip()
    except FileNotFoundError:
        print(
            "ERROR: ydotool not found. Install it with:\n"
            "    sudo dnf install ydotool\n"
            "then start its daemon:\n"
            "    sudo systemctl enable --now ydotool.service",
            file=sys.stderr,
        )
        return False
    except subprocess.TimeoutExpired:
        print("ERROR: ydotool timed out.", file=sys.stderr)
        return False

    if proc.returncode != 0:
        print(
            f"ERROR: ydotool failed (exit {proc.returncode}): "
            f"{stderr or 'no error output'}\n"
            f"Hint: is the user daemon running? Check 'systemctl --user status ydotool.service'.",
            file=sys.stderr,
        )
        return False

    return True


# -----------------------------
# Full pipeline step
# -----------------------------

def copy_and_paste(text):
    """
    Copy text to the clipboard and paste it into the focused application.

    Returns True on success, False if any step failed (with a clear,
    actionable error already printed).
    """
    text = (text or "").strip()
    if not text:
        print("Nothing to paste: transcription is empty.", file=sys.stderr)
        return False

    problems = check_dependencies()
    if problems:
        print(
            "Cannot copy/paste: missing clipboard tooling.",
            file=sys.stderr,
        )
        for problem in problems:
            print("  - " + problem, file=sys.stderr)
        return False

    if not copy_to_clipboard(text):
        return False

    print("Clipboard updated.")

    # Give clipboard managers a moment to pick up the new selection.
    time.sleep(PASTE_DELAY)

    if not paste_at_cursor():
        print(
            "Text is on the clipboard; you can paste it manually with Ctrl+V.",
            file=sys.stderr,
        )
        return False

    print("Pasted at cursor.")
    return True


# -----------------------------
# CLI: quick self-test
# -----------------------------

if __name__ == "__main__":
    if not report_dependencies():
        sys.exit(1)

    demo = "Clipboard test from clipboard_paste.py"
    print(f'Copying "{demo}" and pasting at cursor...')
    if copy_and_paste(demo):
        print("Self-test OK. Check the focused window for the text.")
    else:
        sys.exit(1)
