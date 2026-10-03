"""MicDrop CLI.

Subcommands:
  run              default: start the push-to-talk hotkey listener
  self-test        clipboard/paste dependency check + demo copy/paste
  record-test      record 10s from the microphone, report levels
  transcribe-test  transcribe a WAV file with Whisper
  refine-test      run the text processor on a sentence (REFINE or GENERATE)
  doctor           check components: packages, audio, keyboard, clipboard
  config           show the effective configuration and an example file
"""

from __future__ import annotations

import argparse
import logging
import sys

from .. import __version__
from ..app.config import AppConfig, default_config_example, find_config_file, load_config
from ..app.errors import MicDropError
from ..app.modes import Mode, DESCRIPTIONS, parse_mode


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="micdrop",
        description="Voice-to-prompt: push-to-talk -> Whisper -> LLM -> clipboard paste.",
    )
    parser.add_argument("--version", action="version", version=f"micdrop {__version__}")
    parser.add_argument(
        "--config", "-c", help="Path to a micdrop.toml config file", default=None
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Verbose (debug) logging"
    )

    sub = parser.add_subparsers(dest="command")

    # -- run --------------------------------------------------------------- #
    run = sub.add_parser("run", help="Start the push-to-talk listener (default)")
    run.add_argument("--provider", help="Override refinement.provider (e.g. 'none')", default=None)
    run.add_argument("--device", help="Override keyboard device path", default=None)
    run.add_argument("--debug-audio", action="store_true", help="Save recordings to debug/")

    # -- self-test ---------------------------------------------------------- #
    sub.add_parser("self-test", help="Clipboard/paste dependency check + demo paste")

    # -- record-test -------------------------------------------------------- #
    rec = sub.add_parser("record-test", help="Record N seconds and report audio levels")
    rec.add_argument("--duration", type=float, default=10.0)
    rec.add_argument("--output", default="test.wav")

    # -- transcribe-test ------------------------------------------------------ #
    tr = sub.add_parser("transcribe-test", help="Transcribe a WAV file with Whisper")
    tr.add_argument("file", nargs="?", default="test.wav")

    # -- refine-test --------------------------------------------------------- #
    rf = sub.add_parser("refine-test", help="Run the text processor on a sentence")
    rf.add_argument("text", nargs="*", help="Text to process (default: a demo sentence)")
    rf.add_argument(
        "--mode", default="refine", help="refine or generate (default: refine)"
    )

    # -- doctor --------------------------------------------------------------- #
    sub.add_parser("doctor", help="Check components: packages, audio, keyboard, clipboard")

    # -- hotkey-test ------------------------------------------------------------ #
    sub.add_parser(
        "hotkey-test",
        help="Round-trip both hotkeys through evdev using ydotool (no models, no paste)",
    )

    # -- config ----------------------------------------------------------------- #
    cfg = sub.add_parser("config", help="Show effective configuration")
    cfg.add_argument("--example", action="store_true", help="Print an example config file")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(message)s",
    )

    try:
        if args.command == "self-test":
            return cmd_self_test()
        if args.command == "record-test":
            return cmd_record_test(args)
        if args.command == "transcribe-test":
            return cmd_transcribe_test(args)
        if args.command == "refine-test":
            return cmd_refine_test(args)
        if args.command == "doctor":
            return cmd_doctor(args)
        if args.command == "hotkey-test":
            return cmd_hotkey_test(args)
        if args.command == "config":
            return cmd_config(args)
        if args.command == "run":
            return cmd_run(args)
        # No subcommand: default to run.
        return cmd_run(args)
    except MicDropError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nExiting...")
        return 0


# ---------------------------------------------------------------------------- #
# Commands
# ---------------------------------------------------------------------------- #


def cmd_run(args) -> int:
    """The main application: hotkey listener + full pipeline."""
    config = load_config(args.config)

    if args.provider:
        config.refinement.provider = args.provider
    if args.device:
        config.hotkeys.device = args.device
    if args.debug_audio:
        from dataclasses import replace

        config = replace(config, debug=replace(config.debug, enabled=True))

    print("MicDrop — voice-to-prompt")
    print(config.summary())
    print()

    def pretty(key_name: str) -> str:
        # KEY_RIGHTCTRL -> Right Ctrl, KEY_F5 -> F5
        name = key_name.removeprefix("KEY_").replace("_", " ").title()
        return name if key_name.startswith("KEY_F") and " " not in name else name

    print(
        f"Hold {pretty(config.hotkeys.refine)} and speak to REFINE; "
        f"hold {pretty(config.hotkeys.generate)} and speak to GENERATE."
    )
    print("Press Ctrl+C in the terminal to quit.")
    print()

    pipeline, hotkeys = build_app(config)

    for mode in Mode:
        if mode is Mode.REFINE:
            label = config.hotkeys.refine
        else:
            label = config.hotkeys.generate
        print(f"  {label}: {DESCRIPTIONS[mode]}")

    print()
    if config.refinement.provider == "local_qwen":
        pipeline.processor.warmup()  # load Qwen3 (+LoRA) once, up front
    pipeline.recognizer.warmup()  # load Whisper once, up front

    hotkeys.register(
        Mode.REFINE,
        on_press=pipeline.handle_press,
        on_release=pipeline.handle_release,
    )
    hotkeys.register(
        Mode.GENERATE,
        on_press=pipeline.handle_press,
        on_release=pipeline.handle_release,
    )

    print("\nReady. Waiting for hotkeys...\n")
    try:
        hotkeys.start()
    except KeyboardInterrupt:
        print("\nExiting...")
    return 0


def build_app(config: AppConfig):
    """Wire the platform implementation to the core pipeline.

    This is the only place where Linux specifics are chosen; swapping in a
    WindowsHotkeyManager/WindowsOutputAdapter later only touches this file.
    """
    from ..audio.recorder import SoundDeviceRecorder
    from ..output.base import OutputAdapter
    from ..platform.linux.clipboard import LinuxOutputAdapter
    from ..platform.linux.keyboard import LinuxHotkeyManager
    from ..refinement import create_processor
    from ..transcription.whisper import WhisperRecognizer

    report_output_readiness(config)

    recorder = SoundDeviceRecorder(config.audio)
    recognizer = WhisperRecognizer(config.refinement)
    processor = create_processor(config)
    output: OutputAdapter = LinuxOutputAdapter(config.output)
    hotkeys = LinuxHotkeyManager(config.hotkeys)

    from ..app.pipeline import Pipeline

    pipeline = Pipeline(
        recorder=recorder,
        recognizer=recognizer,
        processor=processor,
        output=output,
        config=config,
    )
    return pipeline, hotkeys


def report_output_readiness(config: AppConfig) -> None:
    """Report clipboard/paste tooling problems once, at startup."""
    try:
        from ..platform.linux.clipboard import report_dependencies

        if not report_dependencies():
            print(
                "Transcribed text will not be pasted automatically "
                "until the problems above are fixed.\n"
            )
    except ImportError:
        print("Platform clipboard module not found; automatic paste disabled.\n")


def cmd_self_test() -> int:
    """The original clipboard_paste.py self-test, unchanged behavior."""
    from ..platform.linux.clipboard import copy_and_paste, report_dependencies

    if not report_dependencies():
        return 1

    demo = "Clipboard test from micdrop self-test"
    print(f'Copying "{demo}" and pasting at cursor...')
    if copy_and_paste(demo):
        print("Self-test OK. Check the focused window for the text.")
        return 0
    return 1


def cmd_record_test(args) -> int:
    """The original record.py microphone test."""
    try:
        import numpy as np
        import sounddevice as sd
        import soundfile as sf
    except (ImportError, OSError) as exc:
        print(f"ERROR: audio stack unavailable: {exc}", file=sys.stderr)
        return 1

    config = load_config(args.config)
    sample_rate = config.audio.sample_rate

    print("Recording...")
    print("Speak normally into your microphone.")

    audio = sd.rec(
        int(args.duration * sample_rate),
        samplerate=sample_rate,
        channels=config.audio.channels,
        dtype=config.audio.dtype,
    )
    sd.wait()

    rms = float(np.sqrt(np.mean(audio**2)))
    peak = float(np.max(np.abs(audio)))

    print("\nRecording finished.")
    print(f"RMS level:  {rms:.6f}")
    print(f"Peak level: {peak:.6f}")

    if rms < 0.001:
        print("WARNING: Audio level is extremely low.")
    elif rms < 0.01:
        print("WARNING: Audio level is quite low.")
    else:
        print("Audio level looks reasonable.")

    sf.write(args.output, audio, sample_rate, subtype="PCM_16")
    print(f"Saved: {args.output}")
    return 0


def cmd_transcribe_test(args) -> int:
    """The original transcribe.py Whisper test."""
    from ..app.config import RefinementSettings
    from ..transcription.whisper import WhisperRecognizer

    config = load_config(args.config)
    recognizer = WhisperRecognizer(config.refinement or RefinementSettings())
    print(f"Transcribing {args.file} with {recognizer.model_name}...")
    text = recognizer.transcribe_file(args.file)
    print("\nTRANSCRIPTION:")
    print(text)
    return 0


def cmd_refine_test(args) -> int:
    """Run the configured TextProcessor on a sentence, for one mode."""
    from ..refinement import create_processor

    config = load_config(args.config)
    mode = parse_mode(args.mode)
    text = " ".join(args.text).strip() or (
        "um can you write a a short poem about the the sea"
    )

    processor = create_processor(config)
    print(f"Provider: {processor.name}")
    print(f"Mode:     {mode.value.upper()}")
    print(f"Input:    {text}")
    print("\nProcessing...\n")

    result = processor.process(text, mode)
    print("OUTPUT:")
    print(result or "(empty)")
    return 0


def cmd_doctor(args) -> int:
    """Check every component; prints a report, never raises."""
    config = load_config(args.config)
    ok = True

    print("MicDrop doctor\n")

    # Python packages
    for mod in ("numpy", "sounddevice", "soundfile", "transformers", "torch", "peft"):
        try:
            __import__(mod)
            print(f"  [ok]   python package: {mod}")
        except (ImportError, OSError) as exc:
            ok = False
            print(f"  [FAIL] python package: {mod} ({exc})")

    try:
        import evdev  # noqa: F401

        print("  [ok]   python package: evdev")
    except ImportError:
        ok = False
        print("  [FAIL] python package: evdev (required on Linux)")

    # Audio input devices
    try:
        import sounddevice as sd

        inputs = [d for d in sd.query_devices() if d.get("max_input_channels", 0) > 0]
        if inputs:
            print(f"  [ok]   audio: {len(inputs)} input device(s) available")
        else:
            ok = False
            print("  [FAIL] audio: no input devices found")
    except Exception as exc:  # noqa: BLE001
        ok = False
        print(f"  [FAIL] audio: {exc}")

    # Keyboard device + hotkeys
    try:
        from ..platform.linux.keyboard import find_keyboard_device, resolve_key_code

        required = [resolve_key_code(config.hotkeys.refine), resolve_key_code(config.hotkeys.generate)]
        configured = config.hotkeys.device
        found = find_keyboard_device(required)
        try:
            import evdev

            evdev.InputDevice(configured)
            print(f"  [ok]   keyboard: configured device {configured} is readable")
        except Exception:  # noqa: BLE001
            if found:
                print(f"  [warn] keyboard: {configured} not readable; auto-detect would use {found}")
            else:
                ok = False
                print(f"  [FAIL] keyboard: {configured} not readable and no fallback found")
    except Exception as exc:  # noqa: BLE001
        ok = False
        print(f"  [FAIL] keyboard: {exc}")

    # Clipboard/paste tooling
    try:
        from ..platform.linux.clipboard import check_dependencies

        problems = check_dependencies()
        if problems:
            ok = False
            for problem in problems:
                print(f"  [FAIL] clipboard: {problem}")
        else:
            print("  [ok]   clipboard: wl-copy and ydotoold available")
    except Exception as exc:  # noqa: BLE001
        ok = False
        print(f"  [FAIL] clipboard: {exc}")

    # Model caches (informational — downloading happens on first use)
    try:
        from pathlib import Path

        hub = Path.home() / ".cache" / "huggingface" / "hub"
        for model in (
            config.refinement.whisper_model.replace("/", "--"),
            config.refinement.model_path.replace("/", "--"),
        ):
            if (hub / f"models--{model}").is_dir():
                print(f"  [ok]   model cached: {model}")
            else:
                print(f"  [info] model not cached (will download on first use): {model}")
    except Exception:  # noqa: BLE001
        pass

    # Config file
    cfg_path = find_config_file(args.config)
    if cfg_path:
        print(f"  [ok]   config file: {cfg_path}")
    else:
        print("  [info] no config file found; using defaults")

    print()
    print("Doctor result: " + ("OK" if ok else "PROBLEMS FOUND"))
    return 0 if ok else 1


def cmd_hotkey_test(args) -> int:
    """Round-trip both configured hotkeys through the real evdev stack.

    Listens on a virtual uinput device (ydotoold's) and injects the
    configured REFINE/GENERATE hotkeys with ydotool, then verifies that
    press/release events were dispatched to the right modes. No models are
    loaded and nothing is pasted, so it is safe to run anytime.
    """
    import time

    from ..platform.linux.clipboard import check_dependencies
    from ..platform.linux.keyboard import (
        LinuxHotkeyManager,
        find_all_virtual_devices,
        resolve_key_code,
    )

    problems = check_dependencies()
    if problems:
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)
        return 1

    config = load_config(args.config)
    keys = [resolve_key_code(config.hotkeys.refine), resolve_key_code(config.hotkeys.generate)]
    virtuals = find_all_virtual_devices(keys)
    if not virtuals:
        print(
            "No virtual uinput device with the hotkey keys found "
            "(is ydotoold running?). Cannot synthesize key events.",
            file=sys.stderr,
        )
        return 1

    # ydotoold may expose several virtual devices; injected events surface
    # on one of them, so listen on all of them.
    from dataclasses import replace as dc_replace

    events: list[str] = []

    def on_press(mode: Mode) -> None:
        events.append(f"press:{mode.value}")

    def on_release(mode: Mode) -> None:
        events.append(f"release:{mode.value}")

    managers = []
    for path in virtuals:
        settings = dc_replace(config.hotkeys, device=path)
        manager = LinuxHotkeyManager(settings)
        manager.register(Mode.REFINE, on_press, on_release)
        manager.register(Mode.GENERATE, on_press, on_release)
        manager.start_in_background()
        managers.append(manager)
        print(f"Listening on virtual device {path}")
    time.sleep(0.5)

    import subprocess

    def inject(code: int) -> None:
        subprocess.run(["ydotool", "key", f"{code}:1"], check=False)
        time.sleep(0.25)
        subprocess.run(["ydotool", "key", f"{code}:0"], check=False)
        time.sleep(0.5)

    print(f"Injecting {config.hotkeys.refine} (REFINE hotkey) via ydotool...")
    inject(resolve_key_code(config.hotkeys.refine))
    print(f"Injecting {config.hotkeys.generate} (GENERATE hotkey) via ydotool...")
    inject(resolve_key_code(config.hotkeys.generate))
    time.sleep(0.5)
    for manager in managers:
        manager.stop()

    expected = [
        f"press:{Mode.REFINE.value}",
        f"release:{Mode.REFINE.value}",
        f"press:{Mode.GENERATE.value}",
        f"release:{Mode.GENERATE.value}",
    ]
    ok = events == expected
    print()
    print(f"Events dispatched: {events}")
    print("Hotkey test: " + ("OK" if ok else f"EXPECTED {expected}"))
    return 0 if ok else 1


def cmd_config(args) -> int:
    if args.example:
        print(default_config_example())
        return 0

    config = load_config(args.config)
    path = find_config_file(args.config)
    print(f"Config file: {path or '(none found — using defaults)'}")
    print()
    print(config.summary())
    print()
    print("Groq placeholder: enabled =", config.groq.enabled, end="")
    key = config.groq_api_key()
    print(
        f", API key: {'set in $' + config.groq.api_key_env if key else 'not set in $' + config.groq.api_key_env}"
    )
    print("(API keys are only read from the environment, never stored.)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
