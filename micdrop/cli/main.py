"""MicDrop CLI.

This is the whole user-facing surface. During normal use you start MicDrop once
and never touch the CLI again — it runs in the background waiting for hotkeys.

Commands, roughly in the order you need them:

    micdrop setup      first run: check everything, create a config file
    micdrop doctor     check everything again, non-interactive
    micdrop run        start the hotkey listener (this is the app)
    micdrop status     what MicDrop is configured to do right now
    micdrop config     show / print the configuration
    micdrop test       run the hardware self-tests
    micdrop models     which models and providers are available

Advanced/manual helpers, kept because they are genuinely useful when something
is wrong:

    micdrop record-test, transcribe-test, refine-test, hotkey-test, self-test
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import replace

from .. import __version__
from ..app.config import (
    AppConfig,
    config_path_for_write,
    default_config_example,
    find_config_file,
    load_config,
)
from ..diagnostics import (
    DiagnosticReport,
    Status,
    check_config_file,
    check_local_models,
    run_full_diagnostics,
)
from ..app.errors import MicDropError
from ..app.modes import DESCRIPTIONS, DISPLAY_NAMES, HOTKEY_KEYS, Mode, parse_mode

logger = logging.getLogger("micdrop")


# --------------------------------------------------------------------------- #
# Parser
# --------------------------------------------------------------------------- #


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="micdrop",
        description=(
            "Hold a hotkey, speak, release. Your words are cleaned up (REFINE) "
            "or turned into the thing you asked for (GENERATE), then pasted."
        ),
        epilog=(
            "Right Ctrl = REFINE, Right Alt = GENERATE by default.\n"
            "Start with:  micdrop setup    then    micdrop run"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--version", action="version", version=f"micdrop {__version__}")
    parser.add_argument(
        "--config", "-c", help="Path to a micdrop.toml config file", default=None
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Verbose (debug) logging"
    )

    sub = parser.add_subparsers(dest="command", metavar="<command>")

    # -- the ones a user actually needs ---------------------------------- #
    sub.add_parser(
        "setup",
        help="Check dependencies and create a config file (first run)",
    )

    run = sub.add_parser("run", help="Start the hotkey listener (the app)")
    run.add_argument(
        "--refine-provider",
        help="Override the REFINE provider for this run (local_qwen|groq|none)",
        default=None,
    )
    run.add_argument(
        "--generate-provider",
        help="Override the GENERATE provider for this run (local_qwen|groq|none)",
        default=None,
    )
    run.add_argument("--provider", help="Override both providers", default=None)
    run.add_argument("--device", help="Override keyboard device path", default=None)
    run.add_argument(
        "--debug-audio",
        action="store_true",
        help="Save recordings and transcripts to debug/ (off by default)",
    )

    sub.add_parser("doctor", help="Report every dependency (non-interactive)")

    sub.add_parser("status", help="Show what MicDrop is configured to do")

    cfg = sub.add_parser("config", help="Show or print the configuration")
    cfg.add_argument("--example", action="store_true", help="Print an example config file")
    cfg.add_argument("--path", action="store_true", help="Print the config file path")

    sub.add_parser("test", help="Run the hardware self-tests")

    models = sub.add_parser("models", help="Show models and provider availability")
    models.add_argument(
        "--download",
        nargs="?",
        const="__all__",
        default=None,
        metavar="MODEL",
        help="Pre-download a model so the first hotkey press is not slow",
    )

    # -- troubleshooting helpers ------------------------------------------ #
    sub.add_parser("self-test", help="[test] clipboard/paste check + demo paste")

    rec = sub.add_parser("record-test", help="[test] record N seconds, report levels")
    rec.add_argument("--duration", type=float, default=10.0)
    rec.add_argument("--output", default="test.wav")

    tr = sub.add_parser("transcribe-test", help="[test] transcribe a WAV file with Whisper")
    tr.add_argument("file", nargs="?", default="test.wav")

    rf = sub.add_parser("refine-test", help="[test] run one mode's processor on a sentence")
    rf.add_argument("text", nargs="*", help="Text to process")
    rf.add_argument("--mode", default="refine", help="refine or generate (default: refine)")

    sub.add_parser(
        "hotkey-test",
        help="[test] round-trip both hotkeys through evdev using ydotool",
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(message)s",
    )

    # Commands that must stay silent unless asked (tests, status, models) get
    # WARNING-level logging so a stray warning line cannot corrupt the output.
    if args.command in ("status", "models", "test", "record-test", "transcribe-test"):
        logging.getLogger("micdrop").setLevel(logging.WARNING)

    try:
        return _dispatch(args)
    except MicDropError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nExiting.")
        return 0


def _dispatch(args) -> int:
    command = args.command
    if command == "setup":
        return cmd_setup(args)
    if command == "doctor":
        return cmd_doctor(args)
    if command == "status":
        return cmd_status(args)
    if command == "config":
        return cmd_config(args)
    if command == "models":
        return cmd_models(args)
    if command == "test":
        return cmd_test(args)
    if command == "run":
        return cmd_run(args)
    # Troubleshooting helpers.
    if command == "self-test":
        return cmd_self_test()
    if command == "record-test":
        return cmd_record_test(args)
    if command == "transcribe-test":
        return cmd_transcribe_test(args)
    if command == "refine-test":
        return cmd_refine_test(args)
    if command == "hotkey-test":
        return cmd_hotkey_test(args)
    # No subcommand: default to run, so `micdrop` alone starts the app.
    return cmd_run(args)


# --------------------------------------------------------------------------- #
# Wiring
# --------------------------------------------------------------------------- #


def apply_run_overrides(config: AppConfig, args) -> AppConfig:
    """Apply CLI overrides for providers, device and debug mode."""
    provider_overrides = {}
    if getattr(args, "provider", None):
        provider_overrides["refine_provider"] = args.provider
        provider_overrides["generate_provider"] = args.provider
    if getattr(args, "refine_provider", None):
        provider_overrides["refine_provider"] = args.refine_provider
    if getattr(args, "generate_provider", None):
        provider_overrides["generate_provider"] = args.generate_provider
    if provider_overrides:
        config = replace(config, models=replace(config.models, **provider_overrides))

    if getattr(args, "device", None):
        config = replace(config, hotkeys=replace(config.hotkeys, device=args.device))
    if getattr(args, "debug_audio", False):
        config = replace(config, debug=replace(config.debug, enabled=True))
    return config


def build_app(config: AppConfig):
    """Wire the selected platform backend to the core pipeline.

    The platform decision happens once, here, via the platform boundary
    (``micdrop.platform.select_platform``). Nothing else in the app branches on
    the operating system.
    """
    from ..audio.recorder import SoundDeviceRecorder
    from ..platform import select_platform
    from ..refinement import create_router
    from ..transcription.whisper import WhisperRecognizer

    selection = select_platform()
    recorder = SoundDeviceRecorder(config.audio)
    recognizer = WhisperRecognizer(config.refinement)
    # One processor per mode; the router dispatches strictly by mode.
    router = create_router(config)
    output = selection.create_output(config)
    hotkeys = selection.create_hotkeys(config)

    from ..app.pipeline import Pipeline

    pipeline = Pipeline(
        recorder=recorder,
        recognizer=recognizer,
        processor=router,
        output=output,
        config=config,
    )
    return pipeline, hotkeys, selection


# --------------------------------------------------------------------------- #
# run
# --------------------------------------------------------------------------- #


def cmd_run(args) -> int:
    """Start the hotkey listener and stay idle until the user quits."""
    config = apply_run_overrides(load_config(args.config), args)

    from .banner import print_banner

    pipeline, hotkeys, selection = build_app(config)
    print_banner(config)

    for mode in Mode:
        hotkeys.register(
            mode,
            on_press=pipeline.handle_press,
            on_release=pipeline.handle_release,
        )

    print("\nReady. Waiting for hotkeys...\n")

    # Load everything once, before listening, so no hotkey press ever pays the
    # model-loading cost.
    #
    # Whisper failing is fatal (nothing works without it). A *mode* whose
    # provider is unavailable is not: the other mode keeps working, and this
    # one reports the same clear error on every press. That also guarantees we
    # never quietly substitute a different provider.
    try:
        from ..refinement import warmup_application

        failures = warmup_application(pipeline)
    except MicDropError as exc:
        print(f"\nERROR: could not prepare MicDrop: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # noqa: BLE001
        print(f"\nERROR: could not prepare MicDrop: {exc}", file=sys.stderr)
        return 1

    if failures:
        print("\nSome modes could not be prepared:", file=sys.stderr)
        for mode in sorted(failures, key=lambda m: m.value):
            print(f"\n  {mode.value.upper()} is unavailable:\n    {failures[mode]}",
                  file=sys.stderr)
        working = [m for m in Mode if m not in failures]
        if working:
            print(
                "\n  Still working: "
                + ", ".join(m.value.upper() for m in working)
                + ".",
                file=sys.stderr,
            )
        print("", file=sys.stderr)

    print("Loaded. MicDrop is listening. Press Ctrl+C here to quit.\n")
    try:
        hotkeys.start()
    except KeyboardInterrupt:
        print("\nExiting.")
    return 0


# --------------------------------------------------------------------------- #
# setup / doctor / status
# --------------------------------------------------------------------------- #


def cmd_setup(args) -> int:
    """First-run checklist. Creates a config file; changes nothing else."""
    config = load_config(args.config)
    report = run_full_diagnostics(config)

    print("MicDrop Setup\n")
    print("Checking everything MicDrop needs...\n")

    from .diagnostics import fixes_for, render_checklist

    print(render_checklist(report))

    # Only ever write the config file, and say so before doing it.
    existing = find_config_file(args.config)
    if existing is not None:
        print(f"\nUsing configuration file: {existing}")
    else:
        target = config_path_for_write()
        print(f"\nNo configuration file found. Creating one: {target}")
        try:
            target.write_text(default_config_example(), encoding="utf-8")
        except OSError as exc:
            print(f"  could not write it: {exc}", file=sys.stderr)
        else:
            print("  All values are the defaults; edit the file to change them.")
            # Re-check so the report no longer tells the user to create the file
            # that was just created.
            report.checks = [c for c in report.checks if c.name != "Config file"]
            report.add(check_config_file())

    print()
    if report.ready:
        if report.warnings:
            print("MicDrop is ready, with warnings.")
            _print_fixes(report)
            print("\nRun 'micdrop run' to start.")
        else:
            print("MicDrop is ready. Run 'micdrop run' to start.")
        return 0

    print("MicDrop is not ready yet. Fix the problems above, then run 'micdrop setup' again.\n")
    _print_fixes(report)
    return 1


def _print_fixes(report: DiagnosticReport) -> None:
    from .diagnostics import fixes_for

    text = fixes_for(report)
    if text:
        print("\nWhat to do:")
        print(text)


def cmd_doctor(args) -> int:
    """Report every dependency. Non-interactive, never changes anything."""
    config = load_config(args.config)
    report = run_full_diagnostics(config)

    from .diagnostics import overall_line, render_table

    print("MicDrop doctor\n")
    print(render_table(report))
    print()
    print(overall_line(report))

    failures = report.failures
    if failures:
        print()
        for check in failures:
            print(f"{check.name}: {check.detail}")
            if check.fix:
                print(f"    {check.fix}")

    print("\nRun 'micdrop setup' for a guided version.")
    return 0 if report.ready else 1


def cmd_status(args) -> int:
    """What MicDrop is configured to do right now."""
    config = load_config(args.config)
    path = find_config_file(args.config)

    print("MicDrop status\n")
    print(f"Config file:    {path or '(none — using defaults)'}")
    print(f"Platform:       {__platform_summary()}")
    print(f"Debug mode:     {'ON (audio and transcripts saved to ' + config.debug.debug_dir + ')' if config.debug.enabled else 'off (audio memory-only, transcripts not logged)'}")
    print()
    print("Modes")
    print(f"  {HOTKEY_KEYS[Mode.REFINE]} hotkey    {config.hotkeys.key_for(Mode.REFINE)}")
    print(f"  {HOTKEY_KEYS[Mode.GENERATE]} hotkey  {config.hotkeys.key_for(Mode.GENERATE)}")
    print(f"  keyboard       {config.hotkeys.device or '(auto-detect)'}")
    print(f"  max recording  {config.audio.max_duration:.0f}s")
    print()
    print("Text processing")
    for mode in Mode:
        provider = config.provider_for(mode)
        model = _model_id_for(config, mode, provider)
        print(
            f"  {DISPLAY_NAMES[mode]:8s} provider={provider:11s} "
            f"max_tokens={config.max_new_tokens_for(mode):4d}  model={model}"
        )
        print(f"           {DESCRIPTIONS[mode]}")
    print()
    print("Speech to text")
    print(f"  {config.refinement.whisper_model} ({config.refinement.language})")
    print()
    print("Providers")
    print(f"  groq api key   {'set' if config.groq_api_key() else 'not set'} (${config.groq.api_key_env})")
    return 0


def _model_id_for(config: AppConfig, mode: Mode, provider: str) -> str:
    if provider == "local_qwen":
        return config.local.model_for(mode)
    if provider == "groq":
        return config.groq.model_for(mode)
    return "(none)"


def __platform_summary() -> str:
    from ..platform import PLATFORM_REGISTRY, current_platform

    key = current_platform()
    return key if key in PLATFORM_REGISTRY else f"{key} (unsupported)"


# --------------------------------------------------------------------------- #
# config / models / test
# --------------------------------------------------------------------------- #


def cmd_config(args) -> int:
    if args.example:
        print(default_config_example())
        return 0

    if args.path:
        print(find_config_file(args.config) or "(no config file; using defaults)")
        return 0

    config = load_config(args.config)
    path = find_config_file(args.config)
    print(f"Config file: {path or '(none found — using defaults)'}")
    print()
    print(config.summary())
    return 0


def cmd_models(args) -> int:
    """Show models and provider availability, optionally pre-downloading."""
    from ..refinement import PROVIDERS, PROVIDER_DESCRIPTIONS

    config = load_config(args.config)

    if args.download:
        return _download_models(config, args.download)

    print("MicDrop models\n")
    print("Configured for each mode:")
    for mode in Mode:
        provider = config.provider_for(mode)
        print(f"  {DISPLAY_NAMES[mode]:8s} provider={provider:11s} model={_model_id_for(config, mode, provider)}")
    print()
    print("Available providers:")
    for name in sorted(PROVIDERS):
        print(f"  {name:11s} {PROVIDER_DESCRIPTIONS.get(name, '')}")
    print()
    print("Model cache:")
    for mode in Mode:
        provider = config.provider_for(mode)
        if provider != "local_qwen":
            print(f"  {DISPLAY_NAMES[mode]:8s} (cloud: {provider}, nothing local to cache)")
            continue
        model = config.local.model_for(mode)
        from ..diagnostics.base import _huggingface_cached

        state = "cached" if _huggingface_cached(model) else "not downloaded"
        print(f"  {DISPLAY_NAMES[mode]:8s} {model} [{state}]")
    print()
    print(f"Whisper         {config.refinement.whisper_model}")
    print()
    print("To pre-download so the first hotkey press is fast:")
    print("  micdrop models --download")
    return 0


def _download_models(config: AppConfig, target: str) -> int:
    """Pre-fetch models into the local cache. Explicit user action only."""
    wanted: list[str] = [config.refinement.whisper_model]
    if target != "__all__":
        wanted = [target]

    for repo in wanted:
        print(f"Downloading {repo} ...")
        try:
            from huggingface_hub import snapshot_download

            snapshot_download(repo)
        except ImportError:
            try:
                from transformers import AutoModelForCausalLM, AutoTokenizer

                AutoTokenizer.from_pretrained(repo)
                AutoModelForCausalLM.from_pretrained(repo)
            except ImportError:
                print("  ERROR: install huggingface_hub or transformers first.")
                return 1
            except Exception as exc:  # noqa: BLE001
                print(f"  ERROR: {exc}")
                return 1
        except Exception as exc:  # noqa: BLE001
            print(f"  ERROR: {exc}")
            return 1
        print(f"  done: {repo}")
    return 0


def cmd_test(args) -> int:
    """Run the hardware self-tests. Safe to run any time."""
    print("MicDrop self-test\n")
    results: list[tuple[str, bool, str]] = []

    def record(name: str, fn) -> None:
        try:
            ok, note = fn()
        except Exception as exc:  # noqa: BLE001
            ok, note = False, str(exc)
        results.append((name, ok, note))

    record("Clipboard / paste", _clipboard_check)
    record("Microphone", _record_check)
    record("Hotkeys (evdev + ydotool)", _hotkey_check)

    print()
    width = max(len(n) for n, _, _ in results)
    for name, ok, note in results:
        print(f"  {'OK  ' if ok else 'FAIL'}  {name.ljust(width)}  {note}")
    print()
    if all(ok for _, ok, _ in results):
        print("All self-tests passed. Run 'micdrop run' to start.")
        return 0
    print("Some self-tests failed. Run 'micdrop doctor' for details.")
    return 1


def _clipboard_check() -> tuple[bool, str]:
    """wl-copy + ydotool + daemon available."""
    try:
        from ..platform.linux.clipboard import check_dependencies

        problems = check_dependencies()
    except Exception as exc:  # noqa: BLE001
        return False, f"could not check: {exc}"
    return (not problems), ("wl-copy + ydotool" if not problems else problems[0])


def _record_check() -> tuple[bool, str]:
    """At least one audio input device."""
    try:
        import sounddevice as sd

        inputs = [d for d in sd.query_devices() if d.get("max_input_channels", 0) > 0]
    except Exception as exc:  # noqa: BLE001
        return False, f"could not query audio: {exc}"
    if not inputs:
        return False, "no input device found"
    return True, f"{len(inputs)} input device(s)"


def _hotkey_check() -> tuple[bool, str]:
    """A virtual uinput device exists to round-trip the hotkeys through."""
    try:
        from ..platform.linux.clipboard import check_dependencies
        from ..platform.linux.keyboard import find_all_virtual_devices, resolve_key_code

        if check_dependencies():
            return False, "clipboard tooling missing"
        keys = [resolve_key_code(config_hotkeys().key_for(mode)) for mode in Mode]
    except Exception as exc:  # noqa: BLE001
        return False, f"could not check: {exc}"
    if find_all_virtual_devices(keys):
        return True, "virtual device ready"
    return False, "ydotoold not running (no virtual device); run: hotkey-test for detail"


def config_hotkeys():
    """The configured hotkeys, for the test probes."""
    return load_config().hotkeys


# --------------------------------------------------------------------------- #
# Troubleshooting helpers (unchanged behavior)
# --------------------------------------------------------------------------- #


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
    """Run the configured TextProcessor for one mode on a sentence.

    Useful for eyeballing what each mode actually does: the same input through
    --mode refine and --mode generate should look very different.
    """
    from ..refinement import create_processor

    config = load_config(args.config)
    mode = parse_mode(args.mode)
    text = " ".join(args.text).strip() or (
        "um can you write a a short poem about the the sea"
    )

    processor = create_processor(config, mode)
    print(f"Provider:  {config.provider_for(mode)} ({processor.name})")
    print(f"Mode:      {DISPLAY_NAMES[mode]}")
    print(f"Behaviour: {DESCRIPTIONS[mode]}")
    print(f"Input:     {text}")
    print("\nProcessing...\n")

    result = processor.process(text, mode)
    print("OUTPUT:")
    print(result or "(empty)")
    return 0


def cmd_hotkey_test(args) -> int:
    """Round-trip both configured hotkeys through the real evdev stack.

    Listens on a virtual uinput device (ydotoold's) and injects the configured
    REFINE/GENERATE hotkeys with ydotool, then verifies that press/release
    events were dispatched to the right modes. No models are loaded and nothing
    is pasted, so it is safe to run anytime.
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
    keys = [resolve_key_code(config.hotkeys.key_for(mode)) for mode in Mode]
    virtuals = find_all_virtual_devices(keys)
    if not virtuals:
        print(
            "No virtual uinput device with the hotkey keys found "
            "(is ydotoold running?). Cannot synthesize key events.",
            file=sys.stderr,
        )
        return 1

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

    for mode in Mode:
        key_name = config.hotkeys.key_for(mode)
        print(f"Injecting {key_name} ({DISPLAY_NAMES[mode]} hotkey) via ydotool...")
        inject(resolve_key_code(key_name))
    time.sleep(0.5)
    for manager in managers:
        manager.stop()

    expected = []
    for mode in Mode:
        expected.append(f"press:{mode.value}")
        expected.append(f"release:{mode.value}")
    ok = events == expected
    print()
    print(f"Events dispatched: {events}")
    print("Hotkey test: " + ("OK" if ok else f"EXPECTED {expected}"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())