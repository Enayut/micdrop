"""MicDrop — hold a hotkey, speak, release; get clean text or a finished artifact.

Two modes:

    REFINE   clean up the words you said, preserving meaning exactly
    GENERATE carry out your instruction and produce the requested result
"""

#: Read from installed package metadata so it cannot drift from pyproject.toml.
#: Falls back for a source checkout that has not been installed.
try:
    from importlib.metadata import PackageNotFoundError, version as _version

    __version__ = _version("micdrop")
except Exception:  # noqa: BLE001 - not installed, or metadata missing
    __version__ = "0.3.0"

__all__ = ["__version__"]