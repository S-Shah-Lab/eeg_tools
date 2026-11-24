"""Channel identity for analysis of an already preprocessed MNE FIF file."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import mne
import numpy as np


MONTAGE_HELPER_FILES = {
    "DSI_24": "DSI24_location.txt",
    "GTEC_32": "GTEC32_location.txt",
    "EGI_64": "EGI64_location.txt",
    "EGI_128": "EGI128_location.txt",
}


def eeg_channel_names(raw: mne.io.BaseRaw) -> list[str]:
    """Read EEG names in their actual data-row order, including marked bads."""
    names = [raw.ch_names[i] for i in mne.pick_types(raw.info, eeg=True, meg=False, exclude=())]
    if not names:
        raise ValueError("Cleaned Raw contains no EEG channels; check MNE channel types")
    normalized = [name.lower() for name in names]
    if len(set(normalized)) != len(names) or any(not name.strip() for name in names):
        raise ValueError(f"Cleaned Raw contains empty or ambiguous EEG channel names: {names}")
    return names


class CleanedEEGChannelSet:
    """Read-only label lookup for already processed EEG, not a spatial filter."""

    def __init__(self, raw: mne.io.BaseRaw) -> None:
        names = eeg_channel_names(raw)
        self._labels = tuple(name.lower() for name in names)
        # Use the saved electrode geometry, never substitute the acquisition helper
        montage = raw.get_montage()
        if montage is None:
            raise ValueError("Cleaned Raw has no saved electrode montage; restore the verified montage before analysis")
        positions = montage.get_positions()["ch_pos"]
        missing = [name for name in names if name not in positions
                   or not np.isfinite(positions[name]).all()]
        if missing:
            raise ValueError(f"Cleaned Raw is missing valid electrode positions for: {', '.join(missing)}. "
                             "Restore the verified montage before analysis")

    def get_labels(self) -> list[str]:
        return list(self._labels)

    def find_labels(self, names: str | Iterable[str]) -> np.ndarray:
        if isinstance(names, str):
            names = names.split()
        targets = {name.lower() for name in names}
        return np.asarray([i for i, label in enumerate(self._labels) if label in targets], dtype=int)


def infer_cleaned_montage_type(raw: mne.io.BaseRaw, helper_dir: Path,
                               override: str | None = None, metadata: dict | None = None) -> str | None:
    """Identify hardware for reporting only; never use it to relabel the EEG."""
    if override:
        if override not in MONTAGE_HELPER_FILES:
            raise ValueError(f"Unknown --montage-type {override!r}; choose from {', '.join(MONTAGE_HELPER_FILES)}")
        return override
    recorded = (metadata or {}).get("montage_type")
    if recorded and str(recorded).strip().lower() not in {"n/a", "unknown", "none"}:
        return str(recorded)

    actual = {name.lower() for name in eeg_channel_names(raw)}
    exact, subsets = [], []
    for montage_type, filename in MONTAGE_HELPER_FILES.items():
        path = helper_dir / filename
        if not path.is_file():
            continue
        # Reference and ground rows beginning with '-' and '@' are not EEG signals
        labels = {line.split()[0].lower() for line in path.read_text().splitlines()
                  if line.strip() and not line.lstrip().startswith(("-", "@", "#"))}
        if actual == labels:
            exact.append(montage_type)
        elif actual < labels:
            subsets.append(montage_type)
    if len(exact) == 1:
        return exact[0]
    if not exact and len(subsets) == 1:
        return subsets[0]
    return None
