"""
Standalone EEG preprocessing utilities for MNE Raw objects.

The preprocessor executes the steps in the exact order provided by
EEGPreprocessorConfig. The command-line wrapper builds the default order as:
notch, band-pass, PREP, annotation, interpolation, re-reference, spatial filter.
"""

from __future__ import annotations

from collections import OrderedDict
from time import perf_counter
from typing import Any, Callable, Iterable, Iterator, Literal, Mapping, Optional, get_args

import mne
import numpy as np
from BCI2000Tools.Electrodes import ChannelSet
from mne.io import BaseRaw
from pyprep.find_noisy_channels import NoisyChannels

import helper.eeg_dict as eeg_dict


StepName = Literal[
    "rereference",
    "spatialfilter",
    "notch",
    "bandpass",
    "prep",
    "interpolation",
    "annotation",
]

VALID_STEPS: set[str] = set(get_args(StepName))
STEP_LABELS: dict[str, str] = {
    "notch": "notch filter",
    "bandpass": "band-pass filter",
    "prep": "PREP noisy-channel detection",
    "annotation": "manual BAD-region annotation",
    "interpolation": "bad-channel interpolation",
    "rereference": "re-reference",
    "spatialfilter": "spatial filter",
}


class EEGPreprocessorConfig(OrderedDict[StepName, dict[str, Any]]):
    """Ordered mapping of preprocessing steps to parameter dictionaries."""

    def __init__(
        self,
        steps: Optional[Iterable[tuple[StepName, Mapping[str, Any]]]] = None,
    ) -> None:
        super().__init__()
        if steps is not None:
            for name, params in steps:
                self.add_step(name, params)

    def add_step(self, name: StepName, params: Mapping[str, Any]) -> None:
        """Add a preprocessing step and preserve insertion order."""
        if name not in VALID_STEPS:
            valid = ", ".join(sorted(VALID_STEPS))
            raise KeyError(f"Unknown preprocessing step {name!r}. Valid steps are: {valid}")
        self[name] = dict(params)

    def iter_steps(self) -> Iterator[tuple[StepName, dict[str, Any]]]:
        """Yield steps in execution order."""
        yield from self.items()


class EEGPreprocessor:
    """
    High-level preprocessing wrapper for an MNE Raw object.

    The class operates directly on an imported Raw object and preserves non-EEG
    channels when re-referencing or spatial filtering changes the EEG matrix.
    """

    def __init__(
        self,
        raw: BaseRaw,
        ch_set: ChannelSet,
        config: Optional[EEGPreprocessorConfig | Mapping[StepName, Mapping[str, Any]]] = None,
        copy: bool = True,
        montage_type: Optional[str] = None,
        conv_dict: Optional[Mapping[str, str]] = None,
        verbose: bool = False,
        feedback: bool = True,
    ) -> None:
        """Initialize the preprocessor with an MNE Raw object and configuration."""
        if not isinstance(raw, BaseRaw):
            raise TypeError("`raw` must be an instance of mne.io.BaseRaw")

        self.raw: BaseRaw = raw.copy() if copy else raw
        self.raw.load_data()
        self.ch_set = ch_set
        self.verbose = verbose
        self.feedback = feedback

        self._montage_type = montage_type
        self._conv_dict = conv_dict or eeg_dict.stand1020_to_egi
        self._orig_ch_pos: dict[str, np.ndarray] = {}
        self._orig_coord_frame = "head"

        self.config = self._coerce_config(config)
        self.history: dict[str, Any] = {"step_log": []}

        self._step_dispatch: dict[StepName, Callable[[Mapping[str, Any]], None]] = {
            "rereference": self._apply_rereference,
            "spatialfilter": self._apply_spatialfilter,
            "notch": self._apply_notch_filter,
            "bandpass": self._apply_bandpass_filter,
            "prep": self._apply_prep,
            "interpolation": self._apply_interpolation_bad_channels,
            "annotation": self._apply_annotation,
        }

        self._cache_original_montage(raw)
        self._eeg_picks = mne.pick_types(self.raw.info, eeg=True, exclude=())
        if self._eeg_picks.size == 0:
            raise RuntimeError("No EEG channels found in Raw.info")

        self.history["initial_bads"] = list(self.raw.info.get("bads", []))
        self.history["initial_highpass"] = self.raw.info.get("highpass", None)
        self.history["initial_lowpass"] = self.raw.info.get("lowpass", None)
        self.history["initial_n_eeg"] = int(self._eeg_picks.size)
        self.history["sfreq"] = float(self.raw.info["sfreq"])

    def run(self) -> tuple[BaseRaw, dict[str, Any]]:
        """Run preprocessing steps in the configured order."""
        active_steps = [
            (step_name, params)
            for step_name, params in self.config.iter_steps()
            if params is not None and params is not False
        ]

        if not active_steps:
            self._log("[EEGPreprocessor] No preprocessing steps requested")
            return self.raw, self.history

        step_names = " -> ".join(STEP_LABELS.get(name, name) for name, _ in active_steps)
        self._log(f"[EEGPreprocessor] Planned steps: {step_names}")

        pipeline_start = perf_counter()
        n_steps = len(active_steps)

        for index, (step_name, params) in enumerate(active_steps, start=1):
            if step_name not in self._step_dispatch:
                raise KeyError(f"Unknown preprocessing step: {step_name!r}")
            if not isinstance(params, Mapping):
                raise TypeError(
                    f"Parameters for step {step_name!r} must be a mapping, got {type(params)}"
                )

            label = STEP_LABELS.get(step_name, step_name)
            start = perf_counter()
            self._log(f"[EEGPreprocessor] [{index}/{n_steps}] START {label}")

            try:
                self._step_dispatch[step_name](params)
            except Exception as exc:
                elapsed = perf_counter() - start
                self.history["step_log"].append(
                    {
                        "step": step_name,
                        "label": label,
                        "status": "failed",
                        "elapsed_sec": elapsed,
                        "error": repr(exc),
                    }
                )
                self._log(
                    f"[EEGPreprocessor] [{index}/{n_steps}] FAILED {label} "
                    f"after {elapsed:.2f}s: {exc}"
                )
                raise

            elapsed = perf_counter() - start
            step_record = {
                "step": step_name,
                "label": label,
                "status": "done",
                "elapsed_sec": elapsed,
            }
            if step_name == "annotation" and bool(params.get("plot", True)):
                step_record["elapsed_note"] = "includes manual inspection time"
                self._log(f"[EEGPreprocessor] [{index}/{n_steps}] DONE {label}")
            else:
                self._log(f"[EEGPreprocessor] [{index}/{n_steps}] DONE {label} in {elapsed:.2f}s")
            self.history["step_log"].append(step_record)

        total_elapsed = perf_counter() - pipeline_start
        computational_elapsed = sum(
            float(item.get("elapsed_sec", 0.0))
            for item in self.history["step_log"]
            if item.get("step") != "annotation"
        )
        self.history["runtime_sec"] = total_elapsed
        self.history["computational_runtime_sec"] = computational_elapsed
        self.history["final_bads"] = list(self.raw.info.get("bads", []))
        self.history["final_highpass"] = self.raw.info.get("highpass", None)
        self.history["final_lowpass"] = self.raw.info.get("lowpass", None)
        self._log(f"[EEGPreprocessor] Pipeline complete, computational steps took {computational_elapsed:.2f}s")
        return self.raw, self.history

    @staticmethod
    def _coerce_config(
        config: Optional[EEGPreprocessorConfig | Mapping[StepName, Mapping[str, Any]]],
    ) -> EEGPreprocessorConfig:
        if config is None:
            return EEGPreprocessorConfig()
        if isinstance(config, EEGPreprocessorConfig):
            return config
        return EEGPreprocessorConfig(list(config.items()))

    def _log(self, message: str) -> None:
        if self.feedback:
            print(message, flush=True)

    def _cache_original_montage(self, raw: BaseRaw) -> None:
        try:
            montage = raw.get_montage()
        except Exception:
            montage = None

        if montage is None:
            return

        positions = montage.get_positions()
        self._orig_ch_pos = dict(positions.get("ch_pos", {}))
        self._orig_coord_frame = positions.get("coord_frame", "head")

    @staticmethod
    def _infer_montage_type(n_ch: int) -> Optional[str]:
        """Infer the montage type string from the number of EEG channels."""
        if n_ch in (21, 24):
            return "DSI_24"
        if n_ch == 32:
            return "GTEC_32"
        if n_ch == 64:
            return "EGI_64"
        if n_ch == 128:
            return "EGI_128"
        return None

    @staticmethod
    def _format_channels(channels: list[str], max_items: int = 12) -> str:
        if not channels:
            return "none"
        if len(channels) <= max_items:
            return ", ".join(channels)
        shown = ", ".join(channels[:max_items])
        return f"{shown}, ... (+{len(channels) - max_items} more)"

    @staticmethod
    def _format_freqs(freqs: np.ndarray) -> str:
        if freqs.size == 0:
            return "none"
        return ", ".join(f"{freq:g}" for freq in freqs)

    def _make_montage_like_importer(
        self,
        ch_names: list[str],
    ) -> Optional[mne.channels.DigMontage]:
        """Recreate the montage selection logic used by the raw importer."""
        montage_type = self._montage_type or self._infer_montage_type(len(ch_names))
        if montage_type is None:
            return None

        if montage_type in {"DSI_24", "GTEC_32"}:
            montage = mne.channels.make_standard_montage("standard_1020")
            is_egi = False
        elif montage_type in {"EGI_64", "EGI_128"}:
            montage = mne.channels.make_standard_montage("GSN-HydroCel-129")
            is_egi = True
        else:
            return None

        montage.ch_names = [name.lower() for name in montage.ch_names]
        idx: list[int] = []
        kept_names: list[str] = []

        for ch_name in ch_names:
            lookup_name = ch_name.lower()
            try:
                mapped_name = self._conv_dict[lookup_name] if is_egi else lookup_name
                idx.append(montage.ch_names.index(mapped_name))
                kept_names.append(ch_name)
            except (KeyError, ValueError):
                continue

        if not idx:
            return None

        montage.ch_names = kept_names
        montage.dig = montage.dig[:3] + [montage.dig[i + 3] for i in idx]
        return montage

    def _reapply_montage_after_channel_change(self, new_labels: list[str]) -> None:
        """Rebuild montage after channel remapping."""
        montage = self._make_montage_like_importer(new_labels)
        if montage is not None:
            self.raw.set_montage(montage, on_missing="ignore")
            return

        ch_pos = {
            name: self._orig_ch_pos[name]
            for name in new_labels
            if name in self._orig_ch_pos
        }
        if not ch_pos:
            self._log("[EEGPreprocessor] Montage update skipped: no matching positions")
            return

        new_montage = mne.channels.make_dig_montage(
            ch_pos=ch_pos,
            coord_frame=self._orig_coord_frame,
        )
        self.raw.set_montage(new_montage, on_missing="ignore")

    def _apply_notch_filter(self, params: Mapping[str, Any]) -> None:
        """Apply notch filtering to EEG channels using Raw.notch_filter."""
        notch_freqs = params.get("freqs")
        notch_kwargs: dict[str, Any] = dict(params.get("kwargs", {}))

        if notch_freqs is None:
            raise ValueError("[Notch filter] Provide at least one frequency value")

        nyquist = float(self.raw.info["sfreq"]) / 2.0
        freqs = self._expand_notch_frequencies(notch_freqs, nyquist)

        self.raw.notch_filter(
            freqs=freqs,
            picks=self._eeg_picks,
            **notch_kwargs,
            verbose=self.verbose,
        )

        self.history["notch_filter"] = {
            "freqs": freqs.tolist(),
            "kwargs": notch_kwargs.copy(),
        }
        self._log(f"[EEGPreprocessor] Notch frequencies: {self._format_freqs(freqs)} Hz")

    @staticmethod
    def _expand_notch_frequencies(notch_freqs: Any, nyquist: float) -> np.ndarray:
        if isinstance(notch_freqs, (int, float)):
            base = float(notch_freqs)
            if base <= 0:
                raise ValueError("[Notch filter] Frequency must be positive")
            freqs = np.arange(base, nyquist, base)
        else:
            freqs = np.asarray(notch_freqs, dtype=float)

        freqs = freqs[np.isfinite(freqs)]
        freqs = freqs[(freqs > 0) & (freqs < nyquist)]
        if freqs.size == 0:
            raise ValueError(
                f"[Notch filter] No valid notch frequencies below Nyquist ({nyquist:g} Hz)"
            )
        return np.unique(freqs)

    def _apply_bandpass_filter(self, params: Mapping[str, Any]) -> None:
        """Apply band-pass, high-pass, or low-pass filtering to EEG channels."""
        l_freq = params.get("l_freq")
        h_freq = params.get("h_freq")
        bandpass_kwargs: dict[str, Any] = dict(params.get("kwargs", {}))
        self._validate_bandpass(l_freq, h_freq)

        self.raw.filter(
            l_freq=l_freq,
            h_freq=h_freq,
            picks=self._eeg_picks,
            **bandpass_kwargs,
            verbose=self.verbose,
        )

        self.history["bandpass_filter"] = {
            "l_freq": l_freq,
            "h_freq": h_freq,
            "kwargs": bandpass_kwargs.copy(),
            "info_highpass": self.raw.info.get("highpass", None),
            "info_lowpass": self.raw.info.get("lowpass", None),
        }
        self._log(f"[EEGPreprocessor] Band-pass range: {l_freq} to {h_freq} Hz")

    def _validate_bandpass(self, l_freq: Any, h_freq: Any) -> None:
        if l_freq is None and h_freq is None:
            raise ValueError("[Band-pass filter] Provide l_freq, h_freq, or both")
        if l_freq is not None and float(l_freq) <= 0:
            raise ValueError("[Band-pass filter] l_freq must be positive")
        if h_freq is not None and float(h_freq) <= 0:
            raise ValueError("[Band-pass filter] h_freq must be positive")
        if l_freq is not None and h_freq is not None and float(l_freq) >= float(h_freq):
            raise ValueError("[Band-pass filter] l_freq must be lower than h_freq")
        if h_freq is not None:
            nyquist = float(self.raw.info["sfreq"]) / 2.0
            if float(h_freq) >= nyquist:
                raise ValueError(
                    f"[Band-pass filter] h_freq must be below Nyquist ({nyquist:g} Hz)"
                )

    def _apply_prep(self, params: Mapping[str, Any]) -> None:
        """Apply PREP-like bad-channel detection using pyprep.NoisyChannels."""
        random_state = int(params.get("random_state", 83092))
        run_correlation = bool(params.get("correlation", True))
        run_deviation = bool(params.get("deviation", True))
        run_hf_noise = bool(params.get("hf_noise", True))
        hf_noise_action = str(params.get("hf_noise_action", "review")).lower()
        run_nan_flat = bool(params.get("nan_flat", True))
        run_ransac = bool(params.get("ransac", True))

        if not any([run_correlation, run_deviation, run_hf_noise, run_nan_flat, run_ransac]):
            raise ValueError("[PREP] Provide at least one enabled method")
        if hf_noise_action not in {"review", "mark"}:
            raise ValueError("[PREP] hf_noise_action must be either 'review' or 'mark'")

        raw_prep = self.raw.copy().pick(self._eeg_picks)
        nc = NoisyChannels(raw_prep, do_detrend=False, random_state=random_state)
        method_bads: dict[str, list[str]] = {}

        if run_correlation:
            nc.find_bad_by_correlation(
                correlation_secs=1.0,
                correlation_threshold=0.4,
                frac_bad=0.01,
            )
        if run_deviation:
            nc.find_bad_by_deviation(deviation_threshold=5.0)
        if run_hf_noise:
            nc.find_bad_by_hfnoise(HF_zscore_threshold=5.0)
        if run_nan_flat:
            nc.find_bad_by_nan_flat()

        bad_dict = nc.get_bads(as_dict=True)
        if run_correlation:
            method_bads["correlation"] = list(bad_dict.get("bad_by_correlation", []))
        if run_deviation:
            method_bads["deviation"] = list(bad_dict.get("bad_by_deviation", []))
        if run_hf_noise:
            method_bads["hf_noise"] = list(bad_dict.get("bad_by_hf_noise", []))
        if run_nan_flat:
            nan_flat = list(bad_dict.get("bad_by_nan", [])) + list(bad_dict.get("bad_by_flat", []))
            method_bads["nan_flat"] = sorted(set(nan_flat))

        reliable_detected: set[str] = set()
        for method, names in method_bads.items():
            if method != "hf_noise":
                reliable_detected.update(names)

        hf_noise_detected = set(method_bads.get("hf_noise", []))
        review_only_by_hf_noise = sorted(hf_noise_detected - reliable_detected)
        all_detected = set(reliable_detected)
        if hf_noise_action == "mark":
            all_detected.update(hf_noise_detected)
            review_only_by_hf_noise = []

        self.raw.info["bads"] = sorted(set(self.raw.info.get("bads", [])).union(all_detected))

        if run_ransac:
            ransac_bads = self._run_prep_ransac(random_state=random_state)
            method_bads["ransac"] = ransac_bads
            if hf_noise_action == "review":
                review_only_by_hf_noise = sorted(
                    hf_noise_detected - reliable_detected - set(ransac_bads)
                )

        n_eeg = int(self._eeg_picks.size)
        eeg_names = {self.raw.ch_names[idx] for idx in self._eeg_picks}
        bad_eeg = sorted(ch for ch in self.raw.info.get("bads", []) if ch in eeg_names)
        frac_bad = len(bad_eeg) / n_eeg if n_eeg else 0.0

        self.history["prep"] = {
            "bads_by_method": method_bads,
            "bads_after_prep": list(self.raw.info.get("bads", [])),
            "review_only_by_hf_noise": review_only_by_hf_noise,
            "hf_noise_action": hf_noise_action,
            "n_eeg": n_eeg,
            "n_bad_eeg": len(bad_eeg),
            "frac_bad": frac_bad,
        }
        self._log(
            f"[EEGPreprocessor] PREP bad EEG channels: {len(bad_eeg)}/{n_eeg} "
            f"({frac_bad:.1%})"
        )
        self._log(f"[EEGPreprocessor] PREP bad list: {self._format_channels(bad_eeg)}")
        if review_only_by_hf_noise:
            self._log(
                "[EEGPreprocessor] HF-noise review-only channels: "
                f"{self._format_channels(review_only_by_hf_noise)}"
            )

    def _run_prep_ransac(self, random_state: int) -> list[str]:
        prev_bads = set(self.raw.info.get("bads", []))
        eeg_names = [self.raw.ch_names[idx] for idx in self._eeg_picks]
        good_eeg_names = [name for name in eeg_names if name not in prev_bads]

        if len(good_eeg_names) < 4:
            self._log("[EEGPreprocessor] RANSAC skipped: fewer than 4 good EEG channels")
            return []

        raw_ransac = self.raw.copy().pick_channels(good_eeg_names)
        nc_ransac = NoisyChannels(raw_ransac, do_detrend=False, random_state=random_state)
        nc_ransac.find_bad_by_ransac(
            n_samples=50,
            sample_prop=0.25,
            corr_thresh=0.75,
            frac_bad=0.4,
            corr_window_secs=5.0,
            channel_wise=False,
            max_chunk_size=None,
        )

        bad_dict = nc_ransac.get_bads(as_dict=True)
        ransac_bads = sorted(set(bad_dict.get("bad_by_ransac", [])))
        self.raw.info["bads"] = sorted(prev_bads.union(ransac_bads))
        return ransac_bads

    def _apply_interpolation_bad_channels(self, params: Mapping[str, Any]) -> None:
        """Interpolate bad channels using Raw.interpolate_bads."""
        before = list(self.raw.info.get("bads", []))
        reset_bads = bool(params.get("reset_bads", params.get("reset_bads_after_interp", True)))

        if not before:
            self.history["interpolation"] = {
                "status": "skipped",
                "reason": "no bad channels",
                "bads_before": [],
                "bads_after": [],
                "reset_bads": reset_bads,
            }
            self._log("[EEGPreprocessor] Interpolation skipped: no bad channels")
            return

        self.raw.interpolate_bads(reset_bads=reset_bads, verbose=self.verbose)
        after = list(self.raw.info.get("bads", []))

        self.history["interpolation"] = {
            "status": "done",
            "bads_before": before,
            "bads_after": after,
            "reset_bads": reset_bads,
        }
        self._log(
            "[EEGPreprocessor] Interpolated bad channels: "
            f"{self._format_channels(before)}"
        )

    def _update_raw_with_new_eeg(
        self,
        signal_new: np.ndarray,
        new_ch_set: ChannelSet,
        step_name: str,
        matrix: np.ndarray,
    ) -> None:
        """Replace the EEG data matrix and preserve non-EEG channels."""
        old_raw = self.raw
        old_info = old_raw.info
        old_eeg_picks = self._eeg_picks
        n_old_eeg = int(old_eeg_picks.size)
        n_new_eeg, n_times = signal_new.shape
        new_labels = list(new_ch_set.get_labels())

        if n_new_eeg != len(new_labels):
            raise ValueError(
                f"[{step_name}] signal_new has {n_new_eeg} channels, "
                f"but new_ch_set has {len(new_labels)} labels"
            )
        if n_times != old_raw.n_times:
            raise ValueError(
                f"[{step_name}] signal_new has {n_times} samples, "
                f"but raw has {old_raw.n_times} samples"
            )

        all_picks = np.arange(old_raw.info["nchan"])
        non_eeg_picks = np.setdiff1d(all_picks, old_eeg_picks)

        if n_new_eeg == n_old_eeg:
            old_raw._data[old_eeg_picks, :] = signal_new
            old_names = list(old_info["ch_names"])
            mapping = {
                old_names[pick]: new_label
                for pick, new_label in zip(old_eeg_picks, new_labels)
                if old_names[pick] != new_label
            }
            if mapping:
                old_raw.rename_channels(mapping)
            new_raw = old_raw
        else:
            eeg_info = mne.create_info(
                ch_names=new_labels,
                sfreq=old_info["sfreq"],
                ch_types="eeg",
            )
            eeg_info["line_freq"] = old_info.get("line_freq", None)
            raw_eeg = mne.io.RawArray(
                signal_new,
                eeg_info,
                first_samp=old_raw.first_samp,
                verbose=self.verbose,
            )
            try:
                raw_eeg.set_meas_date(old_info.get("meas_date", None))
            except Exception:
                pass
            if non_eeg_picks.size:
                raw_other = old_raw.copy().pick(non_eeg_picks)
                raw_eeg.add_channels([raw_other], force_update_info=True)
            raw_eeg.set_annotations(old_raw.annotations.copy())
            new_raw = raw_eeg

        self.raw = new_raw
        self.ch_set = new_ch_set
        self._eeg_picks = mne.pick_types(self.raw.info, eeg=True, exclude=())

        self.history[step_name] = {
            "matrix_shape": tuple(matrix.shape),
            "n_eeg_old": n_old_eeg,
            "n_eeg_new": n_new_eeg,
            "ch_labels": new_labels,
        }

        try:
            self._reapply_montage_after_channel_change(new_labels)
        except Exception as exc:
            self.history[step_name]["montage_warning"] = repr(exc)
            self._log(f"[EEGPreprocessor] Montage update warning after {step_name}: {exc}")

    def _apply_rereference(self, params: Mapping[str, Any]) -> None:
        """Apply re-referencing using ChannelSet.RerefMatrix."""
        channels = params.get("channels")
        if channels is None:
            raise ValueError("[Re-reference] 'channels' must be provided")

        matrix = np.asarray(self.ch_set.RerefMatrix(channels))
        eeg_data = self.raw.get_data(picks=self._eeg_picks)
        self._validate_projection_matrix(matrix, eeg_data, "Re-reference")
        new_ch_set = self.ch_set.copy().spfilt(matrix)
        signal_new = matrix.T @ eeg_data

        self._update_raw_with_new_eeg(
            signal_new=signal_new,
            new_ch_set=new_ch_set,
            step_name="rereference",
            matrix=matrix,
        )
        self._log(
            f"[EEGPreprocessor] Re-reference channels: {channels}; "
            f"EEG channels {eeg_data.shape[0]} -> {signal_new.shape[0]}"
        )

    def _apply_spatialfilter(self, params: Mapping[str, Any]) -> None:
        """Apply spatial filtering using ChannelSet.SLAP."""
        exclude = params.get("exclude")
        matrix = np.asarray(self.ch_set.SLAP(exclude=exclude)) if exclude else np.asarray(self.ch_set.SLAP())
        eeg_data = self.raw.get_data(picks=self._eeg_picks)
        self._validate_projection_matrix(matrix, eeg_data, "Spatial filter")
        new_ch_set = self.ch_set.copy().spfilt(matrix)
        signal_new = matrix.T @ eeg_data

        self._update_raw_with_new_eeg(
            signal_new=signal_new,
            new_ch_set=new_ch_set,
            step_name="spatial_filter",
            matrix=matrix,
        )
        self._log(
            f"[EEGPreprocessor] Spatial filter EEG channels: "
            f"{eeg_data.shape[0]} -> {signal_new.shape[0]}"
        )

    @staticmethod
    def _validate_projection_matrix(matrix: np.ndarray, eeg_data: np.ndarray, step_label: str) -> None:
        if matrix.ndim != 2:
            raise ValueError(f"[{step_label}] Projection matrix must be 2D")
        if matrix.shape[0] != eeg_data.shape[0]:
            raise ValueError(
                f"[{step_label}] Matrix first dimension is {matrix.shape[0]}, "
                f"but EEG data has {eeg_data.shape[0]} channels"
            )

    def _apply_annotation(self, params: Mapping[str, Any]) -> None:
        """Launch an interactive Raw plot and summarize BAD-region annotations."""
        plot = bool(params.get("plot", True))
        region_name = str(params.get("label", "BAD_region"))

        annotations = self.raw.annotations.copy()
        if not any(desc == region_name for desc in annotations.description):
            annotations.append(0.0, 0.0, region_name)
        self.raw.set_annotations(annotations)

        if plot:
            self._log(f"[EEGPreprocessor] Mark bad segments with label: {region_name}")
            self.raw.plot(block=True)

        annot = self.raw.annotations
        is_region = annot.description == region_name
        bad_onsets = annot.onset[is_region]
        bad_durations = annot.duration[is_region]
        positive = bad_durations > 0
        bad_onsets = bad_onsets[positive]
        bad_durations = bad_durations[positive]

        file_time = self.raw.n_times / float(self.raw.info["sfreq"])
        total_bad_time = float(np.sum(bad_durations))
        total_percent = total_bad_time / file_time * 100 if file_time else 0.0

        self.history["annotation"] = {
            "label": region_name,
            "n_bad_regions": int(len(bad_durations)),
            "bad_time_sec": total_bad_time,
            "file_time_sec": file_time,
            "bad_percent": total_percent,
        }
        self._log(
            f"[EEGPreprocessor] BAD regions: {len(bad_durations)}; "
            f"bad time {total_bad_time:.2f}s / {file_time:.2f}s ({total_percent:.2f}%)"
        )
