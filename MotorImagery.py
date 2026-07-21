"""
EEGMotorImagery: class

Provides epoching, spectral feature computation, statistical testing, and report-ready plotting for motor imagery paradigms
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union

import csv
import hashlib
import json
import os
import re
import numpy as np
import mne
from mne.io import BaseRaw
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.font_manager import FontProperties
from mpl_toolkits.axes_grid1 import make_axes_locatable

from BCI2000Tools.Electrodes import ChannelSet



class EEGMotorImagery:
    """Full set of operations to run motor imagery analysis"""
    def __init__(
        self,
        raw    : BaseRaw,
        ch_set : ChannelSet,
        nEpochs: int = 6, 
        duration_task: float = 10., 
        skip: float = 1., 
        resolution: float = 1.,
        freq_bands: List = [4.0, 8.0, 13.0, 31.0],
        nSim: int = 9999,
        strict: bool = False,
        copy: bool = True,
        verbose: bool = False,
        save_path: str = None,
        export_csv: bool = True,
        auto_run: bool = True,
        random_state: Optional[int] = 83092,
        run_decoding: bool = True,
        psd_processing: str = "power",
        background_fit_kwargs: Optional[Dict[str, Any]] = None,
        background_fit_plot_channels: Optional[Union[List[str], bool]] = None,
    ) -> None:

        # Work on a copy if requested
        self.raw: BaseRaw = raw.copy() if copy else raw
        self.raw.load_data()

        self.ch_set        = ch_set
        self.nEpochs       = nEpochs
        self.duration_task = duration_task
        self.skip          = skip
        self.resolution    = resolution
        self.freq_bands    = freq_bands
        self.nSim          = nSim
        
        self.strict        = strict
        self.verbose       = verbose
        self.save_path     = save_path
        self.export_csv    = export_csv
        self.random_state  = random_state
        self.run_decoding  = run_decoding
        self.psd_processing = self._normalize_psd_processing(psd_processing)
        self.background_fit_kwargs = dict(background_fit_kwargs or {})
        self.background_fit_plot_channels = background_fit_plot_channels
        self.psd_values_are_db = self.psd_processing == "one_over_f_subtracted"
        self.psd_value_units = "dB" if self.psd_values_are_db else "power"
        self.psd_value_label = (
            "1/f-subtracted PSD residual (dB)"
            if self.psd_values_are_db
            else "Welch PSD power"
        )
        self.background_fit_outputs: Dict[str, Any] = {}
        self._rng_master   = np.random.default_rng(random_state)
        self.generated_figures: List[str] = []
        self.exported_files: List[str] = []
        self.analysis_warnings: List[str] = []
        self.trial_quality_summary: List[Dict[str, Any]] = []
        self.paradigm_event_summary: List[Dict[str, Any]] = []
        self.parameter_summary: Dict[str, Any] = {}
        
        if self.strict: 
            self.ch_motor_imagery = [
                "fc4", "fc3", "fc2", "fc1",
                "c6" , "c5" , "c4" , "c3" , "c2" , "c1" ,
                "cp6", "cp5", "cp4", "cp3", "cp2", "cp1", 
            ]
        else:
            self.ch_motor_imagery = [
                "fc4", "fc3", "fc2", "fc1",
                "c6" , "c5" , "c4" , "c3" , "c2" , "c1" ,
                "cp6", "cp5", "cp4", "cp3", "cp2", "cp1", 
                "p4" , "p3" , "p2" , "p1" , 
                "e86", "e53", "e79", "e54",
            ]
        
        self._ensure_save_path()

        if auto_run:
            self.run()

    def _info(self, message: str) -> None:
        """Print an informational message when verbose output is enabled"""
        if self.verbose:
            print(f"[motor-imagery] {message}")

    def _warn(self, message: str) -> None:
        """Print a warning message"""
        self.analysis_warnings.append(str(message))
        print(f"[motor-imagery warning] {message}")

    @staticmethod
    def _normalize_psd_processing(value: str) -> str:
        """Normalize the PSD processing mode used downstream by stats and plots"""
        normalized = str(value or "power").strip().lower().replace("-", "_")
        aliases = {
            "raw": "power",
            "raw_power": "power",
            "welch": "power",
            "welch_power": "power",
            "1f": "one_over_f_subtracted",
            "1_over_f": "one_over_f_subtracted",
            "1_over_f_subtracted": "one_over_f_subtracted",
            "one_over_f": "one_over_f_subtracted",
            "one_over_f_subtraction": "one_over_f_subtracted",
            "one_over_f_subtracted": "one_over_f_subtracted",
            "background_removed": "one_over_f_subtracted",
            "background_subtracted": "one_over_f_subtracted",
        }
        normalized = aliases.get(normalized, normalized)
        if normalized not in {"power", "one_over_f_subtracted"}:
            raise ValueError(
                "psd_processing must be 'power' or 'one_over_f_subtracted', "
                f"got {value!r}"
            )
        return normalized

    @staticmethod
    def _power_to_db(power: np.ndarray) -> np.ndarray:
        """Convert PSD power to dB using the convention used by the existing pipeline"""
        power = np.asarray(power, dtype=float)
        safe_power = np.maximum(power, np.finfo(float).tiny)
        return 10.0 * np.log10(safe_power * 1e12)

    def _psd_values_for_analysis(self, values: np.ndarray) -> np.ndarray:
        """Return PSD values in the units used for effects, decoding, and plots"""
        values = np.asarray(values, dtype=float)
        if self.psd_values_are_db:
            return values
        return self._power_to_db(values)

    def _psd_axis_label(self) -> str:
        """Return a display label for PSD-valued axes"""
        if self.psd_values_are_db:
            return "1/f-subtracted PSD residual [dB]"
        return "PSD [dB]"

    def _rng_for(self, label: str) -> np.random.Generator:
        """Create a deterministic RNG for a named analysis component"""
        base = 0 if self.random_state is None else int(self.random_state)
        digest = hashlib.sha256(f"{base}:{label}".encode("utf-8")).hexdigest()[:16]
        seed = int(digest, 16) % (2**32 - 1)
        return np.random.default_rng(seed)

    def _output_subdir(self, dirname: str) -> Optional[str]:
        """Return an output subfolder path and create it when save_path is configured"""
        if self.save_path is None:
            return None
        path = os.path.join(self.save_path, dirname)
        os.makedirs(path, exist_ok=True)
        return path

    def _image_path(self, filename: str) -> Optional[str]:
        """Return the path for a generated image inside <save_path>/images"""
        folder = self._output_subdir("images")
        if folder is None:
            return None
        return os.path.join(folder, filename)

    def _csv_path(self, filename: str) -> Optional[str]:
        """Return the path for a generated CSV inside <save_path>/csv"""
        folder = self._output_subdir("csv")
        if folder is None:
            return None
        return os.path.join(folder, filename)

    def _savefig(self, fig: Figure, basename: str, formats: Tuple[str, ...] = ("png", "svg")) -> None:
        """Save a figure in <save_path>/images and track it for metadata"""
        if self.save_path is None:
            return
        for ext in formats:
            path = self._image_path(f"{basename}.{ext}")
            if path is None:
                continue
            fig.savefig(path, bbox_inches="tight")
            self.generated_figures.append(path)

    def _track_export(self, path: str) -> None:
        """Track exported non-figure files for metadata"""
        self.exported_files.append(path)

    def _relative_output_path(self, path: str) -> str:
        """Return a path relative to save_path when possible"""
        if self.save_path is None:
            return os.path.basename(path)
        try:
            return os.path.relpath(path, self.save_path)
        except ValueError:
            return os.path.basename(path)

    def _ensure_save_path(self) -> None:
        """Create the output folder and analysis subfolders when configured"""
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)
            os.makedirs(os.path.join(self.save_path, "images"), exist_ok=True)
            if self.export_csv:
                os.makedirs(os.path.join(self.save_path, "csv"), exist_ok=True)

    def run(self) -> None:
        """Run the full motor imagery pipeline"""
        self._info("Inferring paradigm annotations")
        self._evaluate_motorimagery_paradigm(swap=False)
        self._build_parameter_summary()

        self._info("Generating epochs")
        self._generate_epochs()
        self._summarize_trial_quality()

        self._info("Computing PSDs")
        self._generate_psds()
        if self.psd_processing == "one_over_f_subtracted":
            self._info("Applying 1/f background subtraction")
            self._apply_one_over_f_background_subtraction()

        self._info("Finding left and right motor channels")
        self._grab_left_right_electrodes()

        self._info("Running lateralized statistics")
        self._run_stat_tests(verbose=self.verbose)
        self._apply_fdr_to_stats()
        self._build_main_result_summary()

        if self.run_decoding:
            self._info("Running optional decoding analysis")
            self._run_decoding_analysis()

        if self.save_path is not None and self.export_csv:
            self._info("Exporting CSV summaries")
            self._export_all_csv()

        self._info("Generating report plots")
        self._generate_report_plots()

        if self.save_path is not None:
            self._export_report_metadata_json()

    def _generate_report_plots(self) -> None:
        """Generate the plots used by the existing PDF report"""
        self._plot_all_test_distributions(transf="r2")
        self._plot_lateralized_topomaps()
        self._plot_band_pvalues_bootstrap_separate()

        if self.strict:
            chs_to_plot = ["fc3", "fc4", "c3", "c4", "cp3", "cp4"]
        else:
            chs_to_plot = ["fc3", "fc4", "c3", "c4", "p3", "p4"]

        for ch in chs_to_plot:
            try:
                self._plot_channel_psd(ch)
            except Exception as exc:
                self._warn(f"Could not create PSD plot for channel '{ch}': {exc}")

        self._plot_paradigm_timeline()
        self._plot_band_effect(ci=95)

        for task, rest in (("left", "left_rest"), ("right", "right_rest")):
            for band in ((8.0, 13.0), (13.0, 31.0)):
                try:
                    self._plot_within_trial_bandpower_overlay_rois(task, rest, band=band)
                except Exception as exc:
                    self._warn(f"Could not create contra/ipsi ERD/ERS overlay for {task} {band}: {exc}")

        try:
            self._plot_decoding_summary()
            self._plot_decoding_confusion_best()
        except Exception as exc:
            self._warn(f"Could not create decoding plots: {exc}")

    @staticmethod
    def step_down_events(signal: np.ndarray, time: Optional[np.ndarray] = None, tol: float = 0.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Identify downward step changes in a trigger-like signal and return onset/offset indices and durations"""
        signal = np.asarray(signal)
        if time is None:
            time = np.arange(signal.size)
        else:
            time = np.asarray(time)
            if time.shape != signal.shape:
                raise RuntimeError("`time` and `signal` must have the same shape")

        diff = np.diff(signal)
        step_indices = np.where(diff < -tol)[0] + 1
        step_times = time[step_indices]

        if step_times.size == 0:
            raise RuntimeError("No downward trigger steps found in PresentationDisplayed")

        while step_times.size and step_times[0] < 2000:
            step_times = step_times[1:]

        if step_times.size < 2:
            raise RuntimeError("Not enough PresentationDisplayed trigger steps to infer onset/offset pairs")

        if step_times.size % 2 != 0:
            step_times = step_times[:-1]

        durations = np.diff(step_times)
        onset = step_times[0::2]
        offset = step_times[1::2]
        return onset, offset, durations

    @staticmethod
    def find_step_intervals(signal: np.ndarray, threshold: float = 0.) -> Tuple[np.ndarray, np.ndarray]:
        """Return onset and offset indices for complete contiguous intervals where the signal exceeds a threshold"""
        signal = np.asarray(signal)
        active = signal > threshold
        changes = np.diff(active.astype(int))
        onsets = np.where(changes == 1)[0] + 1
        offsets = np.where(changes == -1)[0] + 1

        if active.size and active[0]:
            onsets = np.insert(onsets, 0, 0)

        # Do not invent a terminal offset when the trigger never returns to zero
        # The caller uses offsets as task starts, so a fabricated final offset creates an event at the end of file
        n_pairs = min(onsets.size, offsets.size)
        onsets = onsets[:n_pairs]
        offsets = offsets[:n_pairs]

        keep = onsets >= 2000
        onsets = onsets[keep]
        offsets = offsets[keep]

        if onsets.size == 0:
            raise RuntimeError("No valid complete trigger intervals found")

        return onsets, offsets

    @staticmethod
    def dummy(is_treatment: np.ndarray) -> np.ndarray:
        """Dummy-code labels for two groups"""
        is_treatment = np.asarray(is_treatment).astype(bool)
        return (~is_treatment).astype(int)
    
    @staticmethod
    def calculate_eta2(
        x: np.ndarray,
        is_treatment: np.ndarray,
        signed: bool = True,
    ) -> np.ndarray:
        """Compute (signed) eta-squared for two groups"""
        x = np.asarray(x)
        is_treatment = np.asarray(is_treatment).astype(bool)

        n1 = np.sum(is_treatment)
        n2 = np.sum(~is_treatment)

        x1 = x[is_treatment]
        x2 = x[~is_treatment]

        mu1 = np.mean(x1, axis=0)
        mu2 = np.mean(x2, axis=0)
        grand_mean = np.mean(x, axis=0)

        # Within-group and between-group sum of squares
        ssw = np.sum((x1 - mu1) ** 2, axis=0) + np.sum((x2 - mu2) ** 2, axis=0)
        ssb = n1 * (mu1 - grand_mean) ** 2 + n2 * (mu2 - grand_mean) ** 2

        eta2 = ssb / (ssb + ssw + 1e-12)

        if not signed:
            return eta2

        signs = np.where(mu2 - mu1 > 0, 1, -1)
        return eta2 * signs

    @staticmethod
    def shuffle(a: np.ndarray, rng: Optional[np.random.Generator] = None) -> np.ndarray:
        """Return a shuffled copy of a 1D array"""
        rng = rng or np.random.default_rng()
        a = np.asarray(a).copy()
        rng.shuffle(a)
        return a

    @staticmethod
    def bootstrap_resample(
        x: np.ndarray,
        is_treatment: np.ndarray,
        rng: Optional[np.random.Generator] = None,
    ) -> np.ndarray:
        """Resample trials with replacement within each group"""
        rng = rng or np.random.default_rng()
        x = np.asarray(x)
        is_treatment = np.asarray(is_treatment).astype(bool)

        idx_t = np.where(is_treatment)[0]
        idx_c = np.where(~is_treatment)[0]

        res_t = rng.choice(idx_t, size=len(idx_t), replace=True) if len(idx_t) else idx_t
        res_c = rng.choice(idx_c, size=len(idx_c), replace=True) if len(idx_c) else idx_c

        res_idx = np.concatenate([res_t, res_c])
        return x[res_idx]

    @staticmethod
    def pvalue_interval(p: float, n: int) -> Tuple[float, float]:
        """Simple normal-approx. interval for Monte Carlo p-values"""
        p = float(np.clip(p, 0.0, 1.0))
        n = int(max(n, 1))
        se = np.sqrt(p * (1.0 - p) / n)
        return max(0.0, p - 1.96 * se), min(1.0, p + 1.96 * se)

    @staticmethod
    def neg_p(p: float) -> float:
        """Convert p to -log(p) for a compact 'evidence' score"""
        p = float(np.clip(p, 1e-300, 1.0))
        #return -np.log10(p)
        return float(-np.log(p))
            
    @staticmethod
    def bands_from_edges(band_edges: List[float]) -> List[Tuple[float, float]]:
        """
        Convert band edge list to consecutive half-open bands

        Example:
            [4, 8, 13, 31] -> [(4, 8), (8, 13), (13, 31)]

        Band selection later uses [low, high) except for the final band,
        where the high edge is included. This avoids the fragile legacy y - 1 rule.
        """
        edges = list(map(float, band_edges))
        if len(edges) < 2:
            raise ValueError("band_edges must contain at least two values")
        if any(np.diff(edges) <= 0):
            raise ValueError("band_edges must be strictly increasing")
        return list(zip(edges[:-1], edges[1:]))

    @staticmethod
    def _band_indices(freqs: np.ndarray, low_f: float, high_f: float, is_last: bool = False) -> np.ndarray:
        """Return frequency-bin indices for a half-open band interval"""
        freqs = np.asarray(freqs, dtype=float)
        if is_last:
            return np.where((freqs >= low_f) & (freqs <= high_f))[0]
        return np.where((freqs >= low_f) & (freqs < high_f))[0]

    @staticmethod
    def band_key(low_f: float, high_f: float) -> str:
        """Format a canonical band key used by this class"""
        return f"{low_f:g}-{high_f:g}Hz"

    @staticmethod
    def build_custom_cmap(colors: List[str]) -> matplotlib.colormaps:
        """Create a simple three-point diverging colormap from [low, mid, high] color specifications"""
        
        c1, c2, c3 = colors
        
        colors = [
            (0.0, c1),  # Color at -1
            (0.5, c2),  # Color at 0
            (1.0, c3),  # Color at 1
        ]  

        return LinearSegmentedColormap.from_list("custom_cmap", colors)
     

    def _build_parameter_summary(self) -> Dict[str, Any]:
        """Store analysis parameters that should appear in exports and the PDF"""
        self.parameter_summary = {
            "n_epochs_per_trial": int(self.nEpochs),
            "duration_task_sec": float(self.duration_task),
            "skip_sec": float(self.skip),
            "psd_resolution_hz": float(self.resolution),
            "freq_bands_input": list(map(float, self.freq_bands)) if isinstance(self.freq_bands, list) else self.freq_bands,
            "n_simulations": int(self.nSim),
            "strict_channel_set": bool(self.strict),
            "random_state": self.random_state,
            "run_decoding": bool(self.run_decoding),
            "psd_processing": self.psd_processing,
            "psd_value_units": self.psd_value_units,
            "psd_value_label": self.psd_value_label,
            "sfreq": float(self.raw.info.get("sfreq", np.nan)),
            "highpass": float(self.raw.info.get("highpass", np.nan) or 0.0),
            "lowpass": float(self.raw.info.get("lowpass", np.nan) or 0.0),
            "n_raw_channels": int(len(self.raw.info.get("ch_names", []))),
            "n_raw_samples": int(getattr(self.raw, "n_times", 0)),
        }
        return self.parameter_summary

    def _classify_event_codes(self, code: np.ndarray, swap: bool = False) -> List[str]:
        """Map trigger codes to condition labels and validate sequence assumptions"""
        descriptions: List[str] = []
        unknown_codes: Dict[str, int] = {}

        for c in np.asarray(code):
            c_int = int(c) if np.isfinite(c) else -999
            if c_int == 1:
                descriptions.append("left")
            elif c_int == 2:
                descriptions.append("right")
            elif c_int == 3:
                descriptions.append("right_rest" if swap else "left_rest")
            elif c_int == 4:
                descriptions.append("left_rest" if swap else "right_rest")
            else:
                label = f"unknown_{c_int}"
                unknown_codes[label] = unknown_codes.get(label, 0) + 1
                descriptions.append(label)

        # Some BCI2000 files encode both rest periods with code 3
        # Only apply the legacy every-fourth correction when the sequence clearly supports it
        if "right_rest" not in descriptions and len(descriptions) >= 4:
            corrected = list(descriptions)
            corrected_count = 0
            for i, desc in enumerate(corrected):
                if (i + 1) % 4 == 0 and desc == "left_rest":
                    corrected[i] = "right_rest"
                    corrected_count += 1
            if corrected_count:
                descriptions = corrected
                self._info(
                    "Applied legacy rest-code correction: every 4th left_rest was relabeled as right_rest"
                )

        counts: Dict[str, int] = {}
        for desc in descriptions:
            counts[desc] = counts.get(desc, 0) + 1

        expected = ["left", "right", "left_rest", "right_rest"]
        rows = []
        for label in expected:
            rows.append({
                "label": label,
                "count": int(counts.get(label, 0)),
                "status": "ok" if counts.get(label, 0) > 0 else "missing",
            })
        for label, n in sorted(counts.items()):
            if label not in expected:
                rows.append({"label": label, "count": int(n), "status": "unexpected"})

        sequence_complete_blocks = int(len(descriptions) // 4)
        sequence_remainder = int(len(descriptions) % 4)
        rows.append({
            "label": "complete_4_event_blocks",
            "count": sequence_complete_blocks,
            "status": "ok" if sequence_remainder == 0 else f"remainder_{sequence_remainder}",
        })
        self.paradigm_event_summary = rows

        missing = [r["label"] for r in rows if r.get("status") == "missing"]
        if missing:
            self._warn(f"Paradigm validation missing expected labels: {', '.join(missing)}")
        if unknown_codes:
            self._warn(f"Paradigm validation found unexpected trigger codes: {unknown_codes}")

        return descriptions

    def _evaluate_motorimagery_paradigm(self, swap: bool = True) -> None:
        """Infer the motor imagery task/rest structure from trigger channels and annotate the Raw recording accordingly"""
        nEpochs       = self.nEpochs
        duration_task = self.duration_task
        skip          = self.skip
        
        stimCode    = None
        presentDisp = None
        
        stimCode = self.raw.get_data(picks='StimulusCode').ravel()

        try:
            presentDisp = self.raw.get_data(picks='PresentationDisplayed').ravel()
        except Exception as exc:
            self._info(f"PresentationDisplayed channel unavailable, falling back to StimulusCode: {exc}")
        
        if presentDisp is not None:
            onset, offset, duration = self.step_down_events(presentDisp, time=None, tol=0.0)
            duration = np.array([duration_task - skip] * len(onset)) # seconds
            code     = np.array([stimCode[t] for t in onset])
            onset    = offset 
            
            if 0 in code:
                onset, offset = self.find_step_intervals(stimCode, threshold=0.)
                #duration = np.array([off - on for on,off in zip(onset,offset)])
                duration = np.array([duration_task - skip] * len(onset)) # seconds
                code     = np.array([stimCode[t] for t in onset])
                onset    = offset 
        
        elif stimCode is not None:
            onset, offset = self.find_step_intervals(stimCode, threshold=0.)
            #duration = np.array([off - on for on,off in zip(onset,offset)])
            duration = np.array([duration_task - skip] * len(onset)) # seconds
            code     = np.array([stimCode[t] for t in onset])
            onset    = offset 
            
        else:
            raise RuntimeError("Not able to determine paradigm")
        
        n_events = min(len(onset), len(duration), len(code))
        if n_events == 0:
            raise RuntimeError("No motor imagery events could be inferred from trigger channels")

        onset = np.asarray(onset[:n_events])
        duration = np.asarray(duration[:n_events])
        code = np.asarray(code[:n_events])

        description_main = self._classify_event_codes(code, swap=swap)

        # Grab existing annotations
        annot = self.raw.annotations
        # Bad regions annotations
        onset_bad       = list(annot.onset[   np.where(annot.description == "BAD_region")]) # in seconds
        duration_bad    = list(annot.duration[np.where(annot.description == "BAD_region")]) # in seconds
        description_bad = ["BAD_region"] * len(duration_bad)
        # Trial type annotations
        onset_       = []
        duration_    = []
        description_ = []
                     
        for i, desc in enumerate(description_main):
            if desc.startswith("unknown_"):
                continue
            onset_.append(onset[i])
            duration_.append(duration[i])
            description_.append(desc)

        # Skip the first `skip` seconds of each segment
        # Convert onset_ to seconds
        sfreq = float(self.raw.info['sfreq'])
        raw_end = float(self.raw.times[-1]) if self.raw.n_times > 0 else 0.0
        onset_sec = [t / sfreq + skip for t in onset_]

        valid_onset = []
        valid_duration = []
        valid_description = []
        required_window = float(duration_task - skip)
        sample_margin = 1.0 / sfreq

        for onset_value, duration_value, desc in zip(onset_sec, duration_, description_):
            if onset_value + required_window - sample_margin <= raw_end + 1e-12:
                valid_onset.append(float(onset_value))
                valid_duration.append(float(duration_value))
                valid_description.append(desc)
            else:
                self._warn(
                    f"Dropped incomplete terminal event '{desc}' at {onset_value:.3f}s: "
                    f"requires {required_window:.3f}s but data end is {raw_end:.3f}s"
                )

        onset_ = valid_onset
        duration_ = valid_duration
        description_ = valid_description
        
        # Trial number annotations
        delta = (duration_task - skip) / nEpochs
        
        onset_number = []
        duration_number = []
        description_number = []
        
        k = 0
        for i,t in enumerate(onset_):
            if description_[i] == 'left': k += 1
            for j in range(nEpochs):
                epoch_onset = t + delta * j
                if epoch_onset + delta - sample_margin > raw_end + 1e-12:
                    self._warn(
                        f"Skipped incomplete numbered epoch '{description_[i]}_{k}' "
                        f"at {epoch_onset:.3f}s"
                    )
                    continue
                onset_number.append(epoch_onset)
                duration_number.append(delta)
                description_number.append(f"{description_[i]}_{k}")
        
        onset_       = onset_bad       + onset_       + onset_number
        duration_    = duration_bad    + duration_    + duration_number
        description_ = description_bad + description_ + description_number
                 
        # Set annotation to mne.io.Raw
        my_annot = mne.Annotations(onset=onset_, duration=duration_, description=description_)
        self.raw.set_annotations(my_annot)
    
    def _make_epochs_batch(self, event_id: int) -> mne.Epochs:
        """Create fixed-length MNE Epochs for a single annotation event id using the configured task window"""
        
        tmin = 0.0
        sfreq = float(self.raw.info["sfreq"])
        epoch_duration = (self.duration_task - self.skip) / self.nEpochs
        tmax = epoch_duration - (1.0 / sfreq)
        if tmax <= tmin:
            raise RuntimeError(
                f"Invalid epoch window: duration={epoch_duration:.6f}s, sfreq={sfreq:.3f}Hz"
            )
        
        events_from_annot, event_dict = mne.events_from_annotations(self.raw, verbose=False)
        events_subset = events_from_annot[events_from_annot[:, 2] == int(event_id)]
        if events_subset.size == 0:
            raise RuntimeError(f"No events found for event id {event_id}")

        # MNE includes tmax, so subtracting one sample prevents requesting data past an exact boundary
        epochs_ = mne.Epochs(
            self.raw,
            events_subset,
            event_id=int(event_id),
            tmin=tmin,
            tmax=tmax,
            baseline=None,
            preload=True,
            verbose=False,
        )
        # (number of epochs, channel, t)
        return epochs_

    def _generate_epochs(self) -> None:
        """Generate and store per-batch MNE Epochs from numbered task annotations, dropping unusable batches"""
        
        events_from_annot, event_dict = mne.events_from_annotations(self.raw, verbose=False)
        batches = [x for x in event_dict.keys() if any(char.isdigit() for char in x)]
        self.event_ids = {b: event_dict[b] for b in batches}

        #epochs_dict = {}
        #for batch in batches:
        #    try:
        #        epochs_dict[ batch ] = self._make_epochs_batch(event_id=event_dict[ batch ])
        #    except:
        #        pass
        #if len(list(epochs_dict.keys())) == 0: 
        #    raise RuntimeError("There are no available epochs")
        #self.epochs_dict = epochs_dict
        
        epochs_dict = {}
        for key, event_id in self.event_ids.items():
            try:
                epochs_ = self._make_epochs_batch(event_id=event_id)

                if len(epochs_) == 0:
                    if self.verbose:
                        print(f"[epochs] Skipping '{key}': all epochs dropped")
                    continue

                epochs_dict[key] = epochs_

            except Exception as e:
                if self.verbose:
                    print(f"[epochs] Failed '{key}': {e}")
                continue

        self.epochs_dict = epochs_dict

    def _make_psds_batch(
        self, 
        batch: str, 
        resolution: float = None,
        fmin: float = None,
        fmax: float = None,
        nPerSegment: int = None,
        nOverlap: int = None,
        aggregate_epochs: bool = True,
    ) -> np.ndarray:
        """Compute Welch PSDs for one epoch batch, with optional averaging across epochs for downstream statistics"""
        
        fs = self.raw.info['sfreq']
        tmin = self.epochs_dict[ batch ].times[ 0 ]
        tmax = self.epochs_dict[ batch ].times[-1 ]
        
        nfft = int(fs / resolution)        # Number of FFT points
        effective_window = 1 / resolution  # Effective window length in seconds
        expected_segments = (
            int(((tmax - tmin) - effective_window) / (effective_window - nOverlap / fs))
            + 1
        )  # Expected number of segments
        expected_bins = int((fmax - fmin) / resolution + 1)  # Expected number of frequency bins

        # Compute PSD using Welch's method and keep MNE's exact frequency vector
        spectrum = self.epochs_dict[ batch ].compute_psd(
            method="welch",
            fmin=fmin,
            fmax=fmax,
            tmin=tmin,
            tmax=tmax,
            n_fft=nfft,
            n_overlap=nOverlap,
            n_per_seg=nPerSegment,
            average=None,
            window="hann",
            # output="power", # only past version 1.4, before that it's power by default
            verbose=False,
        )
        psd_ = spectrum.get_data()
        freqs_ = np.asarray(getattr(spectrum, "freqs", []), dtype=float)
        if freqs_.size == 0:
            try:
                _, freqs_ = spectrum.get_data(return_freqs=True)
                freqs_ = np.asarray(freqs_, dtype=float)
            except Exception:
                freqs_ = np.arange(fmin, fmax + resolution, resolution, dtype=float)

        # Verbose output for PSD computation details
        if self.verbose:
            print(f"Expected w/ resolution {resolution} [Hz/bin]: ")
            print(f"  - Eff_Window Length [s]: {effective_window}")
            print(f"  - Epochs: {self.epochs_dict[ batch ].get_data(picks='eeg').shape[0]} -> {tmax-tmin} s-long (per type)")
            print(f"  - Channels: {self.epochs_dict[ batch ].get_data(picks='eeg').shape[1]}")
            print(f"  - Bins: {expected_bins}")
            print(f"  - Expected segments (or periodograms): {expected_segments}")
            print(f"Dimension check: (epoch, ch, bins, segments/periodograms) = {psd_.shape}")

        # Always reduce Welch segments but optionally keep the epoch dimension
        # Average across Welch segments/periodograms if present:
        # (epochs, ch, bins, segments) -> (epochs, ch, bins)
        if psd_.ndim > 3:
            psd_ = np.mean(psd_, axis=3)

        # Optionally average across epochs:
        # (epochs, ch, bins) -> (ch, bins)
        if aggregate_epochs:
            psd_ = np.mean(psd_, axis=0)

        return psd_, freqs_

    def _generate_psds(self) -> None:
        """Compute and store PSDs for all epoch batches, keeping both epoch-averaged and per-epoch PSD arrays"""
        fs = self.raw.info["sfreq"]
        fmin = float(self.raw.info.get("highpass", 0.0) or 0.0)
        fmax = float(self.raw.info.get("lowpass", fs / 2.0) or (fs / 2.0))

        sec_per_segment = 1.0 / float(self.resolution)
        sec_overlap = sec_per_segment / 2.0
        n_per_segment = int(sec_per_segment * fs)
        n_overlap = int(sec_overlap * fs)

        psds_dict = {}
        psds_epochs_dict = {}

        for key in self.epochs_dict.keys():
            try:
                psd_epochs, freqs = self._make_psds_batch(
                    batch=key,
                    resolution=self.resolution,
                    fmin=fmin,
                    fmax=fmax,
                    nPerSegment=n_per_segment,
                    nOverlap=n_overlap,
                    aggregate_epochs=False,
                )
                psds_epochs_dict[key] = psd_epochs
                psds_dict[key] = np.mean(psd_epochs, axis=0)
                if not hasattr(self, "freqs"):
                    self.freqs = np.asarray(freqs, dtype=float)
                elif len(self.freqs) != len(freqs) or not np.allclose(self.freqs, freqs):
                    self._warn(f"PSD frequency bins differ for batch '{key}'. Using first batch frequency vector")
            except Exception as exc:
                self._warn(f"PSD computation failed for batch '{key}': {exc}")
                continue

        if len(psds_dict) == 0:
            raise RuntimeError("There are no available PSDs")

        self.psds_dict = psds_dict
        self.psds_epochs_dict = psds_epochs_dict
        self.psds_power_dict = {key: np.asarray(value, dtype=float).copy() for key, value in psds_dict.items()}
        self.psds_power_epochs_dict = {
            key: np.asarray(value, dtype=float).copy()
            for key, value in psds_epochs_dict.items()
        }

    def _channel_labels_for_psd(self, n_channels: int) -> List[str]:
        """Return channel labels aligned to PSD channel rows"""
        labels = [str(label).lower() for label in self.ch_set.get_labels()]
        if len(labels) < n_channels:
            labels = [str(name).lower() for name in self.raw.info.get("ch_names", [])]
        if len(labels) < n_channels:
            raise RuntimeError(
                f"Only {len(labels)} channel labels are available for {n_channels} PSD rows"
            )
        return labels[:n_channels]

    def _write_background_fit_input_csv(self) -> str:
        """Write all-channel PSD dB values used by the 1/f background-fit helper"""
        try:
            import pandas as pd
        except Exception as exc:
            raise RuntimeError(f"pandas is required for 1/f background subtraction: {exc}") from exc

        if not hasattr(self, "psds_power_dict") or not self.psds_power_dict:
            raise RuntimeError("Raw power PSD cache missing; cannot fit 1/f background")

        freqs = np.asarray(getattr(self, "freqs", []), dtype=float)
        if freqs.size == 0:
            raise RuntimeError("Frequency bins are unavailable; cannot fit 1/f background")

        first_psd = np.asarray(next(iter(self.psds_power_dict.values())), dtype=float)
        n_channels, n_bins = first_psd.shape
        if freqs.size != n_bins:
            raise RuntimeError(
                f"PSD frequency vector length ({freqs.size}) does not match PSD bins ({n_bins})"
            )

        labels = self._channel_labels_for_psd(n_channels)
        rows: List[Dict[str, Any]] = []

        for batch, psd_power in self.psds_power_dict.items():
            psd_power = np.asarray(psd_power, dtype=float)
            if psd_power.shape != (n_channels, n_bins):
                raise RuntimeError(
                    f"PSD shape for batch '{batch}' is {psd_power.shape}, expected {(n_channels, n_bins)}"
                )
            psd_db = self._power_to_db(psd_power)
            condition = self._condition_from_epoch_key(batch)
            for ch_idx, channel in enumerate(labels):
                for f_idx, freq in enumerate(freqs):
                    rows.append({
                        "batch": batch,
                        "condition": condition,
                        "channel": channel,
                        "frequency_hz": float(freq),
                        "psd_power": float(psd_power[ch_idx, f_idx]),
                        "psd_db": float(psd_db[ch_idx, f_idx]),
                    })

        csv_dir = self._output_subdir("csv")
        if csv_dir is None:
            raise RuntimeError("A save_path is required for 1/f background subtraction")

        path = os.path.join(csv_dir, "psd_long_format_1overf_background_input.csv")
        pd.DataFrame(rows).to_csv(path, index=False)
        self._track_export(path)
        return path

    def _fit_one_over_f_background(self, input_csv: str) -> Dict[str, Any]:
        """Run the provided aggregate PSD background-fit pipeline"""
        try:
            from MotorImagery_aggregate_psds_fit_background import process_psd_file
        except Exception as exc:
            raise RuntimeError(
                "Could not import MotorImagery_aggregate_psds_fit_background.py "
                f"for 1/f background subtraction: {exc}"
            ) from exc

        plot_channels = self.background_fit_plot_channels
        if plot_channels is None:
            plot_channels = ["c3", "c4"]

        fit_kwargs = {
            "output_aggregated_csv": "psd_long_format_1overf_aggregated.csv",
            "output_best_fit_csv": "psd_long_format_1overf_aggregated_best_fit.csv",
            "output_fit_summary_csv": "psd_1overf_fit_summary_all_models.csv",
            "output_peak_summary_csv": "psd_1overf_selected_gaussian_peak_summary.csv",
            "output_peak_group_summary_csv": "psd_1overf_selected_gaussian_peak_group_summary.csv",
            "plot_channels": plot_channels,
            "plot_condition": None,
            "plot_folder_name": "psd_1overf_best_fit_plots",
        }
        fit_kwargs.update(self.background_fit_kwargs)
        return process_psd_file(input_csv=input_csv, **fit_kwargs)

    def _background_lookup_from_best_fit(self, best_fit_df: Any, labels: List[str]) -> Dict[str, np.ndarray]:
        """Create condition/channel background arrays aligned to self.freqs"""
        freqs = np.asarray(getattr(self, "freqs", []), dtype=float)
        lookup: Dict[str, np.ndarray] = {}

        for (condition, channel), sub in best_fit_df.groupby(["condition", "channel"], sort=False):
            sub = sub.sort_values("frequency_hz")
            sub_freqs = sub["frequency_hz"].to_numpy(dtype=float)
            background = sub["selected_background_1overf_db"].to_numpy(dtype=float)

            if sub_freqs.size != freqs.size or not np.allclose(sub_freqs, freqs):
                background = np.interp(freqs, sub_freqs, background, left=np.nan, right=np.nan)

            lookup[f"{str(condition).lower()}::{str(channel).lower()}"] = background

        missing = [
            key
            for condition in ("left", "right", "left_rest", "right_rest")
            for key in (f"{condition}::{label}" for label in labels)
            if key not in lookup
        ]
        if missing:
            shown = ", ".join(missing[:6])
            self._warn(
                "Missing 1/f background fits for some condition/channel pairs; "
                f"using zero background for: {shown}"
                + (" ..." if len(missing) > 6 else "")
            )

        return lookup

    def _subtract_background_from_power_psds(self, best_fit_df: Any) -> None:
        """Replace PSD caches with background-removed dB residuals"""
        first_psd = np.asarray(next(iter(self.psds_power_dict.values())), dtype=float)
        n_channels, n_bins = first_psd.shape
        labels = self._channel_labels_for_psd(n_channels)
        lookup = self._background_lookup_from_best_fit(best_fit_df, labels)

        corrected_dict: Dict[str, np.ndarray] = {}
        corrected_epochs_dict: Dict[str, np.ndarray] = {}

        for batch, psd_power in self.psds_power_dict.items():
            condition = self._condition_from_epoch_key(batch)
            psd_db = self._power_to_db(psd_power)
            residual = np.empty_like(psd_db, dtype=float)

            for ch_idx, label in enumerate(labels):
                background = lookup.get(f"{condition}::{label}")
                if background is None:
                    background = np.zeros(n_bins, dtype=float)
                residual[ch_idx, :] = psd_db[ch_idx, :] - background

            corrected_dict[batch] = residual

        for batch, psd_power_epochs in self.psds_power_epochs_dict.items():
            condition = self._condition_from_epoch_key(batch)
            psd_db_epochs = self._power_to_db(psd_power_epochs)
            residual_epochs = np.empty_like(psd_db_epochs, dtype=float)

            for ch_idx, label in enumerate(labels):
                background = lookup.get(f"{condition}::{label}")
                if background is None:
                    background = np.zeros(n_bins, dtype=float)
                residual_epochs[:, ch_idx, :] = psd_db_epochs[:, ch_idx, :] - background[np.newaxis, :]

            corrected_epochs_dict[batch] = residual_epochs

        self.psds_dict = corrected_dict
        self.psds_epochs_dict = corrected_epochs_dict
        self.psd_values_are_db = True
        self.psd_value_units = "dB"
        self.psd_value_label = "1/f-subtracted PSD residual (dB)"
        self.parameter_summary["psd_value_units"] = self.psd_value_units
        self.parameter_summary["psd_value_label"] = self.psd_value_label

    def _apply_one_over_f_background_subtraction(self) -> None:
        """Fit and subtract the selected 1/f background from all PSD caches"""
        input_csv = self._write_background_fit_input_csv()
        results = self._fit_one_over_f_background(input_csv=input_csv)
        best_fit_df = results["best_fit_df"]

        self._subtract_background_from_power_psds(best_fit_df)

        fit_summary_df = results.get("fit_summary_df")
        selected_model_counts: Dict[str, int] = {}
        if fit_summary_df is not None and not fit_summary_df.empty:
            selected = fit_summary_df[fit_summary_df.get("is_selected_best_model") == True]
            selected_model_counts = {
                str(key): int(value)
                for key, value in selected["model_name"].value_counts().to_dict().items()
            }

        for key in (
            "aggregated_csv",
            "best_fit_csv",
            "fit_summary_csv",
            "peak_summary_csv",
            "peak_group_summary_csv",
        ):
            path = results.get(key)
            if path is not None:
                self._track_export(str(path))

        for path in results.get("plot_paths", []):
            self._track_export(str(path))

        self.background_fit_outputs = {
            "input_csv": input_csv,
            "aggregated_csv": str(results.get("aggregated_csv")),
            "best_fit_csv": str(results.get("best_fit_csv")),
            "fit_summary_csv": str(results.get("fit_summary_csv")),
            "peak_summary_csv": str(results.get("peak_summary_csv")),
            "peak_group_summary_csv": str(results.get("peak_group_summary_csv")),
            "selected_model_counts": selected_model_counts,
            "n_condition_channel_fits": int(
                best_fit_df[["condition", "channel"]].drop_duplicates().shape[0]
            ),
        }
        self.parameter_summary["one_over_f_background_fit"] = self.background_fit_outputs

    def _grab_left_right_electrodes(self) -> Tuple[List[str], List[str]]:
        """Split motor-imagery channels into left/right hemispheres using montage x-coordinates and store masks"""
        
        dict_chs = self.raw.get_montage().get_positions()['ch_pos']
        # Consider all left and right channels
        chRight = [key for key in dict_chs.keys() if dict_chs[ key ][0] > 0]
        chLeft  = [key for key in dict_chs.keys() if dict_chs[ key ][0] < 0]
        # Filter only the target channels
        chRight = [ch for ch in chRight if ch in self.ch_motor_imagery]
        chLeft  = [ch for ch in chLeft  if ch in self.ch_motor_imagery]
        # Convert ch_set channels into an array of True of False based on the ones to consider
        isRight = [True if x in chRight else False for x in self.ch_set.get_labels()]
        isLeft  = [True if x in chLeft  else False for x in self.ch_set.get_labels()]
        
        self.chRight = np.array(chRight)
        self.chLeft  = np.array(chLeft)
        
        self.isRight = np.array(isRight)
        self.isLeft  = np.array(isLeft )
            
    def _calculate_r(
        self,
        x: np.ndarray,
        is_treatment: np.ndarray,
    ) -> np.ndarray:
        """Compute point-biserial correlation between features and a binary label"""
        x = np.asarray(x)
        is_treatment = np.asarray(is_treatment).astype(bool)

        y = self.dummy(is_treatment)

        x_mean = np.mean(x, axis=0)
        y_mean = np.mean(y)

        numerator = np.sum((x - x_mean) * (y - y_mean)[:, np.newaxis], axis=0)

        x_diff_sq = np.sum((x - x_mean) ** 2, axis=0)
        y_diff_sq = np.sum((y - y_mean) ** 2)

        denominator = np.sqrt(x_diff_sq * y_diff_sq) + 1e-12
        return numerator / denominator

    def _calculate_r2(
        self,
        x: np.ndarray,
        is_treatment: np.ndarray,
        signed: bool = True,
    ) -> np.ndarray:
        """Signed or unsigned r^2 from the correlation coefficient"""
        r = self._calculate_r(x, is_treatment)
        return r * np.abs(r) if signed else r * r

    def _transform_effect(
        self,
        x: np.ndarray,
        is_treatment: np.ndarray,
        transf: str = "r2",
    ) -> np.ndarray:
        """Apply an effect transform across channels"""
        transf = (transf or "r2").lower()
        if transf == "eta2":
            return self.calculate_eta2(x=x, is_treatment=is_treatment)
        if transf == "r2":
            return self._calculate_r2(x=x, is_treatment=is_treatment)
        raise ValueError(f"Unknown transform: {transf}")

    def _approx_permutation_test(
        self,
        x: np.ndarray,
        is_treatment: np.ndarray,
        stat,
        rng: Optional[np.random.Generator] = None,
    ) -> Dict[str, Any]:
        """One-sided approximate permutation test"""
        n_simulations = self.nSim
        rng = rng or np.random.default_rng()
        is_treatment = np.asarray(is_treatment).astype(bool)

        observed = stat(x, is_treatment)
        hist = [stat(x, self.shuffle(is_treatment, rng=rng)) for _ in range(n_simulations)]

        n_reached = int(np.sum(np.asarray(hist) > observed))
        p = (0.5 + n_reached) / (1.0 + n_simulations)

        return {
            "observed": float(observed),
            "p": float(p),
            "p_interval": self.pvalue_interval(p, n_simulations),
            "neg_ln_p": self.neg_p(p),
            "null_distribution": np.asarray(hist),
            "n_simulations": int(n_simulations),
        }

    def _bootstrap_test(
        self,
        x: np.ndarray,
        is_treatment: np.ndarray,
        stat,
        null_hypothesis_value: float = 0.0,
        rng: Optional[np.random.Generator] = None,
    ) -> Dict[str, Any]:
        """Bootstrap test comparing observed stat to a null-hypothesis value"""
        n_simulations = self.nSim
        rng = rng or np.random.default_rng()
        is_treatment = np.asarray(is_treatment).astype(bool)

        observed = stat(x, is_treatment)
        hist = [
            stat(self.bootstrap_resample(x, is_treatment, rng=rng), is_treatment)
            for _ in range(n_simulations)
        ]

        n_reached = int(np.sum(np.asarray(hist) < null_hypothesis_value))
        p = (0.5 + n_reached) / (1.0 + n_simulations)

        return {
            "observed": float(observed),
            "p": float(p),
            "p_interval": self.pvalue_interval(p, n_simulations),
            "neg_ln_p": self.neg_p(p),
            "null_distribution": np.asarray(hist),
            "n_simulations": int(n_simulations),
            "null_value": float(null_hypothesis_value),
        }

    def _build_symmetry_dict(self, tol: float = 1e-6) -> Dict[str, str]:
        """Build a best-effort left/right symmetry mapping using montage positions"""
        montage = self.raw.get_montage()
        if montage is None:
            raise RuntimeError("RAW object must have a montage to build symmetry mapping.")

        ch_pos = montage.get_positions().get("ch_pos", {})
        pos = {k.lower(): np.asarray(v) for k, v in ch_pos.items()}

        names = list(pos.keys())
        coords = np.stack([pos[n] for n in names], axis=0)

        symm: Dict[str, str] = {}

        for i, name in enumerate(names):
            x, y, z = coords[i]
            if abs(x) <= tol:
                continue

            target = np.array([-x, y, z])
            dists = np.linalg.norm(coords - target, axis=1)
            j = int(np.argmin(dists))

            # Loose tolerance because head models are never perfect
            if dists[j] <= 5e-2:
                symm[name] = names[j]

        return symm

    def _find_symmetric_indices(
        self,
        is_contralat: np.ndarray,
        symm_dict: Dict[str, str],
    ) -> np.ndarray:
        """Convert contralateral channel mask to ipsilateral channel indices"""
        ch_names = np.array(self.ch_set.get_labels())
        contra_names = [c.lower() for c in ch_names[is_contralat]]

        ipsi_names = [symm_dict[ch] for ch in contra_names if ch in symm_dict]
        return self.ch_set.find_labels(ipsi_names)

    def _difference_of_sums_effect(
        self,
        x: np.ndarray,
        is_treatment: np.ndarray,
        is_contralat: np.ndarray,
        symm_dict: Dict[str, str],
        transf: str = "r2",
    ) -> float:
        """Compute ipsilateral minus contralateral sum of transformed effects"""
        eff = self._transform_effect(x, is_treatment, transf=transf)

        contra_vals = eff[is_contralat]
        ipsi_idx = self._find_symmetric_indices(is_contralat, symm_dict)
        ipsi_vals = eff[ipsi_idx]

        return float(np.sum(ipsi_vals) - np.sum(contra_vals))

    def _stack_psds_by_prefix(self, prefix: str) -> np.ndarray:
        """
        Collect PSDs matching a base label followed by an integer suffix

        - "left" matches: left_1, left_2, ...
        - "left_rest" matches: left_rest_1, left_rest_2, ...
        """
        if not hasattr(self, "psds_dict"):
            raise RuntimeError("PSDs not generated. Call _generate_psds first.")

        base = prefix.lower().rstrip("_").strip()
        pattern = re.compile(rf"^{re.escape(base)}_(\d+)$", re.IGNORECASE)

        matches = []
        for key in self.psds_dict.keys():
            k = key.strip()
            m = pattern.match(k)
            if m:
                matches.append((int(m.group(1)), key))

        matches.sort(key=lambda x: x[0])

        if not matches:
            raise KeyError(
                f"No PSD entries found for base '{base}'. "
                f"Available keys: {sorted(self.psds_dict.keys())}"
            )

        return np.stack([self.psds_dict[k] for _, k in matches], axis=0)

    def _run_lateralized_band_stats(
        self,
        task_prefix: str,
        control_prefix: str,
        transf: str = "r2",
        rng: Optional[np.random.Generator] = None,
    ) -> Dict[str, Any]:
        """High-level motor imagery stats for task vs control"""
        rng = rng or np.random.default_rng()

        bands = self.freq_bands

        # Normalize bands into a list of (low, high) tuples
        if isinstance(bands, tuple) and len(bands) == 2:
            band_list = [tuple(map(float, bands))]
            band_edges = None
        else:
            band_edges = list(map(float, bands))
            if len(band_edges) < 2:
                raise ValueError("freq_bands must contain at least two values")
            if any(np.diff(band_edges) <= 0):
                raise ValueError("freq_bands must be strictly increasing")
            
            #band_list = list(zip(band_edges[:-1], band_edges[1:]))
            band_list = self.bands_from_edges(band_edges)

        psd_task = self._stack_psds_by_prefix(task_prefix   )
        psd_ctrl = self._stack_psds_by_prefix(control_prefix)

        #if psd_task.shape != psd_ctrl.shape:
        #    raise ValueError(
        #        f"Mismatched PSD shapes for '{task_prefix}' and '{control_prefix}': "
        #        f"{psd_task.shape} vs {psd_ctrl.shape}"
        #    )

        n_blocks_task, n_ch, n_bins = psd_task.shape
        n_blocks_ctrl, n_ch, n_bins = psd_ctrl.shape

        # Use the exact frequency axis returned by MNE
        bins = np.asarray(getattr(self, "freqs", []), dtype=float)
        if bins.size != n_bins:
            raise RuntimeError(
                f"PSD frequency vector length ({bins.size}) does not match PSD bins ({n_bins})"
            )

        # Decide contralateral side from task prefix
        tp = task_prefix.lower().rstrip("_")
        if tp.startswith("left"):
            is_contralat = self.isRight
        else:
            is_contralat = self.isLeft

        symm_dict = self._build_symmetry_dict()

        # Build trial matrix once
        x_task_all = psd_task
        x_ctrl_all = psd_ctrl
        
        # Keep the original labeling convention used elsewhere:
        # True for "rest/control", False for "task"
        is_treatment = np.array([True] * n_blocks_ctrl + [False] * n_blocks_task)

        band_results: Dict[str, Any] = {}

        for low_f, high_f in band_list:
            is_last_band = (low_f, high_f) == tuple(band_list[-1])
            band_idx = self._band_indices(bins, low_f, high_f, is_last=is_last_band)
            if band_idx.size == 0:
                raise ValueError(f"No frequency bins fall inside band [{low_f}, {high_f}) Hz")
            start = int(band_idx[0])
            stop = int(band_idx[-1]) + 1

            x_task = np.mean(x_task_all[:, :, start:stop], axis=2)
            x_ctrl = np.mean(x_ctrl_all[:, :, start:stop], axis=2)
            
            # Convert raw power to dB for the original scheme; residual runs are already dB
            x_task_values = self._psd_values_for_analysis(x_task)
            x_ctrl_values = self._psd_values_for_analysis(x_ctrl)
            
            # Stack trials: task first, control second
            x = np.concatenate([x_ctrl_values, x_task_values], axis=0)

            stat_fn = lambda xx, it: self._difference_of_sums_effect(
                xx, it, is_contralat=is_contralat, symm_dict=symm_dict, transf=transf
            )

            perm = self._approx_permutation_test(
                x=x,
                is_treatment=is_treatment,
                stat=stat_fn,
                rng=rng,
            )
            boot = self._bootstrap_test(
                x=x,
                is_treatment=is_treatment,
                stat=stat_fn,
                null_hypothesis_value=0.0,
                rng=rng,
            )

            effect_per_ch = self._transform_effect(x, is_treatment, transf=transf)

            band_key = f"{low_f:g}-{high_f:g}Hz"
            band_results[band_key] = {
                "band": (low_f, high_f),
                "band_idx": band_idx,
                "x_task": x_task_values,
                "x_control": x_ctrl_values,
                "effect_per_channel": effect_per_ch,
                "permutation": perm,
                "bootstrap": boot,
                "psd_processing": self.psd_processing,
                "psd_value_units": self.psd_value_units,
            }

            #is_contralat_chs = np.array(self.ch_set.get_labels())[is_contralat]
            #keys = [key for key in symm_dict.keys() if key in is_contralat_chs]
            #symm_dict_filt = {}
            #for key in keys: symm_dict_filt.update({key, symm_dict[key]})

        return {
            "task_prefix": task_prefix,
            "control_prefix": control_prefix,
            "bands_input": bands,
            "band_edges": band_edges,
            "bands_expanded": band_list,
            "transf": transf,
            "n_blocks_task_ctrl": [n_blocks_task, n_blocks_ctrl],
            "bins": bins,
            "is_treatment": is_treatment,
            "is_contralat": is_contralat,
            "symmetry_dict": symm_dict,
            "band_results": band_results,
        }

    def _run_stat_tests(
        self,
        transf: str = "r2",
        verbose: bool = False,
    ) -> Dict[str, Any]:
        """Run band-wise statistical tests (including permutation/bootstrap where configured) and store results for later plots"""
        results: Dict[str, Any] = {}

        comparisons = [
            ("left",  "left_rest",  "left_vs_left_rest"  ),
            ("right", "right_rest", "right_vs_right_rest"),
        ]

        for task_prefix, control_prefix, name in comparisons:
            try:
                res = self._run_lateralized_band_stats(
                    task_prefix=task_prefix,
                    control_prefix=control_prefix,
                    transf=transf,
                    rng=self._rng_for(f"stats:{name}:{transf}"),
                )
                results[name] = res

                if verbose:
                    # Print a compact summary per band
                    for bname, bres in res["band_results"].items():
                        p_perm = bres["permutation"]["p"]
                        p_boot = bres["bootstrap"]["p"]
                        obs    = bres["permutation"]["observed"]
                        print(
                            f"[{name}] {bname} transf={transf} "
                            f"obs={obs:.3f} p_perm={p_perm:.3f} p_boot={p_boot:.3f}"
                        )

            except (KeyError, ValueError, RuntimeError) as exc:
                if verbose:
                    print(f"[{name}] skipped: {exc}")

        if not results:
            self._warn("No lateralized statistical comparisons were completed")

        self.stats_results = results
        return results
    

    def _condition_from_epoch_key(self, key: str) -> str:
        """Return base condition from a numbered epoch key"""
        key = str(key).lower().strip()
        for prefix in ("left_rest", "right_rest", "left", "right"):
            if re.match(rf"^{prefix}_\d+$", key):
                return prefix
        return "other"

    def _summarize_trial_quality(self) -> List[Dict[str, Any]]:
        """Summarize candidate, kept, and dropped epochs per condition"""
        try:
            events_from_annot, event_dict = mne.events_from_annotations(self.raw, verbose=False)
        except Exception as exc:
            self._warn(f"Could not summarize trial quality: {exc}")
            self.trial_quality_summary = []
            return []

        grouped: Dict[str, Dict[str, int]] = {}
        for key, event_id in getattr(self, "event_ids", {}).items():
            condition = self._condition_from_epoch_key(key)
            row = grouped.setdefault(condition, {"candidate_epochs": 0, "kept_epochs": 0})
            row["candidate_epochs"] += int(np.sum(events_from_annot[:, 2] == int(event_id)))
            row["kept_epochs"] += int(len(self.epochs_dict.get(key, [])))

        rows: List[Dict[str, Any]] = []
        for condition in ("left", "right", "left_rest", "right_rest", "other"):
            if condition not in grouped:
                continue
            candidate = int(grouped[condition]["candidate_epochs"])
            kept = int(grouped[condition]["kept_epochs"])
            dropped = max(candidate - kept, 0)
            dropped_pct = 100.0 * dropped / candidate if candidate else np.nan
            rows.append({
                "condition": condition,
                "candidate_epochs": candidate,
                "kept_epochs": kept,
                "dropped_epochs": dropped,
                "dropped_pct": float(dropped_pct),
            })

        total_candidate = int(sum(r["candidate_epochs"] for r in rows))
        total_kept = int(sum(r["kept_epochs"] for r in rows))
        total_dropped = max(total_candidate - total_kept, 0)
        rows.append({
            "condition": "total",
            "candidate_epochs": total_candidate,
            "kept_epochs": total_kept,
            "dropped_epochs": total_dropped,
            "dropped_pct": 100.0 * total_dropped / total_candidate if total_candidate else np.nan,
        })
        self.trial_quality_summary = rows
        return rows

    @staticmethod
    def _benjamini_hochberg(p_values: List[float]) -> List[float]:
        """Benjamini-Hochberg FDR correction"""
        p = np.asarray(p_values, dtype=float)
        out = np.full_like(p, np.nan, dtype=float)
        valid = np.where(np.isfinite(p))[0]
        if valid.size == 0:
            return out.tolist()

        pv = p[valid]
        order = np.argsort(pv)
        ranked = pv[order]
        m = float(len(ranked))
        adjusted = ranked * m / np.arange(1, len(ranked) + 1)
        adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
        adjusted = np.clip(adjusted, 0.0, 1.0)
        restored = np.empty_like(adjusted)
        restored[order] = adjusted
        out[valid] = restored
        return out.tolist()

    def _apply_fdr_to_stats(self) -> None:
        """Add FDR-corrected p-values to permutation and bootstrap results"""
        if not hasattr(self, "stats_results") or not self.stats_results:
            return

        for test_name in ("permutation", "bootstrap"):
            records = []
            p_values = []
            for comp_name, comp_res in self.stats_results.items():
                for band_key, bres in (comp_res.get("band_results") or {}).items():
                    test = bres.get(test_name, {})
                    records.append((comp_name, band_key, test_name))
                    p_values.append(float(test.get("p", np.nan)))

            adjusted = self._benjamini_hochberg(p_values)
            for (comp_name, band_key, tname), p_adj in zip(records, adjusted):
                self.stats_results[comp_name]["band_results"][band_key][tname]["p_fdr"] = float(p_adj)
                self.stats_results[comp_name]["band_results"][band_key][tname]["neg_ln_p_fdr"] = self.neg_p(p_adj)

    def _effect_direction_label(self, observed: float) -> str:
        """Explain the observed sign of the lateralized statistic"""
        if not np.isfinite(observed):
            return "unavailable"
        if observed > 0:
            return "ipsilateral effect > contralateral effect"
        if observed < 0:
            return "contralateral effect > ipsilateral effect"
        return "no lateralized difference"

    def _build_main_result_summary(self) -> List[Dict[str, Any]]:
        """Build compact strongest-effect rows for report and CSV export"""
        rows: List[Dict[str, Any]] = []
        if not hasattr(self, "stats_results") or not self.stats_results:
            self.main_result_summary = rows
            return rows

        for comp_name, comp_res in self.stats_results.items():
            candidate_rows = []
            for band_key, bres in (comp_res.get("band_results") or {}).items():
                boot = bres.get("bootstrap", {})
                perm = bres.get("permutation", {})
                observed = float(boot.get("observed", np.nan))
                candidate_rows.append({
                    "comparison": comp_name,
                    "band": band_key,
                    "observed_delta": observed,
                    "bootstrap_p": float(boot.get("p", np.nan)),
                    "bootstrap_p_fdr": float(boot.get("p_fdr", np.nan)),
                    "permutation_p": float(perm.get("p", np.nan)),
                    "permutation_p_fdr": float(perm.get("p_fdr", np.nan)),
                    "direction": self._effect_direction_label(observed),
                    "n_task_blocks": int(comp_res.get("n_blocks_task_ctrl", [0, 0])[0]),
                    "n_control_blocks": int(comp_res.get("n_blocks_task_ctrl", [0, 0])[1]),
                })

            if candidate_rows:
                candidate_rows.sort(key=lambda r: (np.nan_to_num(r["bootstrap_p_fdr"], nan=1.0), -abs(np.nan_to_num(r["observed_delta"], nan=0.0))))
                rows.append(candidate_rows[0])

        self.main_result_summary = rows
        return rows

    def _run_decoding_analysis(self) -> List[Dict[str, Any]]:
        """Run optional task-vs-rest decoding for each comparison, band, and ROI"""
        try:
            from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
            from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, roc_auc_score
            from sklearn.model_selection import StratifiedKFold, cross_val_predict
            from sklearn.pipeline import make_pipeline
            from sklearn.preprocessing import StandardScaler
        except Exception as exc:
            self._warn(f"Skipping decoding because scikit-learn is unavailable: {exc}")
            self.decoding_results = []
            return []

        if not hasattr(self, "stats_results") or not self.stats_results:
            self.decoding_results = []
            return []

        labels = [lbl.lower() for lbl in self.ch_set.get_labels()]
        mi_set = {ch.lower() for ch in self.ch_motor_imagery}

        def _roi_indices(comp_res: Dict[str, Any], roi: str) -> List[int]:
            task_prefix = str(comp_res.get("task_prefix", "")).lower().rstrip("_")
            if task_prefix.startswith("left"):
                contra_names = {ch.lower() for ch in self.chRight}
                ipsi_names = {ch.lower() for ch in self.chLeft}
            else:
                contra_names = {ch.lower() for ch in self.chLeft}
                ipsi_names = {ch.lower() for ch in self.chRight}

            if roi == "contra":
                roi_set = contra_names
            elif roi == "ipsi":
                roi_set = ipsi_names
            else:
                roi_set = mi_set

            return [i for i, lbl in enumerate(labels) if lbl in mi_set and lbl in roi_set]

        def _decode_one(
            x_ctrl: np.ndarray,
            x_task: np.ndarray,
            comp_name: str,
            band_key: str,
            roi_name: str,
        ) -> Optional[Dict[str, Any]]:
            x = np.vstack([x_ctrl, x_task])
            y = np.array([0] * len(x_ctrl) + [1] * len(x_task), dtype=int)

            class_counts = np.bincount(y, minlength=2)
            min_class = int(np.min(class_counts))
            if min_class < 2:
                self._warn(f"Skipping decoding for {comp_name} {band_key} {roi_name}: fewer than 2 samples in one class")
                return None

            n_splits = int(min(5, min_class))
            cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)
            clf = make_pipeline(StandardScaler(), LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"))

            y_pred = cross_val_predict(clf, x, y, cv=cv, method="predict")
            try:
                y_score = cross_val_predict(clf, x, y, cv=cv, method="predict_proba")[:, 1]
                roc_auc = float(roc_auc_score(y, y_score))
            except Exception:
                roc_auc = float("nan")

            cm = confusion_matrix(y, y_pred, labels=[0, 1])
            self.decoding_confusion_matrices[(comp_name, band_key, roi_name)] = cm

            return {
                "comparison": comp_name,
                "band": band_key,
                "roi": roi_name,
                "classifier": "Shrinkage LDA",
                "cv": f"StratifiedKFold({n_splits})",
                "n_samples": int(len(y)),
                "n_control": int(class_counts[0]),
                "n_task": int(class_counts[1]),
                "n_features": int(x.shape[1]),
                "accuracy": float(accuracy_score(y, y_pred)),
                "balanced_accuracy": float(balanced_accuracy_score(y, y_pred)),
                "roc_auc": roc_auc,
                "tn": int(cm[0, 0]),
                "fp": int(cm[0, 1]),
                "fn": int(cm[1, 0]),
                "tp": int(cm[1, 1]),
            }

        rows: List[Dict[str, Any]] = []
        self.decoding_confusion_matrices: Dict[Tuple[str, str, str], np.ndarray] = {}

        for comp_name, comp_res in self.stats_results.items():
            for band_key, bres in (comp_res.get("band_results") or {}).items():
                for roi_name in ("contra", "ipsi"):
                    roi_idx = _roi_indices(comp_res, roi_name)
                    if not roi_idx:
                        self._warn(f"Skipping decoding for {comp_name} {band_key} {roi_name}: no ROI channels found")
                        continue

                    x_ctrl = np.asarray(bres.get("x_control"), dtype=float)[:, roi_idx]
                    x_task = np.asarray(bres.get("x_task"), dtype=float)[:, roi_idx]

                    try:
                        row = _decode_one(x_ctrl, x_task, comp_name, band_key, roi_name)
                        if row is not None:
                            rows.append(row)
                    except Exception as exc:
                        self._warn(f"Decoding failed for {comp_name} {band_key} {roi_name}: {exc}")

        self.decoding_results = rows
        return rows

    def _plot_decoding_summary(self) -> Optional[Tuple[Figure, Axes]]:
        """Plot decoding balanced accuracy per comparison, band, and ROI"""
        rows = getattr(self, "decoding_results", [])
        if not rows:
            return None

        def _band_start(row: Dict[str, Any]) -> float:
            band = str(row.get("band", ""))
            return float(band.split("-")[0]) if "-" in band else 0.0

        groups: List[Tuple[str, str]] = []
        for row in sorted(rows, key=lambda r: (str(r.get("comparison", "")), _band_start(r))):
            key = (str(row.get("comparison", "")), str(row.get("band", "")))
            if key not in groups:
                groups.append(key)

        y = np.arange(len(groups))
        labels = [f"{comp.replace('_vs_', ' vs ')}\n{band}" for comp, band in groups]
        values: Dict[Tuple[str, str, str], float] = {}
        for row in rows:
            values[(str(row.get("comparison", "")), str(row.get("band", "")), str(row.get("roi", "both")))] = float(row.get("balanced_accuracy", np.nan))

        fig, ax = plt.subplots(1, 1, figsize=(8, max(3.2, 0.55 * len(groups))))
        offsets = {"contra": -0.13, "ipsi": 0.13}
        colors = {"contra": "tab:blue", "ipsi": "tab:orange"}
        labels_done: set[str] = set()

        for roi_name in ("contra", "ipsi"):
            roi_values = [values.get((comp, band, roi_name), np.nan) for comp, band in groups]
            label = f"{roi_name} ROI" if roi_name not in labels_done else None
            ax.barh(y + offsets[roi_name], roi_values, height=0.24, label=label, color=colors[roi_name], alpha=0.85)
            labels_done.add(roi_name)

        ax.axvline(0.5, linestyle="--", linewidth=1, color="0.3", label="chance")
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_xlim(0.0, 1.0)
        ax.set_xlabel("Cross-validated balanced accuracy")
        ax.set_title("Task-vs-rest decoding by ROI", loc="left")
        ax.legend(frameon=False, loc="lower right")
        ax.grid(True, axis="x", linestyle=":", alpha=0.35)
        fig.subplots_adjust(bottom=0.18)
        fig.text(
            0.01,
            -0.04,
            "Features: band-averaged PSD from contra or ipsi motor ROI channels\nValues above 0.50 indicate task/rest separation beyond chance",
            ha="left",
            va="bottom",
            fontsize=8,
        )
        self._savefig(fig, "decoding_summary")
        return fig, ax

    def _plot_decoding_confusion_best(self) -> Optional[Tuple[Figure, Axes]]:
        """Plot the confusion matrix for the strongest decoding result"""
        rows = getattr(self, "decoding_results", [])
        matrices = getattr(self, "decoding_confusion_matrices", {})
        if not rows or not matrices:
            return None

        best = sorted(rows, key=lambda r: np.nan_to_num(r.get("balanced_accuracy", np.nan), nan=-1.0), reverse=True)[0]
        key = (best["comparison"], best["band"], best.get("roi", "both"))
        cm = matrices.get(key)
        if cm is None:
            return None

        fig, ax = plt.subplots(1, 1, figsize=(4.6, 4.3))
        im = ax.imshow(cm)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Pred rest", "Pred task"])
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["True rest", "True task"])
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(int(cm[i, j])), ha="center", va="center")
        ax.set_title(
            f"Best decoding: {best['comparison']} {best['band']} {best.get('roi', 'both')} ROI",
            loc="left",
            fontsize=10,
        )
        ax.set_xlabel("Cross-validated prediction")
        ax.set_ylabel("True label")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        self._savefig(fig, "decoding_confusion_best")
        return fig, ax

    def _export_all_csv(self) -> None:
        """Export all analysis tables in a consistent, report-friendly form"""
        self._export_parameter_csv()
        self._export_paradigm_validation_csv()
        self._export_trial_quality_csv()
        self._export_psd_csv()
        self._export_psd_long_csv()
        self._export_r2_csv()
        self._export_stat_summary_csv()
        self._export_bootstrap_csv()
        self._export_erd_ers_csv()
        self._export_decoding_csv()

    def _write_csv_rows(self, filename: str, rows: List[Dict[str, Any]], fieldnames: Optional[List[str]] = None) -> Optional[str]:
        """Write a list of dictionaries to CSV"""
        if self.save_path is None or not rows:
            return None
        if fieldnames is None:
            fieldnames = list(rows[0].keys())
        fname = self._csv_path(filename)
        if fname is None:
            return None
        with open(fname, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow({k: row.get(k, "") for k in fieldnames})
        self._track_export(fname)
        print(f"[CSV] Saved → {fname}")
        return fname

    def _export_parameter_csv(self) -> None:
        """Export run parameters"""
        rows = [{"parameter": k, "value": json.dumps(v) if isinstance(v, (list, dict)) else v} for k, v in self.parameter_summary.items()]
        self._write_csv_rows("analysis_parameters.csv", rows, ["parameter", "value"])

    def _export_paradigm_validation_csv(self) -> None:
        """Export inferred paradigm label validation"""
        self._write_csv_rows("paradigm_validation_summary.csv", getattr(self, "paradigm_event_summary", []), ["label", "count", "status"])

    def _export_trial_quality_csv(self) -> None:
        """Export per-condition epoch drop summary"""
        self._write_csv_rows(
            "trial_quality_summary.csv",
            getattr(self, "trial_quality_summary", []),
            ["condition", "candidate_epochs", "kept_epochs", "dropped_epochs", "dropped_pct"],
        )

    def _export_stat_summary_csv(self) -> None:
        """Export compact band-wise statistical summary with FDR-corrected p-values"""
        rows: List[Dict[str, Any]] = []
        for comp_name, comp_res in getattr(self, "stats_results", {}).items():
            n_task, n_ctrl = comp_res.get("n_blocks_task_ctrl", [np.nan, np.nan])
            for band_key, bres in (comp_res.get("band_results") or {}).items():
                boot = bres.get("bootstrap", {})
                perm = bres.get("permutation", {})
                dist = np.asarray(boot.get("null_distribution", []), dtype=float)
                ci_low, ci_high = self._percentile_ci(dist, ci=95.0) if dist.size else (np.nan, np.nan)
                observed = float(boot.get("observed", np.nan))
                rows.append({
                    "comparison": comp_name,
                    "band": band_key,
                    "observed_delta": observed,
                    "direction": self._effect_direction_label(observed),
                    "bootstrap_ci_low": ci_low,
                    "bootstrap_ci_high": ci_high,
                    "bootstrap_p": float(boot.get("p", np.nan)),
                    "bootstrap_p_fdr": float(boot.get("p_fdr", np.nan)),
                    "permutation_p": float(perm.get("p", np.nan)),
                    "permutation_p_fdr": float(perm.get("p_fdr", np.nan)),
                    "neg_ln_bootstrap_p": float(boot.get("neg_ln_p", np.nan)),
                    "neg_ln_bootstrap_p_fdr": float(boot.get("neg_ln_p_fdr", np.nan)),
                    "n_task_blocks": int(n_task),
                    "n_control_blocks": int(n_ctrl),
                    "n_simulations": int(boot.get("n_simulations", self.nSim)),
                })
        self._write_csv_rows("stat_summary_per_band.csv", rows)

    def _export_decoding_csv(self) -> None:
        """Export optional decoding metrics"""
        self._write_csv_rows("decoding_summary.csv", getattr(self, "decoding_results", []))

    def _export_erd_ers_csv(self) -> None:
        """Export ERD/ERS time-course values for common mu and beta bands"""
        rows: List[Dict[str, Any]] = []
        for task, rest in (("left", "left_rest"), ("right", "right_rest")):
            for band in ((8.0, 13.0), (13.0, 31.0)):
                try:
                    curve = self._compute_within_trial_erd_ers(task, rest, band=band, roi="contra")
                except Exception as exc:
                    self._warn(f"Could not export ERD/ERS for {task} {band}: {exc}")
                    continue
                for i, t in enumerate(curve["time_sec"]):
                    value = float(curve["mean_pct"][i])
                    rows.append({
                        "task": task,
                        "rest": rest,
                        "band": f"{band[0]:g}-{band[1]:g}Hz",
                        "roi": "contra",
                        "time_sec": float(t),
                        "erd_ers_pct": value,
                        "erd_ers_value": value,
                        "erd_ers_unit": curve.get("value_unit", "%"),
                        "ci_low": float(curve["ci_low"][i]),
                        "ci_high": float(curve["ci_high"][i]),
                        "n_trials": int(curve["n_trials"]),
                        "psd_processing": curve.get("psd_processing", self.psd_processing),
                    })
        self._write_csv_rows("erd_ers_timecourse.csv", rows)

    def _export_report_metadata_json(self) -> None:
        """Export report and analysis metadata alongside the PDF"""
        if self.save_path is None:
            return
        payload = {
            "parameters": self.parameter_summary,
            "paradigm_validation": getattr(self, "paradigm_event_summary", []),
            "trial_quality": getattr(self, "trial_quality_summary", []),
            "main_results": getattr(self, "main_result_summary", []),
            "decoding_results": getattr(self, "decoding_results", []),
            "psd_processing": self.psd_processing,
            "psd_value_units": self.psd_value_units,
            "background_fit": getattr(self, "background_fit_outputs", {}),
            "generated_figures": [self._relative_output_path(x) for x in self.generated_figures],
            "exported_files": [self._relative_output_path(x) for x in self.exported_files],
            "warnings": self.analysis_warnings,
        }
        path = os.path.join(self.save_path, "report_metadata.json")
        with open(path, "w") as fh:
            json.dump(payload, fh, indent=2)
        self._track_export(path)
        self._info(f"Metadata saved → {path}")

    # Plotting utilities
    def _ensure_stats_results(self, transf: str = "r2") -> Dict[str, Any]:
        """Ensure that lateralized stats results exist on the instance"""
        if hasattr(self, "stats_results") and isinstance(self.stats_results, dict):
            if len(self.stats_results.keys()) > 0:
                return self.stats_results

        return self._run_stat_tests(transf=transf, verbose=False)

    def _plot_permutation_distribution(
        self,
        perm: Dict[str, Any],
        ax,
        xlabel: str = r"$\Delta$",
        title: Optional[str] = None,
    ) -> None:
        """Plot a permutation null distribution on a provided axis"""
        hist     = np.asarray(perm.get("null_distribution", []))
        observed = perm.get("observed", None)
        p        = perm.get("p", None)
        n_sim    = int(perm.get("n_simulations", len(hist) or 0))

        if hist.size:
            ax.hist(hist, label=f"N = {n_sim}", color='deepskyblue')
        if observed is not None:
            ax.axvline(float(observed), color="black", label=f"Obs: {float(observed):.3f}")

        ax.set_xlabel(xlabel, loc="right")

        if p is not None:
            ax.legend(
                title=f"Permutation (p = {float(p):.3f})",
                loc="upper left",
                frameon=False,
            )
        else:
            ax.legend(loc="upper left", frameon=False)

        if title:
            ax.set_title(title, loc="left")

    def _plot_bootstrap_distribution(
        self,
        boot: Dict[str, Any],
        ax,
        xlabel: str = r"$\Delta$",
        title: Optional[str] = None,
    ) -> None:
        """Plot a bootstrap null distribution on a provided axis"""
        hist = np.asarray(boot.get("null_distribution", []))
        observed = boot.get("observed", None)
        null_val = boot.get("null_value", 0.0)
        p = boot.get("p", None)
        n_sim = int(boot.get("n_simulations", len(hist) or 0))

        if hist.size:
            ax.hist(hist, label=f"N = {n_sim}", color="limegreen")
        if observed is not None:
            ax.axvline(float(observed), color="black", label=f"Obs: {float(observed):.3f}")
        if null_val is not None:
            #ax.axvline(float(null_val), color="red", label=f"$H_0$: $\Delta$ = {float(null_val):.1f}")
            ax.axvline(float(null_val), color="red", label=fr"$H_0$: $\Delta$ = {float(null_val):.1f}")

        ax.set_xlabel(xlabel, loc="right")

        if p is not None:
            ax.legend(
                title=f"Bootstrap (p = {float(p):.3f})",
                loc="upper left",
                frameon=False,
            )
        else:
            ax.legend(loc="upper left", frameon=False)

        if title:
            ax.set_title(title, loc="left")

    def _plot_test_distributions(
        self,
        comparison: str = "left_vs_left_rest",
        transf: str = "r2",
        results: Optional[Dict[str, Any]] = None,
        figsize: Optional[Tuple[float, float]] = None,
        suptitle: Optional[str] = None,
    ) -> Tuple[Figure, np.ndarray]:
        """
        Plot permutation and bootstrap distributions across frequency bands

        The layout is 2xN, with permutation on the first row and bootstrap on
        the second row, matching the structure of the legacy script

        Args:
            comparison: Key produced by _run_stat_tests
            transf: Effect transform name ("r2" or "eta2")
            results: Optional precomputed stats results dict
            figsize: Optional matplotlib figsize
            suptitle: Optional super-title for the figure

        Returns:
            (fig, axs): Matplotlib figure and axes array
        """
        results = results or self._ensure_stats_results(transf=transf)

        if comparison not in results:
            raise KeyError(
                f"Comparison '{comparison}' not found. Available: {list(results.keys())}"
            )

        comp_res = results[comparison]
        band_list = comp_res.get("bands_expanded") or []
        band_results = comp_res.get("band_results", {})

        n_bands = len(band_list) if band_list else len(band_results.keys())
        if n_bands == 0:
            raise RuntimeError(f"No band results found for comparison '{comparison}'.")

        figsize = figsize or (4 * n_bands, 6)
        fig, axs = plt.subplots(nrows=2, ncols=n_bands, figsize=figsize, squeeze=False)

        # Build a stable ordering of keys if we don't have band_list
        fallback_keys = list(band_results.keys())

        for i in range(n_bands):
            if band_list:
                low_f, high_f = band_list[i]
                bkey = self.band_key(low_f, high_f)
                band_label = f"[{low_f:g}-{high_f:g}] Hz"
            else:
                bkey = fallback_keys[i]
                band_label = bkey.replace("Hz", " Hz")

            bres = band_results.get(bkey, band_results.get(fallback_keys[i]))

            perm = (bres or {}).get("permutation", {})
            boot = (bres or {}).get("bootstrap"  , {})

            self._plot_permutation_distribution(
                perm=perm,
                ax=axs[0, i],
            )
            axs[0, i].text(
                0.98,
                0.97,
                band_label,
                va="top",
                ha="right",
                transform=axs[0, i].transAxes,
                fontsize=11,
            )

            self._plot_bootstrap_distribution(
                boot=boot,
                ax=axs[1, i],
            )
            axs[1, i].text(
                0.98,
                0.97,
                band_label,
                va="top",
                ha="right",
                transform=axs[1, i].transAxes,
                fontsize=11,
            )

        if suptitle:
            fig.suptitle(suptitle)
            
        fig.tight_layout()
        
        if self.save_path is not None:
            if 'left' in comparison:
                key = 'left'
            elif 'right' in comparison:
                key = 'right'
            
            self._savefig(fig, f"stat_distribution_{key}")
            
        return fig, axs       
            
    def _plot_all_test_distributions(
        self,
        transf: str = "r2",
        results: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Convenience wrapper to plot both left and right task comparisons
        Returns a dict mapping comparison name -> (fig, axs)
        """
        results = results or self._ensure_stats_results(transf=transf)

        out: Dict[str, Any] = {}
        for name in ("left_vs_left_rest", "right_vs_right_rest"):
            if name in results:
                out[name] = self._plot_test_distributions(
                    comparison=name,
                    transf=transf,
                    results=results,
                    suptitle=name.replace("_", " "),
                )
        return out
    
    def _build_topomap_masks(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Build boolean masks"""
        ch_names = np.array(self.ch_set.get_labels())
        bad_names = {name.lower() for name in self.raw.info.get("bads", [])}

        mask_bad = np.array([ch.lower() in bad_names for ch in ch_names])
        mask_right = np.asarray(self.isRight, dtype=bool)
        mask_left = np.asarray(self.isLeft, dtype=bool)

        return mask_bad, mask_right, mask_left

    def _plot_lateralized_topomaps(
        self,
        transf: str = "r2",
        results: Optional[Dict[str, Any]] = None,
        vlim: Optional[Tuple[float, float]] = (-0.3, 0.3),
        figsize: Optional[Tuple[float, float]] = None,
        suptitle: Optional[str] = None,
        bad_channels: Optional[List[str]] = None,
    ) -> Figure:
        """Plot signed-r² topomaps in a layout mimicking the legacy pipeline"""
        # Ensure results exist and extract left/right comparisons
        results = results or self._ensure_stats_results(transf=transf)
        left_res = results.get("left_vs_left_rest")
        right_res = results.get("right_vs_right_rest")

        if left_res is None or right_res is None:
            raise RuntimeError(
                "Both left_vs_left_rest and right_vs_right_rest results are "
                "required to plot lateralized topomaps."
            )

        # Custom blue-white-red colormap to match the old implementation
        cmap = self.build_custom_cmap(colors=["blue", "white", "red"])

        band_list   = left_res.get("bands_expanded") or []
        left_bands  = left_res.get("band_results", {})
        right_bands = right_res.get("band_results", {})

        # Decide how many bands we actually have
        n_bands = len(band_list) if band_list else max(
            len(left_bands), len(right_bands)
        )
        if n_bands == 0:
            raise RuntimeError("No band results found for topomap plotting.")

        # Color limits symmetric around zero by default
        if vlim is None:
            vlim = (-0.3, 0.3)
        vmin, vmax = vlim

        # Channel list for masks (use EEG channels from Raw)
        picks = mne.pick_types(
            self.raw.info,
            meg=False,
            eeg=True,
            stim=False,
            eog=False,
            misc=False,
            exclude=[],
        )
        ch_names = np.array(self.raw.info["ch_names"])[picks]
        ch_lower = np.char.lower(ch_names.astype(str))

        # Bad / interpolated channels
        if bad_channels is None:
            bad_channels = list(self.raw.info.get("bads", []))
        bad_set = {ch.lower() for ch in bad_channels}
        mask_bad = np.array([ch in bad_set for ch in ch_lower])

        # Target channels from montage-based left/right definition
        # self.chRight / self.chLeft are set in _grab_left_right_electrodes()
        left_set  = {ch.lower() for ch in getattr(self, "chLeft",  [])}
        right_set = {ch.lower() for ch in getattr(self, "chRight", [])}

        # For the LEFT task:
        #   target = right hemisphere motor-imagery channels (contralateral)
        mask_target_left = np.array([ch in right_set for ch in ch_lower])
        # For the RIGHT task:
        #   target = left hemisphere motor-imagery channels (contralateral)
        mask_target_right = np.array([ch in left_set for ch in ch_lower])

        # Sanity check: data must match number of EEG channels
        example_band = next(iter(left_bands.values()))
        example_data = np.asarray(example_band.get("effect_per_channel"))
        if example_data.shape[0] != ch_names.shape[0]:
            raise RuntimeError(
                "Mismatch between number of EEG channels in Raw and "
                "effect_per_channel length. Cannot safely plot topomaps"
            )

        # Figure & GridSpec: N rows, 3 columns (Left | Colorbar/Legend | Right)
        if figsize is None: figsize = (6.0, float(2 * n_bands))

        fig = plt.figure(figsize=figsize)
        gs = matplotlib.gridspec.GridSpec(
            n_bands,
            3,
            width_ratios=[2.0, 0.15, 2.0],
        )

        # Masks / marker styles to mirror the legacy implementation
        mask_params_bad = dict(
            marker="X",
            markersize=8,
            markerfacecolor="darkgrey",
        )
        mask_params_target = dict(
            marker="o",
            markersize=5,
            markerfacecolor="lime",
            alpha=0.75,
        )

        # For band key fallback when bands_expanded is missing
        fallback_left_keys  = list(left_bands.keys() )
        fallback_right_keys = list(right_bands.keys())

        axs_rows: List[List[plt.Axes]] = []

        for i in range(n_bands):
            # Resolve band key & label
            if band_list:
                low_f, high_f = band_list[i]
                bkey = self.band_key(low_f, high_f)
                band_label = bkey.replace("Hz", " Hz")
            else:
                bkey = (
                    fallback_left_keys[i]
                    if i < len(fallback_left_keys)
                    else None
                )
                band_label = (bkey or "").replace("Hz", " Hz")

            l_bres = left_bands.get(bkey)  if bkey else None
            r_bres = right_bands.get(bkey) if bkey else None

            if l_bres is None and i < len(fallback_left_keys):
                l_bres = left_bands[fallback_left_keys[i]]
                band_label = fallback_left_keys[i].replace("Hz", " Hz")

            if r_bres is None and i < len(fallback_right_keys):
                r_bres = right_bands[fallback_right_keys[i]]

            if l_bres is None or r_bres is None:
                # Nothing to plot for this band
                continue

            data_left  = np.asarray(l_bres.get("effect_per_channel"))
            data_right = np.asarray(r_bres.get("effect_per_channel"))

            # Double-check length again per band in case of pathological cases
            if data_left.shape[0] != ch_names.shape[0]:
                raise RuntimeError(
                    f"Band {band_label}: effect_per_channel has "
                    f"{data_left.shape[0]} entries, but Raw has "
                    f"{ch_names.shape[0]} EEG channels"
                )

            # Create the three axes for this row
            ax_left  = fig.add_subplot(gs[i, 0])
            ax_mid   = fig.add_subplot(gs[i, 1])
            ax_right = fig.add_subplot(gs[i, 2])
            axs_rows.append([ax_left, ax_mid, ax_right])

            # LEFT TASK TOPO
            im, _ = mne.viz.plot_topomap(
                data_left,
                self.raw.info,
                ch_type="eeg",
                sensors=True,
                cmap=cmap,
                vlim=vlim,
                mask=mask_bad,
                mask_params=mask_params_bad,
                show=False,
                axes=ax_left,
            )
            # overlay target channels (contralateral)
            im, _ = mne.viz.plot_topomap(
                data_left,
                self.raw.info,
                ch_type="eeg",
                sensors=True,
                cmap=cmap,
                vlim=vlim,
                mask=mask_target_left,
                mask_params=mask_params_target,
                show=False,
                axes=ax_left,
            )
            ax_left.set_title(f"{band_label} (Left)", loc="left", fontsize=11)
            

            # RIGHT TASK TOPO
            im, _ = mne.viz.plot_topomap(
                data_right,
                self.raw.info,
                ch_type="eeg",
                sensors=True,
                cmap=cmap,
                vlim=vlim,
                mask=mask_bad,
                mask_params=mask_params_bad,
                show=False,
                axes=ax_right,
            )
            # overlay target channels (contralateral)
            im, _ = mne.viz.plot_topomap(
                data_right,
                self.raw.info,
                ch_type="eeg",
                sensors=True,
                cmap=cmap,
                vlim=vlim,
                mask=mask_target_right,
                mask_params=mask_params_target,
                show=False,
                axes=ax_right,
            )
            ax_right.set_title(f"{band_label} (Right)", loc="left", fontsize=11)

            # COLORBAR + LEGEND (center axis)
            ax_mid.set_yticks([])
            ax_mid.set_xticks([])
            ax_mid.axis("off")

            # Create a vertical colorbar aligned to the middle axis
            clim = dict(kind="value", lims=[vmin, 0.0, vmax])
            divider = make_axes_locatable(ax_mid)
            cax = divider.append_axes(
                position="right",
                size="25%",
                pad=-0.1,
            )

            cbar = mne.viz.plot_brain_colorbar(
                cax,
                clim=clim,
                colormap=cmap,
                transparent=False,
                orientation="vertical",
                label=None,
            )
            cbar.set_ticks([vmax])

            # ERS / ERD labels to match old behavior
            ax_mid.text(
                -0.7,
                0.90,
                "ERS",
                va="bottom",
                ha="left",
                transform=ax_mid.transAxes,
                color="black",
                fontsize=9,
            )
            ax_mid.text(
                -0.7,
                0.0,
                "ERD",
                va="bottom",
                ha="left",
                transform=ax_mid.transAxes,
                color="black",
                fontsize=9,
            )

            # Only add the global legend text once, on the first row
            if i == 0:
                ax_mid.text(
                    -2.5,
                    1.35,
                    r"signed-r$^{2}$ Coefficients",
                    va="bottom",
                    ha="left",
                    transform=ax_mid.transAxes,
                    color="black",
                )
                # "Target (O)" in lime + black outline
                ax_mid.text(
                    -0.5,
                    1.21,
                    "Target (O)",
                    va="bottom",
                    ha="right",
                    transform=ax_mid.transAxes,
                    color="limegreen",
                    fontsize=12,
                )
                ax_mid.text(
                    -0.5,
                    1.20,
                    "Target (O)",
                    va="bottom",
                    ha="right",
                    transform=ax_mid.transAxes,
                    color="black",
                    fontsize=12,
                )
                # "Interpolated (X)" in dark grey + black outline
                ax_mid.text(
                    1.5,
                    1.21,
                    "Interpolated (X)",
                    va="bottom",
                    ha="left",
                    transform=ax_mid.transAxes,
                    color="grey",
                    fontsize=12,
                )
                ax_mid.text(
                    1.5,
                    1.20,
                    "Interpolated (X)",
                    va="bottom",
                    ha="left",
                    transform=ax_mid.transAxes,
                    color="black",
                    fontsize=12,
                )

        if suptitle:
            fig.suptitle(suptitle)

        fig.subplots_adjust(wspace=0)
        
        if self.save_path is not None:
            self._savefig(fig, "topoplot")
        
        return fig, axs_rows

    def _plot_frequency_bands(self, ax: Optional[Axes] = None, ylim: Optional[Tuple[float, float]] = None, fontsize: int = 12, fraction: float = 0.13) -> None:
        """Adds frequency band annotations to a plot"""
        # Frequency band annotations with their upper limit and label position
        bands = {
            r"$\delta$": [4,  2.5 ],
            r"$\theta$": [8,  6   ],
            r"$\alpha$": [13, 10.5],
            r"$\beta$" : [31, 22  ],
            r"$\gamma$": [50, 35  ],
        }
        for band, [freq, text_pos] in bands.items():
            # Draw vertical line for each band
            if ax:
                ax.axvline(x=freq, color="grey", linestyle="--", linewidth=1, alpha=0.5)
            else:
                plt.axvline(
                    x=freq, color="grey", linestyle="--", linewidth=1, alpha=0.5
                )
            # Place text label if ylim is provided
            if ylim:
                delta = (
                    abs(ylim[1] - ylim[0]) * fraction
                )  # Calculate vertical position for text
                if ax:
                    ax.text(
                        text_pos,
                        ylim[1] - delta,
                        band,
                        horizontalalignment="center",
                        fontsize=fontsize,
                    )
                else:
                    plt.text(
                        text_pos,
                        ylim[1] - delta,
                        band,
                        horizontalalignment="center",
                        fontsize=fontsize,
                    )

    def _plot_channel_psd(
        self,
        ch_name: str = "c3",
        signed_r2: bool = False,
        figsize: Tuple[float, float] = (5.0, 7.0),
    ) -> Figure:
        """Plot PSDs and frequency-wise R² for a single channel"""
        # Ensure PSDs and stats exist
        if not hasattr(self, "psds_dict"):
            raise RuntimeError("PSDs not generated. They should be created in __init__ via _generate_psds()")

        # Stack PSDs across blocks for each condition: (blocks, ch, bins)
        psds_left        = self._stack_psds_by_prefix("left")
        psds_right       = self._stack_psds_by_prefix("right")
        psds_left_rest   = self._stack_psds_by_prefix("left_rest")
        psds_right_rest  = self._stack_psds_by_prefix("right_rest")

        n_blocks, n_ch, n_bins = psds_left.shape

        # Resolve channel index
        ch_name_lower = ch_name.lower()

        ch_idx = None
        # Prefer ChannelSet, since everything else uses it
        try:
            idxs = self.ch_set.find_labels(ch_name_lower)
            if idxs:
                ch_idx = int(idxs[0])
        except Exception:
            ch_idx = None

        # Fallback: search in Raw's channel names
        if ch_idx is None:
            ch_names = np.array(self.raw.info["ch_names"], dtype=str)
            ch_lower = np.char.lower(ch_names)
            matches = np.where(ch_lower == ch_name_lower)[0]
            if matches.size == 0:
                raise ValueError(f"Channel '{ch_name}' not found in ChannelSet or Raw.")
            ch_idx = int(matches[0])

        # Human-readable label
        try:
            ch_label = self.ch_set.get_labels()[ch_idx]
        except Exception:
            ch_label = self.raw.info["ch_names"][ch_idx]

        # Frequency bins (use the same bins as the stats)
        results = self._ensure_stats_results(transf="r2")

        if "left_vs_left_rest" in results and "bins" in results["left_vs_left_rest"]:
            freqs = np.asarray(results["left_vs_left_rest"]["bins"])
        elif "right_vs_right_rest" in results and "bins" in results["right_vs_right_rest"]:
            freqs = np.asarray(results["right_vs_right_rest"]["bins"])
        else:
            freqs = np.asarray(getattr(self, "freqs", []), dtype=float)

        if freqs.shape[0] != n_bins:
            raise RuntimeError(
                f"Frequency bins length ({freqs.shape[0]}) does not match PSD bins ({n_bins})"
            )

        # PSDs per condition: shape (blocks, bins)
        x_left        = psds_left[:,       ch_idx, :]
        x_right       = psds_right[:,      ch_idx, :]
        x_left_rest   = psds_left_rest[:,  ch_idx, :]
        x_right_rest  = psds_right_rest[:, ch_idx, :]

        x_left_db        = self._psd_values_for_analysis(x_left)
        x_right_db       = self._psd_values_for_analysis(x_right)
        x_left_rest_db   = self._psd_values_for_analysis(x_left_rest)
        x_right_rest_db  = self._psd_values_for_analysis(x_right_rest)

        # Global min/max across all conditions for reasonable y-limits
        x_all_db = np.vstack([x_left_db, x_right_db, x_left_rest_db, x_right_rest_db])
        finite_y = x_all_db[np.isfinite(x_all_db)]
        if finite_y.size:
            min_val = float(np.min(finite_y))
            max_val = float(np.max(finite_y))
            span = max(max_val - min_val, 1.0)
            min_y = float(np.floor(min_val - span * 0.08))
            max_y = float(np.ceil(max_val + span * 0.08))
        else:
            min_y, max_y = -1.0, 1.0

        # Helper: mean and standard error across blocks
        def _mean_and_se(x: np.ndarray, axis: int = 0) -> Tuple[np.ndarray, np.ndarray]:
            x = np.asarray(x)
            mean = np.mean(x, axis=axis)
            std = np.std(x, axis=axis, ddof=0)
            se = std / np.sqrt(x.shape[axis])
            return mean, se

        # Figure and layout (2 rows: PSDs + R²)
        fig = plt.figure(figsize=figsize)
        gs = matplotlib.gridspec.GridSpec(2, 1, height_ratios=[3, 1])

        # ==========================
        # Upper panel: PSDs in dB
        # ==========================
        ax1 = fig.add_subplot(gs[0, 0])

        # Move Left
        mean, se = _mean_and_se(x_left_db, axis=0)
        ax1.fill_between(freqs, mean - se, mean + se, color="blue", alpha=0.5)
        ax1.plot(freqs, mean, color="blue", label="Move Left", alpha=1.0)

        # Rest Left
        mean, se = _mean_and_se(x_left_rest_db, axis=0)
        ax1.fill_between(freqs, mean - se, mean + se, color="dodgerblue", alpha=0.25)
        ax1.plot(freqs, mean, color="dodgerblue", label="Rest Left", alpha=0.35)

        # Move Right
        mean, se = _mean_and_se(x_right_db, axis=0)
        ax1.fill_between(freqs, mean - se, mean + se, color="red", alpha=0.5)
        ax1.plot(freqs, mean, color="red", label="Move Right", alpha=1.0)

        # Rest Right
        mean, se = _mean_and_se(x_right_rest_db, axis=0)
        ax1.fill_between(freqs, mean - se, mean + se, color="magenta", alpha=0.25)
        ax1.plot(freqs, mean, color="magenta", label="Rest Right", alpha=0.35)

        # Axis formatting
        ax1.set_xlim(freqs[0], freqs[-1])
        ax1.set_ylim(min_y, max_y)
        ax1.set_xscale("log")
        if self.psd_values_are_db:
            ax1.axhline(0.0, lw=1, ls="--", color="black", alpha=0.35)

        # Vertical grid at key frequencies
        for f in [2, 4, 6, 8, 10, 20, 30]:
            ax1.axvline(f, lw=1, ls=":", color="grey", alpha=0.4)

        # Horizontal grid every 5 dB between min_y and max_y
        def _multiples_of_n(start: int, end: int, n: int) -> List[int]:
            if start % n != 0:
                start_multiple = start + (n - start % n)
            else:
                start_multiple = start
            if end % n != 0:
                end_multiple = end - (end % n)
            else:
                end_multiple = end
            return list(range(start_multiple, end_multiple + 1, n))

        for y in _multiples_of_n(int(np.floor(min_y)), int(np.ceil(max_y)), 5):
            ax1.axhline(y, lw=1, ls=":", color="grey", alpha=0.4)

        legend = ax1.legend(loc="lower left")

        # Bold the "dominant" side if C3 or C4, like in the legacy script
        bold_font = FontProperties(weight="bold")
        for t in legend.get_texts():
            txt = t.get_text()
            if ch_label.lower() in self.chLeft:
                if txt in ("Move Right", "Rest Right"):
                    t.set_fontproperties(bold_font)
            elif ch_label.lower() in self.chRight:
                if txt in ("Move Left", "Rest Left"):
                    t.set_fontproperties(bold_font)

        ax1.set_xticks([])
        ax1.set_ylabel(self._psd_axis_label(), loc="top")
        ax1.text(
            0.0,
            1.05,
            f"Channel {ch_label}",
            weight="bold",
            va="top",
            ha="left",
            transform=ax1.transAxes,
            fontsize=12,
        )
        if self.psd_values_are_db:
            ax1.text(
                0.0,
                1.00,
                "1/f background removed",
                va="top",
                ha="left",
                transform=ax1.transAxes,
                fontsize=8,
            )
        
        self._plot_frequency_bands(ax=ax1, ylim=(min_y, max_y), fontsize=12, fraction=0.07)

        # ==========================
        # Lower panel: R² per bin
        # ==========================
        ax2 = fig.add_subplot(gs[1, 0])

        # Left vs Left Rest
        x_left_all = np.vstack([x_left_rest_db, x_left_db])  # (blocks_rest + blocks_task, bins)
        is_treatment_left = np.arange(x_left_all.shape[0]) < x_left_rest_db.shape[0]

        r2_left = self._calculate_r2(
            x=x_left_all,
            is_treatment=is_treatment_left,
            signed=signed_r2,
        )
        ax2.plot(freqs, r2_left, color="blue", label="Left")

        # Right vs Right Rest
        x_right_all = np.vstack([x_right_rest_db, x_right_db])
        is_treatment_right = np.arange(x_right_all.shape[0]) < x_right_rest_db.shape[0]

        r2_right = self._calculate_r2(
            x=x_right_all,
            is_treatment=is_treatment_right,
            signed=signed_r2,
        )
        ax2.plot(freqs, r2_right, color="red", label="Right")

        ax2.set_xlim(freqs[0], freqs[-1])
        ax2.set_ylim(0.0, 0.8)
        ax2.set_xscale("log")

        # Gridlines
        for f in [2, 4, 6, 8, 10, 20, 30]:
            ax2.axvline(f, lw=1, ls=":", color="grey", alpha=0.4)
        for y in [0.2, 0.4, 0.6, 0.8]:
            ax2.axhline(y, lw=1, ls=":", color="grey", alpha=0.4)

        # Legend & labels
        legend2 = ax2.legend(loc="upper left")
        bold_font2 = FontProperties(weight="bold")
        for t in legend2.get_texts():
            txt = t.get_text()
            if ch_label.lower() in self.chLeft and txt == "Right":
                t.set_fontproperties(bold_font2)
            elif ch_label.lower() in self.chRight and txt == "Left":
                t.set_fontproperties(bold_font2)

        ax2.set_xlabel("Frequency [Hz]", loc="right")
        ax2.set_ylabel(r"R$^{2}$")

        # Custom x/y ticks similar to the old script
        ax2.set_xticks([2, 4, 6, 8, 10, 20, 30, 40])
        ax2.set_xticklabels([2, 4, 6, 8, 10, 20, 30, 40])

        ax2.set_yticks([0.0, 0.2, 0.4, 0.6])
        ax2.set_yticklabels([0.0, 0.2, 0.4, 0.6])
        
        self._plot_frequency_bands(ax=ax2, ylim=None, fontsize=12)

        fig.subplots_adjust(hspace=0)
        
        if self.save_path is not None:
            self._savefig(fig, f"psd_{ch_name}")

        return fig, (ax1, ax2)

    def _plot_band_pvalues_bootstrap_separate(
        self,
        transf: str = "r2",
        results: Optional[Dict[str, Any]] = None,
        figsize: Optional[Tuple[float, float]] = None,
    ) -> Figure:
        """
        Plot -ln(p) bootstrap p-values per band in two separate figures,
        mimicking the 'Left only' and 'Right only' plots from motorimagery.py
        """
        # Fetch stats results
        results = results or self._ensure_stats_results(transf=transf)

        left_res = results.get("left_vs_left_rest")
        right_res = results.get("right_vs_right_rest")

        if left_res is None or right_res is None:
            raise RuntimeError(
                "Both 'left_vs_left_rest' and 'right_vs_right_rest' results "
                "are required to plot bootstrap p-values."
            )

        left_bands = left_res.get("band_results", {})
        right_bands = right_res.get("band_results", {})

        # Derive band ordering and labels
        band_list = left_res.get("bands_expanded") or []

        band_keys: List[str] = []
        if band_list:
            for low_f, high_f in band_list:
                band_keys.append(f"{low_f:g}-{high_f:g}Hz")
        else:
            band_keys = list(left_bands.keys())

        n_bands = len(band_keys)
        if n_bands == 0:
            raise RuntimeError("No band results found for bootstrap p-value plotting.")

        # Helper: extract -ln(p) and intervals from bootstrap results
        def _extract_neglog_p_bootstrap(
            band_results: Dict[str, Any],
            keys: List[str],
        ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
            """Return arrays of -ln(p_down), -ln(p), -ln(p_up) per band."""
            x_vals = []
            x_down_vals = []
            x_up_vals = []

            for key in keys:
                bres = (band_results or {}).get(key, {})
                boot = (bres or {}).get("bootstrap", {})

                p = boot.get("p", np.nan)
                p_int = boot.get("p_interval", (np.nan, np.nan))

                if p is None or np.isnan(p):
                    x_vals.append(np.nan)
                    x_down_vals.append(np.nan)
                    x_up_vals.append(np.nan)
                    continue

                p = float(p)
                p_down, p_up = p_int

                x_vals.append(self.neg_p(p))
                x_down_vals.append(self.neg_p(p_down))
                x_up_vals.append(self.neg_p(p_up))

            return (
                np.asarray(x_down_vals, dtype=float),
                np.asarray(x_vals, dtype=float),
                np.asarray(x_up_vals, dtype=float),
            )

        x_down_left, x_left, x_up_left = _extract_neglog_p_bootstrap(
            left_bands, band_keys
        )
        x_down_right, x_right, x_up_right = _extract_neglog_p_bootstrap(
            right_bands, band_keys
        )

        # Vertical positioning of bands (same trick as motorimagery.py)
        y_max = 0.5
        y_min = -0.5

        # Internal positions between y_max and y_min
        y = np.linspace(start=y_max, stop=y_min, num=n_bands + 2)[1:-1]
        deltay = 0.04

        if n_bands >= 2:
            dy = float(abs(y[-1] - y[-2]))
        else:
            dy = 0.1

        # X-range and alpha thresholds in -ln(p)
        alpha_05 = self.neg_p(0.05)

        # Upper bound fixed to 6 to match the legacy plot,
        # but ensure it contains the 0.05 threshold.
        x_max = max(6.0, alpha_05 + 0.5)

        # Figure size for a single side
        if figsize is None:
            figsize_single = (3.0, float(max(2, 2 * n_bands)))
        else:
            figsize_single = figsize

        # Inner helper to plot one side (Left-only or Right-only)
        def _plot_single_side(
            x_down: np.ndarray,
            x_val: np.ndarray,
            x_up: np.ndarray,
            side: str,
        ) -> Tuple[plt.Figure, plt.Axes]:
            """Create one figure for either left or right side."""
            fig, ax = plt.subplots(figsize=figsize_single)

            # Title text position and alignment
            if side.lower() == "left":
                ax.text(
                    0.0,
                    1.05,
                    "Open/Close Left",
                    ha="left",
                    va="top",
                    transform=ax.transAxes,
                    fontsize=12,
                    weight="bold",
                )
                xlabel_loc = "left"
            else:
                ax.text(
                    1.05,
                    1.05,
                    "Open/Close Right",
                    ha="right",
                    va="top",
                    transform=ax.transAxes,
                    fontsize=12,
                    weight="bold",
                )
                xlabel_loc = "right"

            # Confidence intervals as grey horizontal bands
            for i in range(n_bands):
                if np.isnan(x_val[i]):
                    continue
                ax.fill_betweenx(
                    [y[i] - deltay, y[i] + deltay],
                    x_down[i],
                    x_up[i],
                    color="gray",
                    alpha=0.3,
                )

            # Horizontal lines from 0 to point estimate + dot at the end
            ax.hlines(y, 0.0, x_val, colors="black", lw=3, alpha=1.0)
            ax.scatter(x_val, y, color="black", marker="o", s=50)

            # X-axis settings / mirroring
            ax.set_xlabel("-ln(p)", loc=xlabel_loc, fontsize=12)
            ax.set_xlim(ax.get_xlim()[::-1])  # mimic legacy pattern

            if side.lower() == "left":
                # Values grow to the left
                ax.set_xlim(right=0.0, left=x_max)
            else:
                # Values grow to the right
                ax.set_xlim(left=0.0, right=x_max)

            # Y-limits and silence the y-axis (no labels)
            ax.set_ylim(
                y_min + dy * 2.0 / 3.0,
                y_max - dy * 2.0 / 3.0,
            )

            ax.set_yticks([])
            ax.set_yticklabels([])

            # Vertical line at alpha=0.05
            ax.axvline(
                alpha_05,
                0.0,
                0.94,
                color="black",
                lw=1,
                ls=":",
                alpha=0.5,
                label="95% C.L.",
            )

            # Spine visibility
            ax.spines["top"].set_visible(False)
            if side.lower() == "left":
                ax.spines["left"].set_visible(False)
                alpha_text_x = 0.69
                alpha_ha = "right"
            else:
                ax.spines["right"].set_visible(False)
                alpha_text_x = 0.42
                alpha_ha = "left"

            # Small text 'α = 0.05' near the top, like motorimagery.py
            ax.text(
                alpha_text_x,
                0.98,
                r"$\alpha=0.05$",
                ha=alpha_ha,
                va="top",
                transform=ax.transAxes,
                fontsize=11,
            )

            plt.subplots_adjust(wspace=0.0)
            return fig, ax

        # Build the two separate figures
        fig_left, ax_left = _plot_single_side(
            x_down_left,
            x_left,
            x_up_left,
            side="left",
        )
        fig_right, ax_right = _plot_single_side(
            x_down_right,
            x_right,
            x_up_right,
            side="right",
        )
        
        if self.save_path is not None:
            self._savefig(fig_left, "pvalue_left")
            self._savefig(fig_right, "pvalue_right")

        return (fig_left, ax_left), (fig_right, ax_right)





    # 1) Paradigm sanity: timeline plot + counts
    def _summarize_annotations(self) -> Dict[str, int]:
        """
        Count occurrences of each annotation description
        Useful to sanity-check the paradigm parsing
        """
        annot = self.raw.annotations
        counts: Dict[str, int] = {}
        for d in annot.description:
            d = str(d)
            counts[d] = counts.get(d, 0) + 1
        return dict(sorted(counts.items(), key=lambda x: (-x[1], x[0])))

    def _plot_paradigm_timeline(
        self,
        include_numbered: bool = False,
        figsize: Tuple[float, float] = (14.0, 3.0),
    ) -> Figure:
        """Plot the task/rest/BAD_region annotations as a horizontal timeline"""
        annot = self.raw.annotations
        if annot is None or len(annot) == 0:
            raise RuntimeError("No annotations found on Raw.")

        # Labels to include (unless include_numbered=True)
        base = {"left", "left_rest", "right", "right_rest", "BAD_region"}

        # Decide which labels appear as lane bars (BAD_region handled separately)
        labels = []
        for desc in annot.description:
            s = str(desc)
            if include_numbered:
                labels.append(s)
            else:
                labels.append(s if s in base else None)

        # Y lanes (BAD_regions is a lane label for ticks, but not drawn as a bar)
        lanes = ["left", "left_rest", "right", "right_rest", "BAD_regions"]
        y_map = {name: i for i, name in enumerate(lanes)}

        fig, ax = plt.subplots(1, 1, figsize=figsize)

        # Compute max_time: end of data OR end of last annotation
        data_end = float(self.raw.times[-1]) if self.raw.n_times > 0 else 0.0
        annot_end = 0.0
        for onset, dur in zip(annot.onset, annot.duration):
            annot_end = max(annot_end, float(onset) + float(dur))
        max_time = max(data_end, annot_end)

        # First: draw the horizontal bars for non-BAD labels
        for onset, dur, desc in zip(annot.onset, annot.duration, labels):
            if desc is None:
                continue

            # BAD_region is handled as a vertical span overlay, not a lane bar
            if desc == "BAD_region":
                continue

            if desc not in y_map:
                continue

            y = y_map[desc]
            ax.broken_barh([(float(onset), float(dur))], (y - 0.35, 0.7))

        # Second: overlay BAD regions as translucent red spans across the full plot
        # Using axvspan ensures it covers all lanes and doesn't hide bars completely.
        for onset, dur, desc in zip(annot.onset, annot.duration, labels):
            if desc != "BAD_region":
                continue
            start = float(onset)
            end = float(onset) + float(dur)
            ax.axvspan(start, end, alpha=0.2, color="red", zorder=0)
            
        # Discarded epochs percentage:
        # MNE drops epochs that overlap annotations marked as BAD_* when creating Epochs.
        # Here we report how many candidate epochs existed vs how many survived epoching.
        try:
            events_from_annot, event_dict = mne.events_from_annotations(self.raw, verbose=False)
            epoch_keys = [
                k for k in event_dict.keys() if any(ch.isdigit() for ch in str(k))
            ]
            total_epochs = int(
                sum(
                    int(np.sum(events_from_annot[:, 2] == int(event_dict[k])))
                    for k in epoch_keys
                )
            )
        except Exception as exc:
            self._warn(f"Could not estimate discarded epochs: {exc}")
            total_epochs = 0

        kept_epochs = 0
        if hasattr(self, "epochs_dict") and isinstance(self.epochs_dict, dict):
            kept_epochs = int(sum(len(ep) for ep in self.epochs_dict.values()))

        discarded_epochs = max(total_epochs - kept_epochs, 0)
        discarded_pct = (100.0 * discarded_epochs / total_epochs) if total_epochs > 0 else float("nan")

        if total_epochs > 0:
            label = f"Discarded epochs: {discarded_pct:.0f}% ({discarded_epochs}/{total_epochs})"
        else:
            label = "Discarded epochs: N/A"

        ax.text(
            0.99,
            1.05,
            label,
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=12,
            bbox=dict(
                boxstyle="round,pad=0.3",
                facecolor="white",
                edgecolor="black",
                alpha=0.9,
            ),
        )
    
        # Axes formatting
        ax.set_xlim(0.0, max_time)
        ax.set_yticks(range(len(lanes)))
        ax.set_yticklabels(lanes, fontsize=12)
        ax.set_xlabel("Time [s]", fontsize=12)
        #ax.set_title("Motor imagery paradigm timeline", loc="left")
        ax.grid(True, axis="x", linestyle=":", alpha=0.4)

        # Color the BAD_regions y tick label red
        for tick in ax.get_yticklabels():
            if tick.get_text() == "BAD_regions":
                tick.set_color("red")

        fig.tight_layout()

        if self.save_path is not None:
            self._savefig(fig, "paradigm_timeline")

        return fig, ax



    # 2) "Effect + uncertainty" summary: forest plot across bands (left vs right)
    @staticmethod
    def _percentile_ci(x: np.ndarray, ci: float = 95.0) -> Tuple[float, float]:
        """Percentile CI for a bootstrap distribution"""
        x = np.asarray(x, dtype=float)
        if x.size == 0:
            return (np.nan, np.nan)
        lo = (100.0 - ci) / 2.0
        hi = 100.0 - lo
        return (float(np.percentile(x, lo)), float(np.percentile(x, hi)))

    def _plot_band_effect(
        self,
        transf: str = "r2",
        ci: float = 95.0,
        figsize: Tuple[float, float] = (8.0, 4.2),
    ) -> Figure:
        r"""
        Forest plot: observed $\Delta$ per band with bootstrap CI,
        for left_vs_left_rest and right_vs_right_rest
        """
        results = self._ensure_stats_results(transf=transf)

        left_res  = results.get("left_vs_left_rest", {})
        right_res = results.get("right_vs_right_rest", {})

        band_keys = list((left_res.get("band_results") or {}).keys())
        if not band_keys:
            band_keys = list((right_res.get("band_results") or {}).keys())
        if not band_keys:
            raise RuntimeError("No band_results found to plot.")

        band_keys = sorted(
            band_keys,
            key=lambda s: float(s.split("-")[0]) if "-" in s else 0.0,
        )

        def _extract(comp: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
            obs, lo, hi = [], [], []
            for bk in band_keys:
                b = (comp.get("band_results") or {}).get(bk, {})
                boot = b.get("bootstrap", {})
                o = float(boot.get("observed", np.nan))
                hist = np.asarray(boot.get("null_distribution", []), dtype=float)
                cilo, cihi = self._percentile_ci(hist, ci=ci)
                obs.append(o)
                lo.append(cilo)
                hi.append(cihi)
            return np.array(obs), np.array(lo), np.array(hi)

        obs_l, lo_l, hi_l = _extract(left_res)
        obs_r, lo_r, hi_r = _extract(right_res)
        y = np.arange(len(band_keys))

        fig, ax = plt.subplots(1, 1, figsize=figsize)

        finite_values = np.concatenate([obs_l, lo_l, hi_l, obs_r, lo_r, hi_r])
        finite_values = finite_values[np.isfinite(finite_values)]
        if finite_values.size:
            span = float(np.nanmax(finite_values) - np.nanmin(finite_values))
            pad = max(span * 0.15, 0.05)
            x_min = min(0.0, float(np.nanmin(finite_values)) - pad)
            x_max = max(0.0, float(np.nanmax(finite_values)) + pad)
        else:
            x_min, x_max = -1.0, 1.0

        ax.axvspan(x_min, 0.0, color="#f7c7c7", alpha=0.28, zorder=0)
        #ax.axvspan(0.0, x_max, color="#c8e6c9", alpha=0.28, zorder=0, label=r"Good: $\Delta > 0$")

        ax.errorbar(
            obs_l,
            y - 0.12,
            xerr=[obs_l - lo_l, hi_l - obs_l],
            fmt="o",
            capsize=3,
            color="tab:blue",
            ecolor="tab:blue",
            label="Left hand (vs rest)",
            zorder=3,
        )
        ax.errorbar(
            obs_r,
            y + 0.12,
            xerr=[obs_r - lo_r, hi_r - obs_r],
            fmt="s",
            capsize=3,
            color="tab:red",
            ecolor="tab:red",
            label="Right hand (vs rest)",
            zorder=3,
        )

        ax.axvline(0.0, linestyle="--", linewidth=1, color="0.25", label=r"$\Delta = 0$", zorder=2)
        ax.set_xlim(x_min, x_max)
        ax.set_yticks(y)
        ax.set_yticklabels([bk.replace("Hz", " Hz") for bk in band_keys])
        ax.invert_yaxis()
        ax.set_xlabel(r"Observed $\Delta$", loc="right")
        ax.set_title("Band-wise bootstrap CI and one-sided target region", loc="left", fontsize=10)
        ax.text(
            0.995,
            1.02,
            r"One-sided target: $\Delta > 0$",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=9,
            weight="bold",
        )
        ax.grid(True, axis="x", linestyle=":", alpha=0.35)
        ax.legend(frameon=False, fontsize=8, ncols=1)

        fig.subplots_adjust(bottom=0.24)

        "Features: band-averaged PSD from contra or ipsi motor ROI channels\nValues above 0.50 indicate task/rest separation beyond chance",


        fig.text(0.01,0.07, 
                 r"$\Delta$ = $\sum$ ipsilateral signed $r^2$ $-$ $\sum$ contralateral signed $r^2$",
                 ha="left", va="bottom", fontsize=9)
        fig.text(0.01,0.04, 
                 "Computed from band-averaged PSD (tast vs rest) using target channels",
                 ha="left", va="bottom", fontsize=9)
        fig.text(0.01,0.01, 
                 r"Bad/opposite effect: $\Delta \leq 0$",
                 ha="left", va="bottom", fontsize=9)

        if self.save_path is not None:
            self._savefig(fig, "band_effect")

        return fig, ax



    # 3) Within-trial dynamics: bandpower over time (ERD/ERS-ish), with CI
    def _trial_keys_for_prefix(self, prefix: str) -> List[str]:
        """
        Return ordered batch keys like: left_1, left_2, ...
        """
        prefix = prefix.lower().rstrip("_")
        pat = re.compile(rf"^{re.escape(prefix)}_(\d+)$", re.IGNORECASE)

        keys = []
        for k in self.epochs_dict.keys():
            m = pat.match(k.strip())
            if m:
                keys.append((int(m.group(1)), k))
        keys.sort(key=lambda x: x[0])
        return [k for _, k in keys]

    def _resolve_picks_from_names(self, names_lower: List[str]) -> List[int]:
        """
        Resolve channel picks (indices) from a list of lowercase channel names,
        in the Epochs' channel order
        """
        if not hasattr(self, "raw"):
            raise RuntimeError("Missing Raw.")
        # Use raw channel order (Epochs inherit it)
        chs = [c.lower() for c in self.raw.info["ch_names"]]
        name_set = set([n.lower() for n in names_lower])
        return [i for i, c in enumerate(chs) if c in name_set]

    def _export_psd_csv(self) -> None:
        """
        Export the mean PSD (averaged across all trial batches) per channel per
        frequency bin for motor imagery channels of interest.

        Output file: <save_path>/csv/psd_mean_per_channel_per_bin.csv
        Layout:
            rows    — one per motor imagery channel found in the recording
            columns — 'channel', then one column per frequency bin labelled
                      as the bin centre in Hz (e.g. '1.0', '2.0', ...)
            values  — raw Welch power (same units as MNE compute_psd output,
                      typically V²/Hz)
        """
        import csv as _csv

        if not hasattr(self, "psds_dict") or not self.psds_dict:
            print("[CSV] No PSD data available. Skipping PSD CSV export.")
            return

        # ── Channel mapping ────────────────────────────────────────────────
        # ch_set.get_labels() is ordered identically to the second (channel)
        # dimension of every psds_dict value array.
        all_labels    = [lbl.lower() for lbl in self.ch_set.get_labels()]
        mi_labels_set = {ch.lower() for ch in self.ch_motor_imagery}

        mi_indices = [i for i, lbl in enumerate(all_labels) if lbl in mi_labels_set]
        mi_names   = [all_labels[i] for i in mi_indices]

        if not mi_indices:
            print("[CSV] No motor imagery channels found in channel set. "
                  "Skipping PSD CSV export.")
            return

        # ── Average PSD across all batches ─────────────────────────────────
        # Each psds_dict value has shape (n_ch, n_bins)
        all_psds = np.stack(list(self.psds_dict.values()), axis=0)  # (n_batches, ch, bins)
        mean_psd  = np.mean(all_psds, axis=0)                        # (ch, bins)

        n_bins = mean_psd.shape[1]
        freqs = np.asarray(getattr(self, "freqs", []), dtype=float)
        if freqs.size != n_bins:
            raise RuntimeError(
                f"PSD frequency vector length ({freqs.size}) does not match PSD bins ({n_bins})"
            )

        mi_psd = mean_psd[mi_indices, :]  # (n_mi_ch, bins)

        # ── Write CSV ──────────────────────────────────────────────────────
        fname = self._csv_path("psd_mean_per_channel_per_bin.csv")
        if fname is None:
            return
        with open(fname, "w", newline="") as fh:
            writer = _csv.writer(fh)
            writer.writerow(["channel"] + [f"{f:.4g}" for f in freqs])
            for ch_name, psd_row in zip(mi_names, mi_psd):
                writer.writerow([ch_name] + [f"{v:.6e}" for v in psd_row])

        self._track_export(fname)
        print(f"[CSV] PSD mean per channel per bin saved → {fname}")

    def _export_psd_long_csv(self) -> None:
        """Export condition-level PSD in tidy long format"""
        if not hasattr(self, "psds_dict") or not self.psds_dict:
            return
        labels = [lbl.lower() for lbl in self.ch_set.get_labels()]
        mi_set = {ch.lower() for ch in self.ch_motor_imagery}
        mi_indices = [i for i, lbl in enumerate(labels) if lbl in mi_set]
        freqs = np.asarray(getattr(self, "freqs", []), dtype=float)
        rows: List[Dict[str, Any]] = []
        for key, psd in self.psds_dict.items():
            condition = self._condition_from_epoch_key(key)
            psd = np.asarray(psd, dtype=float)
            psd_values = self._psd_values_for_analysis(psd)
            for ch_idx in mi_indices:
                for f_idx, freq in enumerate(freqs):
                    value = float(psd_values[ch_idx, f_idx])
                    rows.append({
                        "batch": key,
                        "condition": condition,
                        "channel": labels[ch_idx],
                        "frequency_hz": float(freq),
                        "psd_power": "" if self.psd_values_are_db else float(psd[ch_idx, f_idx]),
                        "psd_db": value,
                        "psd_value": value,
                        "psd_units": self.psd_value_units,
                        "psd_processing": self.psd_processing,
                    })
        self._write_csv_rows("psd_long_format.csv", rows)

    def _export_r2_csv(self) -> None:
        """
        Export signed r² per frequency band per channel for motor imagery
        channels of interest.

        Output file: <save_path>/csv/signed_r2_per_band_per_channel.csv
        Layout:
            rows    — one per (comparison × frequency band) combination,
                      e.g. ('left_vs_left_rest', '4-7Hz')
            columns — 'comparison', 'band', then one column per motor
                      imagery channel found in the recording
            values  — signed r² (negative = power suppression during task,
                      positive = power increase during task)
        """
        import csv as _csv

        if not hasattr(self, "stats_results") or not self.stats_results:
            print("[CSV] No stats results available. Skipping r² CSV export.")
            return

        # ── Channel mapping ────────────────────────────────────────────────
        all_labels    = [lbl.lower() for lbl in self.ch_set.get_labels()]
        mi_labels_set = {ch.lower() for ch in self.ch_motor_imagery}

        mi_indices = [i for i, lbl in enumerate(all_labels) if lbl in mi_labels_set]
        mi_names   = [all_labels[i] for i in mi_indices]

        if not mi_indices:
            print("[CSV] No motor imagery channels found. Skipping r² CSV export.")
            return

        # ── Write CSV ──────────────────────────────────────────────────────
        fname = self._csv_path("signed_r2_per_band_per_channel.csv")
        if fname is None:
            return
        with open(fname, "w", newline="") as fh:
            writer = _csv.writer(fh)
            writer.writerow(["comparison", "band"] + mi_names)
            for comp_name, comp_res in self.stats_results.items():
                band_results = comp_res.get("band_results", {})
                for band_key, bres in band_results.items():
                    effect     = np.asarray(bres["effect_per_channel"])  # (n_ch,)
                    mi_effects = effect[mi_indices]
                    writer.writerow(
                        [comp_name, band_key] + [f"{v:.6f}" for v in mi_effects]
                    )

        self._track_export(fname)
        print(f"[CSV] Signed r² per band per channel saved → {fname}")

    def _export_bootstrap_csv(self, ci: float = 95.0) -> None:
        """
        Export bootstrap statistics per frequency band and comparison to a CSV file.

        The bootstrap resamples trials with replacement *within* each group, so
        the resulting distribution characterises the sampling variability of the
        observed test statistic (difference of sums of signed r²,
        contralateral minus ipsilateral).  The confidence interval reported
        here is therefore a bootstrap CI on that effect estimate.

        Output file: <save_path>/csv/bootstrap_stats_per_band.csv
        Columns:
            comparison   — e.g. 'left_vs_left_rest'
            band         — e.g. '4-7Hz'
            observed     — observed test statistic (ipsi − contra Σ signed-r²)
            boot_mean    — mean of the bootstrap distribution
            ci_low       — lower percentile of the bootstrap distribution
            ci_high      — upper percentile of the bootstrap distribution
            ci_level     — confidence level used (e.g. 95.0)
            p_value      — one-sided bootstrap p-value (H₀: effect ≤ 0)
            p_ci_low     — lower bound of the Monte-Carlo p-value interval
            p_ci_high    — upper bound of the Monte-Carlo p-value interval
            n_simulations — number of bootstrap resamples
        """
        import csv as _csv

        if not hasattr(self, "stats_results") or not self.stats_results:
            print("[CSV] No stats results available. Skipping bootstrap CSV export.")
            return

        alpha_lo = (100.0 - ci) / 2.0
        alpha_hi = 100.0 - alpha_lo

        fname = self._csv_path("bootstrap_stats_per_band.csv")
        if fname is None:
            return
        with open(fname, "w", newline="") as fh:
            writer = _csv.writer(fh)
            writer.writerow([
                "comparison", "band",
                "observed",
                "boot_mean", "ci_low", "ci_high", "ci_level",
                "p_value", "p_ci_low", "p_ci_high",
                "n_simulations",
            ])
            for comp_name, comp_res in self.stats_results.items():
                band_results = comp_res.get("band_results", {})
                for band_key, bres in band_results.items():
                    boot = bres.get("bootstrap", {})
                    dist = np.asarray(boot.get("null_distribution", []))
                    observed  = boot.get("observed",      float("nan"))
                    p_val     = boot.get("p",             float("nan"))
                    p_ci      = boot.get("p_interval",    (float("nan"), float("nan")))
                    n_sim     = boot.get("n_simulations", len(dist))

                    if dist.size > 0:
                        boot_mean = float(np.mean(dist))
                        ci_low    = float(np.percentile(dist, alpha_lo))
                        ci_high   = float(np.percentile(dist, alpha_hi))
                    else:
                        boot_mean = ci_low = ci_high = float("nan")

                    writer.writerow([
                        comp_name, band_key,
                        f"{observed:.6f}",
                        f"{boot_mean:.6f}", f"{ci_low:.6f}", f"{ci_high:.6f}", f"{ci:.1f}",
                        f"{p_val:.6f}", f"{p_ci[0]:.6f}", f"{p_ci[1]:.6f}",
                        n_sim,
                    ])

        self._track_export(fname)
        print(f"[CSV] Bootstrap stats per band saved → {fname}")

    def _bandpower_envelope(
        self,
        batch_key: str,
        band: Tuple[float, float],
        picks: Optional[List[int]] = None,
    ) -> np.ndarray:
        """Compute mean band power per epoch from precomputed Welch PSDs"""
        if not hasattr(self, "psds_epochs_dict"):
            raise RuntimeError(
                "Per-epoch PSD cache missing. Run _generate_psds() after enabling psds_epochs_dict."
            )

        if batch_key not in self.psds_epochs_dict:
            raise KeyError(f"No per-epoch PSDs for batch '{batch_key}'")

        psd = np.asarray(self.psds_epochs_dict[batch_key], dtype=float)  # (epochs, ch, bins)
        if psd.ndim != 3:
            raise RuntimeError(
                f"Expected per-epoch PSDs with shape (epochs, ch, bins), got {psd.shape} for '{batch_key}'"
            )

        # Channel selection
        if picks is not None:
            psd = psd[:, picks, :]

        n_bins = psd.shape[2]

        freqs = np.asarray(getattr(self, "freqs", []), dtype=float)
        if freqs.size != n_bins:
            raise RuntimeError(
                f"PSD frequency vector length ({freqs.size}) does not match PSD bins ({n_bins})"
            )

        low_f, high_f = float(band[0]), float(band[1])
        band_idx = self._band_indices(freqs, low_f, high_f, is_last=True)
        if band_idx.size == 0:
            raise ValueError(f"No frequency bins fall inside band [{low_f}, {high_f}] Hz")

        start = int(band_idx[0])
        stop = int(band_idx[-1]) + 1  # inclusive

        # Mean across channels and frequency bins => bandpower per epoch
        bp = np.mean(psd[:, :, start:stop], axis=(1, 2))
        return bp


    def _compute_within_trial_erd_ers(
        self,
        task_prefix: str,
        rest_prefix: str,
        band: Tuple[float, float],
        roi: str = "contra",
        ci: float = 95.0,
    ) -> Dict[str, Any]:
        """Compute ERD/ERS percent change over within-trial segments"""
        task_prefix_l = task_prefix.lower().rstrip("_")
        contra_set = set([c.lower() for c in (self.chRight if task_prefix_l.startswith("left") else self.chLeft)])
        ipsi_set = set([c.lower() for c in (self.chLeft if task_prefix_l.startswith("left") else self.chRight)])
        both_set = set([c.lower() for c in self.ch_motor_imagery])

        if roi == "contra":
            picks = self._resolve_picks_from_names(list(contra_set))
        elif roi == "ipsi":
            picks = self._resolve_picks_from_names(list(ipsi_set))
        else:
            picks = self._resolve_picks_from_names(list(both_set))

        task_keys = self._trial_keys_for_prefix(task_prefix)
        rest_keys = self._trial_keys_for_prefix(rest_prefix)
        n_trials = min(len(task_keys), len(rest_keys))
        if n_trials == 0:
            raise RuntimeError(f"No trials found for {task_prefix=} or {rest_prefix=}")

        delta = (self.duration_task - self.skip) / float(self.nEpochs)
        t = np.arange(self.nEpochs) * delta + (delta / 2.0)
        task_mat = []
        rest_mat = []

        for i in range(n_trials):
            bp_task = self._bandpower_envelope(task_keys[i], band=band, picks=picks)[: self.nEpochs]
            bp_rest = self._bandpower_envelope(rest_keys[i], band=band, picks=picks)[: self.nEpochs]
            if bp_task.size == self.nEpochs and bp_rest.size == self.nEpochs:
                task_mat.append(bp_task)
                rest_mat.append(bp_rest)

        task_mat = np.asarray(task_mat, dtype=float)
        rest_mat = np.asarray(rest_mat, dtype=float)
        if task_mat.size == 0:
            raise RuntimeError("No usable trials after filtering/truncation")

        if self.psd_values_are_db:
            rel_trials = task_mat - rest_mat
            value_unit = "dB"
            value_label = "Task minus matched rest residual [dB]"
        else:
            rel_trials = 100.0 * (task_mat - rest_mat) / (rest_mat + 1e-12)
            value_unit = "%"
            value_label = "ERD/ERS vs rest [%]"
        rel = np.mean(rel_trials, axis=0)
        lo, hi = np.percentile(rel_trials, [(100 - ci) / 2.0, 100 - (100 - ci) / 2.0], axis=0)

        return {
            "time_sec": t,
            "mean_pct": rel,
            "ci_low": lo,
            "ci_high": hi,
            "n_trials": int(rel_trials.shape[0]),
            "roi": roi,
            "band": band,
            "task": task_prefix,
            "rest": rest_prefix,
            "value_unit": value_unit,
            "value_label": value_label,
            "psd_processing": self.psd_processing,
        }

    def _plot_within_trial_bandpower(
        self,
        task_prefix: str,
        rest_prefix: str,
        band: Tuple[float, float],
        roi: str = "contra",
        ci: float = 95.0,
        figsize: Tuple[float, float] = (10.0, 4.0),
    ) -> Figure:
        """Plot ERD/ERS-style relative bandpower across within-trial segments"""
        curve = self._compute_within_trial_erd_ers(
            task_prefix=task_prefix,
            rest_prefix=rest_prefix,
            band=band,
            roi=roi,
            ci=ci,
        )
        t = curve["time_sec"]
        rel = curve["mean_pct"]
        lo = curve["ci_low"]
        hi = curve["ci_high"]

        fig, ax = plt.subplots(1, 1, figsize=figsize)
        ax.plot(t, rel, marker="o")
        ax.fill_between(t, lo, hi, alpha=0.2)
        ax.axhline(0.0, linestyle="--", linewidth=1)

        ax.set_xlabel("Time within trial [s]")
        ax.set_ylabel(curve.get("value_label", "ERD/ERS vs rest [%]"))
        ax.set_title(
            f"{task_prefix} vs {rest_prefix} | {band[0]:g}-{band[1]:g} Hz | ROI={roi} | n={curve['n_trials']}",
            loc="left",
        )
        ax.grid(True, linestyle=":", alpha=0.4)
        fig.tight_layout()

        if self.save_path is not None:
            basename = f"erd_ers_{task_prefix}_{band[0]:g}-{band[1]:g}Hz_{roi}".replace(" ", "")
            self._savefig(fig, basename)

        return fig, ax




    # 4) Laterality index over time (the thing reviewers love)
    def _plot_within_trial_bandpower_overlay_rois(
        self,
        task_prefix: str,
        rest_prefix: str,
        band: Tuple[float, float],
        ci: float = 95.0,
        figsize: Tuple[float, float] = (10.0, 4.6),
    ) -> Figure:
        """Plot contra and ipsi ERD/ERS curves on the same task-vs-rest axis"""
        curves = {
            roi: self._compute_within_trial_erd_ers(
                task_prefix=task_prefix,
                rest_prefix=rest_prefix,
                band=band,
                roi=roi,
                ci=ci,
            )
            for roi in ("contra", "ipsi")
        }
        value_label = next(iter(curves.values())).get("value_label", "Bandpower change vs matched rest [%]")

        fig, ax = plt.subplots(1, 1, figsize=figsize)
        style = {
            "contra": {"color": "tab:blue", "label": "Contralateral ROI"},
            "ipsi": {"color": "tab:orange", "label": "Ipsilateral ROI"},
        }

        for roi, curve in curves.items():
            t = curve["time_sec"]
            rel = curve["mean_pct"]
            lo = curve["ci_low"]
            hi = curve["ci_high"]
            color = style[roi]["color"]
            ax.plot(t, rel, marker="o", color=color, label=f"{style[roi]['label']} mean")
            ax.fill_between(t, lo, hi, color=color, alpha=0.16, label=f"{style[roi]['label']} {ci:g}% CI")

        ax.axhline(0.0, linestyle="--", linewidth=1, color="0.3")
        ax.set_xlabel("Time within trial [s]")
        ax.set_ylabel(value_label)
        ax.set_title(
            f"Task-vs-rest ERD/ERS overlay | {task_prefix} | {band[0]:g}-{band[1]:g} Hz",
            loc="left",
        )
        ax.text(
            0.01,
            0.04,
            "Negative values indicate ERD/suppression relative to rest; positive values indicate ERS/increase\nContra and ipsi are shown together to expose hemispheric asymmetry",
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=8,
        )
        ax.grid(True, linestyle=":", alpha=0.4)
        ax.legend(frameon=False, fontsize=8, ncols=2, loc="upper right")
        fig.tight_layout()

        if self.save_path is not None:
            basename = f"erd_ers_{task_prefix}_{band[0]:g}-{band[1]:g}Hz_contra_ipsi".replace(" ", "")
            self._savefig(fig, basename)

        return fig, ax



    def _plot_within_trial_laterality_index(
        self,
        task_prefix: str,
        rest_prefix: str,
        band: Tuple[float, float],
        ci: float = 95.0,
        figsize: Tuple[float, float] = (10.0, 4.0),
    ) -> Figure:
        """
        Laterality index (contra - ipsi) / (contra + ipsi) within trial
        Computed on bandpower envelopes per segment
        """
        task_prefix_l = task_prefix.lower().rstrip("_")
        contra_names = self.chRight if task_prefix_l.startswith("left") else self.chLeft
        ipsi_names = self.chLeft if task_prefix_l.startswith("left") else self.chRight

        contra_picks = self._resolve_picks_from_names([c.lower() for c in contra_names])
        ipsi_picks = self._resolve_picks_from_names([c.lower() for c in ipsi_names])

        task_keys = self._trial_keys_for_prefix(task_prefix)
        rest_keys = self._trial_keys_for_prefix(rest_prefix)
        n_trials = min(len(task_keys), len(rest_keys))
        if n_trials == 0:
            raise RuntimeError(f"No trials found for {task_prefix=} or {rest_prefix=}.")

        delta = (self.duration_task - self.skip) / float(self.nEpochs)
        t = np.arange(self.nEpochs) * delta + (delta / 2.0)

        li_trials = []

        for i in range(n_trials):
            bp_contra = self._bandpower_envelope(task_keys[i], band=band, picks=contra_picks)[: self.nEpochs]
            bp_ipsi = self._bandpower_envelope(rest_keys[i], band=band, picks=ipsi_picks)[: self.nEpochs]

            if bp_contra.size != self.nEpochs or bp_ipsi.size != self.nEpochs:
                continue

            li = (bp_contra - bp_ipsi) / (bp_contra + bp_ipsi + 1e-12)
            li_trials.append(li)

        li_trials = np.asarray(li_trials, dtype=float)
        if li_trials.size == 0:
            raise RuntimeError("No usable trials to compute laterality index.")

        li_mean = np.mean(li_trials, axis=0)
        lo, hi = np.percentile(li_trials, [(100 - ci) / 2.0, 100 - (100 - ci) / 2.0], axis=0)

        fig, ax = plt.subplots(1, 1, figsize=figsize)
        ax.plot(t, li_mean, marker="o")
        ax.fill_between(t, lo, hi, alpha=0.2)
        ax.axhline(0.0, linestyle="--", linewidth=1)

        ax.set_xlabel("Time within trial [s]")
        ax.set_ylabel("Laterality index")
        ax.set_title(f"Laterality index | {task_prefix} | {band[0]:g}-{band[1]:g} Hz", loc="left")
        ax.grid(True, linestyle=":", alpha=0.4)
        fig.tight_layout()

        if self.save_path is not None:
            basename = f"laterality_{task_prefix}_{band[0]:g}-{band[1]:g}Hz".replace(" ", "")
            self._savefig(fig, basename, formats=("png",))

        return fig, ax
