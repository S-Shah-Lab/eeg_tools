"""
EEGRawImporter

Standalone utility for importing BCI2000 .dat EEG recordings into MNE RawArray objects

The importer assigns an inferred montage, can optionally append BCI2000 states as
stim channels, and can save validation plots for quick import checks
"""

from __future__ import annotations

from pathlib import Path
from time import perf_counter
from typing import Any

import mne
import numpy as np
from BCI2000Tools.Electrodes import ChannelSet
from BCI2000Tools.FileReader import bcistream

import helper.eeg_dict as eeg_dict


SUPPORTED_EXTENSIONS = {".dat"}
DSI_AUX_CHANNELS = {"X1", "X2", "X3", "TRG"}
EGI_FLAT_STD_THRESHOLD = 0.01
EGI_64_MIN_FLAT_CHANNELS = 20
MICROVOLTS_TO_VOLTS = 1e-6
VOLTS_TO_MICROVOLTS = 1e6
DEFAULT_PLOT_FORMAT = "png"
SUPPORTED_PLOT_FORMATS = {"png", "pdf", "svg"}


class EEGRawImporter:
    """Import EEG recordings from BCI2000 .dat files into MNE Raw objects"""

    def __init__(
        self,
        path_to_file: str | Path,
        helper_dir: str | Path | None = None,
        keep_stim: bool = False,
        verbose: bool = False,
    ) -> None:
        """Initialize the importer and immediately load the input file"""
        self.path_to_file = Path(path_to_file).expanduser()
        self.helper_dir = Path(helper_dir).expanduser() if helper_dir else None
        self.keep_stim = keep_stim
        self.verbose = verbose

        self.stream: dict[str, Any] = {}
        self.montage: dict[str, Any] = {}
        self.ch_set: ChannelSet | None = None
        self.raw: mne.io.RawArray | None = None

        self._import_file()

    def _log(self, message: str, *, verbose_only: bool = False) -> None:
        """Print importer status messages"""
        if self.verbose or not verbose_only:
            print(f"[RawImporter] {message}")

    def _read_bci2000_stream(self) -> dict[str, Any]:
        """Decode the BCI2000 stream and return signal, states, and metadata"""
        self._log("Decoding BCI2000 stream", verbose_only=True)
        start_time = perf_counter()

        stream = bcistream(str(self.path_to_file))
        signal, states = stream.decode()

        signal = np.asarray(signal, dtype=float)
        fs = float(stream.samplingrate())
        ch_names = list(stream.params["ChannelNames"])
        n_channels, n_samples = signal.shape
        duration_sec = n_samples / fs if fs else 0.0

        storage_time = stream.params.get("StorageTime", "")
        date_test = str(storage_time).split("T", 1)[0] if storage_time else "unknown"
        elapsed = perf_counter() - start_time

        self._log(
            f"Decoded {n_channels} channels x {n_samples} samples "
            f"at {fs:g} Hz ({duration_sec:.1f} s) in {elapsed:.2f} s"
        )

        return {
            "signal": signal,
            "states": states,
            "fs": fs,
            "ch_names": ch_names,
            "n_channels": n_channels,
            "n_samples": n_samples,
            "duration_sec": duration_sec,
            "date_test": date_test,
        }

    def _resolve_path(self, file_name: str) -> str:
        """Resolve a helper-file path relative to helper_dir or the current working directory"""
        search_roots: list[Path] = []
        if self.helper_dir is not None:
            search_roots.append(self.helper_dir)
        search_roots.append(Path.cwd())

        tried_paths = [root / file_name for root in search_roots]
        for path in tried_paths:
            if path.is_file():
                self._log(f"Using helper file: {path}", verbose_only=True)
                return str(path)

        tried = "\n  - ".join(str(path) for path in tried_paths)
        raise FileNotFoundError(f"Could not locate helper file '{file_name}'. Tried:\n  - {tried}")

    def _resolve_montage(self) -> dict[str, Any]:
        """Infer montage type and associated location file from channel count and names"""
        n_channels = int(self.stream["n_channels"])
        ch_names = list(self.stream["ch_names"])
        signal = np.asarray(self.stream["signal"], dtype=float)

        if n_channels in (21, 24):
            keep_idx = [idx for idx, ch_name in enumerate(ch_names) if ch_name not in DSI_AUX_CHANNELS]
            dropped = len(ch_names) - len(keep_idx)
            if dropped:
                self._log(f"Dropping {dropped} DSI aux or trigger channel(s)", verbose_only=True)

            return {
                "montage_type": "DSI_24",
                "ch_info": self._resolve_path("DSI24_location.txt"),
                "signal": signal[keep_idx],
                "ch_names": list(np.asarray(ch_names)[keep_idx]),
            }

        if n_channels == 32:
            return {
                "montage_type": "GTEC_32",
                "ch_info": self._resolve_path("GTEC32_location.txt"),
                "signal": signal,
                "ch_names": ch_names,
            }

        if n_channels == 128:
            stds = np.std(signal, axis=1)
            n_flat_channels = int(np.sum(stds < EGI_FLAT_STD_THRESHOLD))
            is_egi_64 = n_flat_channels >= EGI_64_MIN_FLAT_CHANNELS

            self._log(
                f"Detected {n_flat_channels} nearly-flat channel(s) "
                f"below std<{EGI_FLAT_STD_THRESHOLD}",
                verbose_only=True,
            )

            if is_egi_64:
                keep_idx = np.asarray(eeg_dict.id_ch_64_keep, dtype=int)
                return {
                    "montage_type": "EGI_64",
                    "ch_info": self._resolve_path("EGI64_location.txt"),
                    "signal": signal[keep_idx],
                    "ch_names": list(np.asarray(ch_names)[keep_idx]),
                }

            return {
                "montage_type": "EGI_128",
                "ch_info": self._resolve_path("EGI128_location.txt"),
                "signal": signal,
                "ch_names": ch_names,
            }

        raise ValueError(
            f"Unknown montage with {n_channels} channels. "
            "Supported layouts are DSI 24, GTEC 32, EGI 64, and EGI 128"
        )

    @staticmethod
    def _make_raw_with_montage(
        signal: np.ndarray,
        fs: float,
        ch_names: list[str],
        montage_type: str,
        conv_dict: dict[str, str] | None = None,
    ) -> mne.io.RawArray:
        """Build an MNE RawArray with an appropriate montage"""
        if signal.shape[0] != len(ch_names):
            raise ValueError(f"Signal has {signal.shape[0]} rows but {len(ch_names)} channel labels were provided")
        if fs <= 0:
            raise ValueError("Sampling rate must be positive")

        info = mne.create_info(ch_names=ch_names, sfreq=fs, ch_types="eeg")
        raw = mne.io.RawArray(signal, info, verbose=False)

        if montage_type in {"DSI_24", "GTEC_32"}:
            template = mne.channels.make_standard_montage("standard_1020")
        elif montage_type in {"EGI_64", "EGI_128"}:
            template = mne.channels.make_standard_montage("GSN-HydroCel-129")
        else:
            raise ValueError(f"Unknown montage_type '{montage_type}'")

        template_index = {name.lower(): idx for idx, name in enumerate(template.ch_names)}
        selected_indices: list[int] = []

        for ch_name in ch_names:
            lookup_name = ch_name.lower()
            if montage_type in {"EGI_64", "EGI_128"}:
                if conv_dict is None:
                    raise ValueError("conv_dict must be provided for EGI montages")
                mapped_name = conv_dict.get(lookup_name)
                if mapped_name is None:
                    raise KeyError(f"Missing EGI channel mapping for '{ch_name}'")
                lookup_name = mapped_name.lower()

            if lookup_name not in template_index:
                raise ValueError(f"Channel '{ch_name}' mapped to '{lookup_name}' was not found in the MNE montage")
            selected_indices.append(template_index[lookup_name])

        template.ch_names = ch_names
        template.dig = template.dig[:3] + [template.dig[idx + 3] for idx in selected_indices]
        raw.set_montage(template)
        return raw

    def _add_stim_to_raw(self) -> None:
        """Append BCI2000 state vectors as MNE stim channels onto self.raw"""
        raw = self._require_raw()
        states = self.stream["states"]
        state_names = list(states.keys())

        if not state_names:
            self._log("No BCI2000 states found, skipping stim channels")
            return

        stim_rows: list[np.ndarray] = []
        for name in state_names:
            state = np.asarray(states[name])
            flattened = np.squeeze(state)
            if flattened.ndim > 1:
                flattened = flattened.reshape(-1)[-raw.n_times:]
            if flattened.shape[-1] != raw.n_times:
                raise ValueError(f"State channel '{name}' has {flattened.shape[-1]} samples, expected {raw.n_times}")
            stim_rows.append(flattened.astype(float))

        stim_data = np.vstack(stim_rows)
        stim_info = mne.create_info(
            ch_names=state_names,
            sfreq=raw.info["sfreq"],
            ch_types="stim",
        )
        stim = mne.io.RawArray(stim_data, stim_info, first_samp=0, verbose=False)
        raw.add_channels([stim])
        self._log(f"Added {len(state_names)} stim channel(s)")

    @staticmethod
    def _get_plotting_modules() -> Any:
        """Import plotting lazily so normal import stays lightweight"""
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt

        return plt

    def _require_raw(self) -> mne.io.RawArray:
        """Return raw data or fail with a clear error"""
        if self.raw is None:
            raise RuntimeError("Raw data is not available")
        return self.raw

    def _eeg_picks(self) -> np.ndarray:
        """Return EEG channel indices"""
        raw = self._require_raw()
        picks = mne.pick_types(raw.info, eeg=True, stim=False, exclude=[])
        if len(picks) == 0:
            raise RuntimeError("No EEG channels are available")
        return picks

    def _stim_picks(self) -> np.ndarray:
        """Return stim channel indices"""
        raw = self._require_raw()
        return mne.pick_types(raw.info, eeg=False, stim=True, exclude=[])

    @staticmethod
    def _select_plot_picks(picks: np.ndarray, max_channels: int) -> np.ndarray:
        """Select a readable subset of channels for trace plots"""
        if max_channels <= 0:
            raise ValueError("max_channels must be positive")
        return picks[: min(len(picks), max_channels)]

    @staticmethod
    def _sanitize_plot_format(plot_format: str) -> str:
        """Validate requested plot format"""
        normalized = plot_format.lower().lstrip(".")
        if normalized not in SUPPORTED_PLOT_FORMATS:
            allowed = ", ".join(sorted(SUPPORTED_PLOT_FORMATS))
            raise ValueError(f"Unsupported plot format '{plot_format}'. Supported formats: {allowed}")
        return normalized

    @staticmethod
    def _save_figure(fig: Any, path: Path) -> Path:
        """Save and close a matplotlib figure"""
        fig.savefig(path, dpi=150, bbox_inches="tight")
        import matplotlib.pyplot as plt

        plt.close(fig)
        return path

    def _plot_signal_overview(
        self,
        output_dir: Path,
        plot_format: str,
        start_sec: float,
        duration_sec: float,
        max_channels: int,
    ) -> Path:
        """Save a stacked EEG trace segment"""
        if duration_sec <= 0:
            raise ValueError("duration_sec must be positive")
        if start_sec < 0:
            raise ValueError("start_sec cannot be negative")

        plt = self._get_plotting_modules()
        raw = self._require_raw()
        picks = self._select_plot_picks(self._eeg_picks(), max_channels)

        sfreq = float(raw.info["sfreq"])
        start = int(round(start_sec * sfreq))
        stop = int(round((start_sec + duration_sec) * sfreq))
        start = min(max(start, 0), raw.n_times - 1)
        stop = min(max(stop, start + 1), raw.n_times)

        data_uv = raw.get_data(picks=picks, start=start, stop=stop) * VOLTS_TO_MICROVOLTS
        times = np.arange(start, stop) / sfreq
        channel_names = [raw.ch_names[idx] for idx in picks]

        centered = data_uv - np.nanmedian(data_uv, axis=1, keepdims=True)
        scale = np.nanpercentile(np.abs(centered), 95)
        if not np.isfinite(scale) or scale == 0:
            scale = 1.0

        stacked = centered / scale
        offsets = np.arange(len(picks))[::-1] * 2.0

        fig, ax = plt.subplots(figsize=(12, max(5.0, 0.25 * len(picks) + 2.0)))
        for row, offset in zip(stacked, offsets):
            ax.plot(times, row + offset, linewidth=0.8)

        ax.set_yticks(offsets)
        ax.set_yticklabels(channel_names)
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("EEG channels")
        ax.set_title(f"EEG trace validation ({len(picks)} channels, {times[0]:.2f}-{times[-1]:.2f} s)")
        ax.grid(True, axis="x", alpha=0.25)
        fig.tight_layout()

        return self._save_figure(fig, output_dir / f"01_eeg_trace_overview.{plot_format}")

    def _plot_channel_quality(self, output_dir: Path, plot_format: str) -> Path:
        """Save simple channel-amplitude quality metrics"""
        plt = self._get_plotting_modules()
        raw = self._require_raw()
        picks = self._eeg_picks()
        montage_signal = self.montage.get("signal")

        if isinstance(montage_signal, np.ndarray) and montage_signal.shape[0] == len(picks):
            data_uv = montage_signal
        else:
            data_uv = raw.get_data(picks=picks) * VOLTS_TO_MICROVOLTS

        channel_names = np.asarray([raw.ch_names[idx] for idx in picks])
        std_uv = np.nanstd(data_uv, axis=1)
        ptp_uv = np.nanmax(data_uv, axis=1) - np.nanmin(data_uv, axis=1)
        order = np.argsort(std_uv)
        flat_mask = std_uv < EGI_FLAT_STD_THRESHOLD

        fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        axes[0].bar(np.arange(len(order)), std_uv[order])
        axes[0].set_ylabel("Std [µV]")
        axes[0].set_title("Channel standard deviation")
        axes[0].grid(True, axis="y", alpha=0.25)

        axes[1].bar(np.arange(len(order)), ptp_uv[order])
        axes[1].set_ylabel("Peak-to-peak [µV]")
        axes[1].set_title("Channel peak-to-peak amplitude")
        axes[1].grid(True, axis="y", alpha=0.25)
        axes[1].set_xticks(np.arange(len(order)))
        axes[1].set_xticklabels(channel_names[order], rotation=90, fontsize=7)
        axes[1].set_xlabel("Channels sorted by std")

        fig.suptitle(f"EEG channel quality validation ({int(np.sum(flat_mask))} nearly-flat channels)")
        fig.tight_layout()
        return self._save_figure(fig, output_dir / f"02_channel_quality.{plot_format}")

    def _plot_psd_summary(self, output_dir: Path, plot_format: str, psd_fmax: float) -> Path:
        """Save a median EEG power spectral density plot"""
        if psd_fmax <= 0:
            raise ValueError("psd_fmax must be positive")

        plt = self._get_plotting_modules()
        raw = self._require_raw()
        picks = self._eeg_picks()
        fmax = min(psd_fmax, float(raw.info["sfreq"]) / 2.0)
        n_fft = min(2048, raw.n_times)

        spectrum = raw.compute_psd(picks=picks, fmin=0.5, fmax=fmax, n_fft=n_fft, verbose=False)
        psds, freqs = spectrum.get_data(return_freqs=True)
        psd_db = 10.0 * np.log10(psds * (VOLTS_TO_MICROVOLTS**2) + np.finfo(float).eps)
        median_psd = np.nanmedian(psd_db, axis=0)
        low_psd = np.nanpercentile(psd_db, 10, axis=0)
        high_psd = np.nanpercentile(psd_db, 90, axis=0)

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(freqs, median_psd, linewidth=1.8, label="Median EEG PSD")
        ax.fill_between(freqs, low_psd, high_psd, alpha=0.25, label="10th-90th percentile")
        ax.set_xlabel("Frequency [Hz]")
        ax.set_ylabel("Power [dB µV²/Hz]")
        ax.set_title(f"PSD validation ({len(picks)} EEG channels)")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best")
        fig.tight_layout()

        return self._save_figure(fig, output_dir / f"03_psd_summary.{plot_format}")

    def _plot_sensor_layout(self, output_dir: Path, plot_format: str) -> Path:
        """Save the inferred EEG sensor layout"""
        plt = self._get_plotting_modules()
        raw = self._require_raw()
        eeg_ch_count = len(self._eeg_picks())
        show_names = eeg_ch_count <= 64

        fig, ax = plt.subplots(figsize=(8, 8))
        mne.viz.plot_sensors(
            raw.info,
            kind="topomap",
            ch_type="eeg",
            title=f"Sensor layout validation ({self.montage.get('montage_type', 'unknown')})",
            show_names=show_names,
            axes=ax,
            show=False,
        )
        fig.tight_layout()

        return self._save_figure(fig, output_dir / f"04_sensor_layout.{plot_format}")

    def _plot_stim_overview(self, output_dir: Path, plot_format: str, max_channels: int) -> Path | None:
        """Save a stim/state channel activity summary"""
        stim_picks = self._stim_picks()
        if len(stim_picks) == 0:
            return None

        plt = self._get_plotting_modules()
        raw = self._require_raw()
        picks = self._select_plot_picks(stim_picks, max_channels)
        data = raw.get_data(picks=picks)
        names = [raw.ch_names[idx] for idx in picks]

        change_counts = np.count_nonzero(np.diff(data, axis=1) != 0, axis=1)
        nonzero_counts = np.count_nonzero(data != 0, axis=1)
        order = np.argsort(change_counts + nonzero_counts)

        fig, ax = plt.subplots(figsize=(10, max(4.0, 0.28 * len(picks) + 1.5)))
        y = np.arange(len(order))
        ax.barh(y, change_counts[order], label="State changes")
        ax.barh(y, nonzero_counts[order], left=change_counts[order], alpha=0.5, label="Non-zero samples")
        ax.set_yticks(y)
        ax.set_yticklabels(np.asarray(names)[order])
        ax.set_xlabel("Samples or transitions")
        ax.set_title("Stim/state channel validation")
        ax.grid(True, axis="x", alpha=0.25)
        ax.legend(loc="best")
        fig.tight_layout()

        return self._save_figure(fig, output_dir / f"05_stim_state_overview.{plot_format}")

    def save_validation_plots(
        self,
        output_dir: str | Path,
        plot_format: str = DEFAULT_PLOT_FORMAT,
        start_sec: float = 0.0,
        duration_sec: float = 10.0,
        max_channels: int = 32,
        psd_fmax: float = 60.0,
    ) -> list[Path]:
        """Save validation plots and return generated file paths"""
        self._require_raw()
        normalized_format = self._sanitize_plot_format(plot_format)
        output_path = Path(output_dir).expanduser()
        output_path.mkdir(parents=True, exist_ok=True)

        self._log(f"Saving validation plots to {output_path}")
        saved_paths = [
            self._plot_signal_overview(output_path, normalized_format, start_sec, duration_sec, max_channels),
            self._plot_channel_quality(output_path, normalized_format),
            self._plot_psd_summary(output_path, normalized_format, psd_fmax),
            self._plot_sensor_layout(output_path, normalized_format),
        ]

        stim_path = self._plot_stim_overview(output_path, normalized_format, max_channels)
        if stim_path is not None:
            saved_paths.append(stim_path)

        self._log(f"Saved {len(saved_paths)} validation plot(s)")
        for path in saved_paths:
            self._log(f"Saved {path}", verbose_only=True)
        return saved_paths

    def _import_dat(self) -> None:
        """Import a BCI2000 .dat file and populate self.raw"""
        start_time = perf_counter()
        self._log(f"Importing .dat file: {self.path_to_file}")

        self.stream = self._read_bci2000_stream()
        self.montage = self._resolve_montage()

        montage_signal = np.asarray(self.montage["signal"], dtype=float)
        montage_type = str(self.montage["montage_type"])
        self._log(f"Resolved montage: {montage_type} ({montage_signal.shape[0]} EEG channels)")

        self.ch_set = ChannelSet(self.montage["ch_info"])
        ch_labels = list(self.ch_set.get_labels())

        self.raw = self._make_raw_with_montage(
            signal=montage_signal * MICROVOLTS_TO_VOLTS,
            fs=float(self.stream["fs"]),
            ch_names=ch_labels,
            montage_type=montage_type,
            conv_dict=eeg_dict.stand1020_to_egi,
        )

        self._log(f"Created RawArray with {len(self.raw.ch_names)} channels and {self.raw.n_times} samples")

        if self.keep_stim:
            self._add_stim_to_raw()
        else:
            self._log("Stim channels skipped", verbose_only=True)

        self._log(f"Import finished in {perf_counter() - start_time:.2f} s", verbose_only=True)

    def _import_file(self) -> None:
        """Dispatch import based on file extension and populate self.raw"""
        path = self.path_to_file
        ext = path.suffix.lower()

        if not path.is_file():
            raise FileNotFoundError(f"Input file does not exist: {path}")
        if ext not in SUPPORTED_EXTENSIONS:
            allowed = ", ".join(sorted(SUPPORTED_EXTENSIONS))
            raise ValueError(f"Unsupported file type '{ext}'. Supported types: {allowed}")

        self._import_dat()
