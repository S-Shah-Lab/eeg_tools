"""
BridgingChecker

Standalone EEG quality checker for detecting candidate bridged channels

The class works on an existing MNE Raw object and keeps the original behavior:
- copy the Raw object
- enforce a band-pass filter when the current filter settings do not match
- normalize each EEG channel within each analysis window
- compute windowed correlation, distance, affinity, and bridge score matrices
- derive connected candidate channel groups
- save bridge candidate topomaps when a save path is provided
"""

from __future__ import annotations

import csv
from collections import deque
from pathlib import Path
from time import perf_counter
from typing import Any, Optional

import matplotlib.pyplot as plt
import mne
import numpy as np
from matplotlib.axes import Axes
from matplotlib.colors import ListedColormap
from matplotlib.figure import Figure


class BridgingChecker:
    """Detect candidate bridged EEG channels from an MNE Raw object"""

    def __init__(
        self,
        raw: mne.io.BaseRaw | None = None,
        verbose: bool = False,
        fmin: float = 1.0,
        fmax: float = 40.0,
        sigma: float = 0.05,
        window_sec: float = 20.0,
        bridge_score_threshold: float = 0.095,
        show_extra: bool = False,
        figure: Figure | int | None = None,
        axes: Axes | dict[str, float] | tuple[float, float, float, float] | None = None,
        save_path: str | Path | None = None,
        auto_run: bool = True,
    ) -> None:
        """Initialize the checker and optionally run the default bridge analysis"""
        if raw is None:
            raise ValueError("raw must be an initialized mne.io.Raw object")
        if fmin < 0:
            raise ValueError("fmin must be non-negative")
        if fmax <= fmin:
            raise ValueError("fmax must be greater than fmin")
        if sigma <= 0:
            raise ValueError("sigma must be positive")
        if window_sec <= 0:
            raise ValueError("window_sec must be positive")
        if bridge_score_threshold < 0:
            raise ValueError("bridge_score_threshold must be non-negative")

        self.raw = raw.copy()
        self.verbose = verbose
        self.fmin = float(fmin)
        self.fmax = float(fmax)
        self.sigma = float(sigma)
        self.window_sec = float(window_sec)
        self.bridge_score_threshold = float(bridge_score_threshold)
        self.show_extra = show_extra
        self.save_path = Path(save_path).expanduser() if save_path is not None else None

        self._figure_target = figure
        self._axes_bbox = axes

        self.data: np.ndarray
        self.ch_names: list[str]
        self.info: mne.Info
        self.windows: list[tuple[int, int]] = []
        self.groups: list[list[int]] = []
        self.group_channel_names: list[list[str]] = []

        self._filter_raw()
        self.data, self.ch_names, self.info = self._get_qc_data()
        self.file_time = self.data.shape[1] / float(self.raw.info["sfreq"])

        if self.file_time <= 0:
            raise ValueError("raw contains no usable samples")
        if self.window_sec > self.file_time:
            self._log(
                f"Requested window ({self.window_sec:.2f} s) exceeds recording length "
                f"({self.file_time:.2f} s); using one full-recording window"
            )
            self.window_sec = self.file_time

        if auto_run:
            self.run_bridge_analysis()

    def _log(self, message: str, *, verbose_only: bool = False) -> None:
        """Print checker status messages"""
        if self.verbose or not verbose_only:
            print(f"[BridgingChecker] {message}")

    def _filter_raw(self) -> None:
        """Filter the Raw object when current limits do not match the requested limits"""
        current_highpass = float(self.raw.info.get("highpass", 0.0))
        current_lowpass = float(self.raw.info.get("lowpass", np.inf))
        already_filtered = np.isclose(current_highpass, self.fmin) and np.isclose(current_lowpass, self.fmax)

        if already_filtered:
            self._log(
                f"Input already filtered at {current_highpass:g}-{current_lowpass:g} Hz",
                verbose_only=True,
            )
            return

        self._log(
            f"Filtering Raw copy from {current_highpass:g}-{current_lowpass:g} Hz "
            f"to {self.fmin:g}-{self.fmax:g} Hz"
        )
        self.raw.filter(l_freq=self.fmin, h_freq=self.fmax, verbose=False)

    def _get_qc_data(self) -> tuple[np.ndarray, list[str], mne.Info]:
        """Extract EEG data and matching channel names"""
        picks = mne.pick_types(self.raw.info, eeg=True, stim=False, exclude=[])
        if len(picks) == 0:
            raise RuntimeError("No EEG channels were found in the Raw object")

        data = self.raw.get_data(picks=picks)
        ch_names = [self.raw.ch_names[idx] for idx in picks]
        return data, ch_names, self.raw.info

    @staticmethod
    def _sliding_windows(
        n_samples: int,
        sfreq: float,
        window_sec: float,
        step_sec: float,
    ) -> list[tuple[int, int]]:
        """Generate non-overlapping sliding windows"""
        window_size = int(round(window_sec * sfreq))
        step_size = int(round(step_sec * sfreq))

        if window_size <= 0:
            raise ValueError("window_sec must map to at least one sample")
        if step_size <= 0:
            raise ValueError("step_sec must map to at least one sample")
        if n_samples <= window_size:
            return [(0, n_samples)]

        windows: list[tuple[int, int]] = []
        start = 0
        while start + window_size <= n_samples:
            stop = start + window_size
            windows.append((start, stop))
            start += step_size

        return windows

    @staticmethod
    def _normalize_data(data: np.ndarray) -> np.ndarray:
        """Normalize each channel to zero mean and unit variance"""
        means = np.mean(data, axis=1, keepdims=True)
        stds = np.std(data, axis=1, ddof=1, keepdims=True)
        stds = np.where(stds == 0, 1.0, stds)
        return (data - means) / stds

    @staticmethod
    def _distance_matrix(x: np.ndarray) -> np.ndarray:
        """Compute pairwise channel distances without explicit channel loops"""
        n_samples = x.shape[1]
        squared_norms = np.mean(x * x, axis=1)
        gram = (x @ x.T) / max(n_samples, 1)
        squared_distances = squared_norms[:, None] + squared_norms[None, :] - 2.0 * gram
        squared_distances = np.maximum(squared_distances, 0.0)
        return np.sqrt(squared_distances)

    @staticmethod
    def _gauss_affinity(d: np.ndarray, sigma: float) -> np.ndarray:
        """Build a row-normalized Gaussian affinity matrix from a distance matrix"""
        affinity = np.exp(-(d * d) / (2.0 * sigma * sigma))
        row_sums = affinity.sum(axis=1, keepdims=True)
        return np.divide(
            affinity,
            row_sums,
            out=np.zeros_like(affinity),
            where=row_sums > 0,
        )

    def _windowed_analysis(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Compute median correlation, distance, affinity, and score matrices across windows"""
        correlations: list[np.ndarray] = []
        distances: list[np.ndarray] = []
        affinities: list[np.ndarray] = []
        scores: list[np.ndarray] = []

        for start, stop in self.windows:
            segment = self.data[:, start:stop]
            if segment.shape[1] < 2:
                continue

            norm_segment = self._normalize_data(segment)
            corr_mat = np.corrcoef(norm_segment)
            dist_mat = self._distance_matrix(norm_segment)
            aff_mat = self._gauss_affinity(dist_mat, sigma=self.sigma)
            score_mat = corr_mat * aff_mat

            correlations.append(corr_mat)
            distances.append(dist_mat)
            affinities.append(aff_mat)
            scores.append(score_mat)

        if not scores:
            raise RuntimeError("No valid windows found for bridge analysis")

        return (
            np.median(np.asarray(correlations), axis=0),
            np.median(np.asarray(distances), axis=0),
            np.median(np.asarray(affinities), axis=0),
            np.median(np.asarray(scores), axis=0),
        )

    @staticmethod
    def _connected_groups_from_upper_mat(matrix: np.ndarray) -> list[list[int]]:
        """Convert an upper triangular adjacency matrix into connected groups"""
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            raise ValueError("matrix must be a square 2D array")

        adjacency = (matrix > 0) | (matrix.T > 0)
        n_nodes = adjacency.shape[0]
        visited = np.zeros(n_nodes, dtype=bool)
        groups: list[list[int]] = []

        for start in range(n_nodes):
            if visited[start]:
                continue

            queue: deque[int] = deque([start])
            visited[start] = True
            group: list[int] = []

            while queue:
                node = queue.popleft()
                group.append(node)

                for neighbor in np.flatnonzero(adjacency[node]):
                    if not visited[neighbor]:
                        visited[neighbor] = True
                        queue.append(int(neighbor))

            if len(group) > 1:
                groups.append(sorted(group))

        return groups

    def _plot_matrices(self) -> Figure:
        """Plot matrices used by the bridge analysis"""
        matrix_specs = [
            ("Windowed correlation", self.corr_mat, -1, 1),
            ("Windowed distance [a.u.]", self.dist_mat, None, None),
            ("Windowed affinity [Gaussian]", self.aff_mat, 0, 1),
            ("Windowed bridge score [corr x affinity]", self.score_mat, -1, 1),
            ("Binary adjacency mask", self.mask_bridge_score, 0, 1),
        ]

        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.ravel()

        for ax, (title, matrix, vmin, vmax) in zip(axes, matrix_specs):
            ax.set_title(title)
            image = ax.imshow(matrix, vmin=vmin, vmax=vmax)
            fig.colorbar(image, ax=ax, fraction=0.0475, pad=0.01)
            ax.set_xticks(np.arange(len(self.ch_names)))
            ax.set_xticklabels(self.ch_names, rotation=-45, ha="left", fontsize=7)
            ax.set_yticks(np.arange(len(self.ch_names)))
            ax.set_yticklabels(self.ch_names, fontsize=7)

        axes[5].set_axis_off()
        axes[5].set_title("Candidate groups", fontsize=12)
        if self.group_channel_names:
            for idx, group in enumerate(self.group_channel_names, start=1):
                axes[5].text(
                    0.05,
                    0.95 - (idx - 1) * 0.1,
                    f"Group {idx}: {', '.join(group)}",
                    transform=axes[5].transAxes,
                    fontsize=10,
                    va="top",
                )
        else:
            axes[5].text(0.05, 0.95, "No candidate groups", transform=axes[5].transAxes, fontsize=10)

        fig.tight_layout()
        return fig

    @staticmethod
    def _plot_group_topomap(
        info: mne.Info,
        groups: list[list[int]],
        ax: Axes,
        title: str,
    ) -> None:
        """Plot candidate groups on a topomap"""
        picks = mne.pick_types(info, eeg=True, stim=False, exclude=[])
        sub_info = mne.pick_info(info, sel=picks)
        ch_names = [info["ch_names"][idx] for idx in picks]

        group_values = np.zeros(len(picks), dtype=int)
        pick_index_map = {ch_idx: local_idx for local_idx, ch_idx in enumerate(picks)}

        for group_id, group_indices in enumerate(groups, start=1):
            for ch_idx in group_indices:
                local_idx = pick_index_map.get(ch_idx)
                if local_idx is not None:
                    group_values[local_idx] = group_id

        n_groups = len(groups)
        default_cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
        if len(default_cycle) < n_groups:
            repeats = int(np.ceil(n_groups / max(len(default_cycle), 1)))
            default_cycle = (default_cycle * repeats)[:n_groups]

        color_list = ["white"] + default_cycle[:n_groups]
        cmap = ListedColormap(color_list)

        mne.viz.plot_topomap(
            group_values.astype(float),
            sub_info,
            ch_type="eeg",
            sensors=True,
            cmap=cmap,
            show=False,
            axes=ax,
            names=ch_names,
            contours=0,
            vlim=(0, max(1, n_groups)),
            image_interp="nearest",
        )
        ax.set_title(title, fontsize=10)

    @staticmethod
    def _axes_to_bbox(axes: Any) -> dict[str, float]:
        """Convert supported axes placement inputs into GridSpec arguments"""
        if isinstance(axes, dict):
            return {
                key: float(value)
                for key, value in axes.items()
                if key in {"left", "bottom", "right", "top", "wspace", "hspace"}
            }
        if isinstance(axes, (list, tuple)) and len(axes) == 4:
            left, bottom, right, top = [float(value) for value in axes]
            return {"left": left, "bottom": bottom, "right": right, "top": top}
        return {}

    def _plot_bridged_candidates(
        self,
        figure: Figure | int | None = None,
        axes: Axes | dict[str, float] | tuple[float, float, float, float] | None = None,
    ) -> Figure:
        """Plot the final candidate groups on a topomap"""
        from matplotlib.gridspec import GridSpec

        bbox = self._axes_to_bbox(axes)
        if isinstance(figure, int):
            fig = plt.figure(figure)
        elif figure is not None:
            fig = figure
        else:
            fig = plt.figure(figsize=(5, 5))

        if bbox:
            grid = GridSpec(1, 1, figure=fig, **bbox)
            ax = fig.add_subplot(grid[0, 0])
        else:
            ax = fig.add_subplot(1, 1, 1)

        self._plot_group_topomap(
            info=self.raw.info,
            groups=self.groups,
            ax=ax,
            title=f"Bridged candidate groups: filtered [{self.fmin:g}-{self.fmax:g}] Hz",
        )
        fig.tight_layout()
        return fig

    def _save_outputs(self) -> list[Path]:
        """Save bridge figures and CSV summaries"""
        if self.save_path is None:
            return []

        self.save_path.mkdir(parents=True, exist_ok=True)
        saved_paths: list[Path] = []

        if hasattr(self, "bridging_checker_fig"):
            png_path = self.save_path / "bridged_candidates.png"
            svg_path = self.save_path / "bridged_candidates.svg"
            self.bridging_checker_fig.savefig(png_path, bbox_inches="tight", dpi=150)
            self.bridging_checker_fig.savefig(svg_path, bbox_inches="tight")
            saved_paths.extend([png_path, svg_path])

        csv_path = self.save_path / "bridged_candidate_groups.csv"
        with csv_path.open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["group_id", "channel_index", "channel_name"])
            for group_id, group in enumerate(self.groups, start=1):
                for channel_index in group:
                    writer.writerow([group_id, channel_index, self.ch_names[channel_index]])
        saved_paths.append(csv_path)

        return saved_paths

    def run_bridge_analysis(
        self,
        plot: bool = True,
        figure: Figure | int | None = None,
        axes: Axes | dict[str, float] | tuple[float, float, float, float] | None = None,
    ) -> list[list[int]]:
        """Run the bridge analysis and return candidate channel index groups"""
        start_time = perf_counter()
        sfreq = float(self.info["sfreq"])
        n_channels, n_samples = self.data.shape

        self._log(
            f"Analyzing {n_channels} EEG channels, {n_samples} samples, "
            f"{self.file_time:.2f} s at {sfreq:g} Hz"
        )

        self.windows = self._sliding_windows(
            n_samples=n_samples,
            sfreq=sfreq,
            window_sec=self.window_sec,
            step_sec=self.window_sec,
        )
        self._log(
            f"Using {len(self.windows)} window(s) of ~{self.window_sec:.2f} s "
            f"with threshold {self.bridge_score_threshold:g}"
        )

        self.corr_mat, self.dist_mat, self.aff_mat, self.score_mat = self._windowed_analysis()
        self.mask_bridge_score = np.triu(self.score_mat >= self.bridge_score_threshold, k=1)
        self.groups = self._connected_groups_from_upper_mat(self.mask_bridge_score)
        self.group_channel_names = [[self.ch_names[idx] for idx in group] for group in self.groups]

        if plot and self.show_extra:
            self.matrix_fig = self._plot_matrices()
            plt.show()

        if plot:
            self.bridging_checker_fig = self._plot_bridged_candidates(
                figure=figure if figure is not None else self._figure_target,
                axes=axes if axes is not None else self._axes_bbox,
            )
            if self.show_extra:
                plt.show()

        saved_paths = self._save_outputs()
        elapsed = perf_counter() - start_time

        if self.group_channel_names:
            formatted_groups = [", ".join(group) for group in self.group_channel_names]
            self._log(f"Detected {len(self.groups)} candidate group(s): {' | '.join(formatted_groups)}")
        else:
            self._log("Detected no candidate bridged-channel groups")

        if saved_paths:
            for path in saved_paths:
                self._log(f"Saved {path}", verbose_only=True)

        self._log(f"Bridge analysis finished in {elapsed:.2f} s", verbose_only=True)
        return self.groups
