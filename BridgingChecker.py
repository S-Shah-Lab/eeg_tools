"""
BridgingChecker: class

Standalone EEG quality checker for detecting candidate bridged channels

This module provides the `BridgingChecker` class, which:
    * enforces a band-pass filter between `fmin` and `fmax` when needed on an existing mne.io.Raw object
    * normalizes each EEG channel independently to zero mean and unit variance
    * performs a windowed analysis of:
        - channel correlation
        - electrical distance
        - Gaussian affinity
        - a bridge score (correlation x affinity)
    * derives connected groups of channels with high bridge score
    * visualizes results via:
        - matrix plots (correlation, distance, affinity, bridge score, mask)
        - scalp topographies highlighting candidate bridged-channel groups
"""

import os, sys

from typing import List, Tuple

import mne
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

if "IPython" in sys.modules:
    plt.ion()

class BridgingChecker:
    """
    Standalone EEG quality checker for bridged channels
    Works on a mne.io.Raw object
    """

    def __init__(
        self,
        raw: mne.io.BaseRaw = None,
        verbose: bool = False,
        fmin: float = 1.0,
        fmax: float = 40.0,
        sigma: float = 0.05,
        window_sec: float = 20.0,
        bridge_score_threshold: float = 0.095,
        show_extra: bool = False,
        figure = None,      
        axes = None,  
        save_path: str = None,
    ) -> None:
        """
        Initialize the bridging checker with analysis parameters and an optional MNE Raw object
        
        Parameters
        ----------
        raw : mne.io.BaseRaw
            Raw object containing EEG time series
        verbose : bool
            Verbosity flag for showing more information on terminal
        fmin : float
            Lower cutoff frequency for band-pass filtering (Hz)
        fmax : float
            Upper cutoff frequency for band-pass filtering (Hz)
        sigma : float
            Threshold value (sigma) to use for similarity (gaussian kernel)
        window_sec : float
            Length in seconds for generating windowed analysis
        bridge_score_threshold: float
            Threshold to use on the correlation-affinity product to identify candidates
        show_extra: bool
            Flag to display matrices in the bridging analysis
        figure:
            TODO
        axes:
            TODO
        """

        self.raw                    = raw.copy()
        self.verbose                = verbose
        self.fmin                   = fmin
        self.fmax                   = fmax
        self.sigma                  = sigma
        self.window_sec             = window_sec
        self.bridge_score_threshold = bridge_score_threshold
        self.show_extra             = show_extra

        self._figure_target         = figure
        self._axes_bbox             = axes
        self.save_path              = save_path

        self._filter_raw()
        self.data, self.ch_names, self.info = self._get_qc_data()

        # Adapt the window to the file length
        self.fileTime = self.data.shape[1] / self.raw.info['sfreq']
        if self.fileTime < self.window_sec:
            self.window_sec = self.fileTime - 1
            print(f"Changed the window to the current file length: {round(self.window_sec, 1)} s")

        # Default bridge analysis on initialization
        self.run_bridge_analysis()

    # Helpers

    def _filter_raw(self) -> None:
        """Filter mne.io.Raw object for consistency"""
        if self.raw is not None:
            if self.raw.info['highpass'] == 1.0 and self.raw.info['lowpass'] == 40.0:
                pass
            else:
                if self.verbose:
                    print(
                        f"[BridgingChecker] Filtering mne.io.Raw from "
                        f"{self.fmin} to {self.fmax} Hz"
                    )
                self.raw.filter(
                    l_freq=self.fmin,
                    h_freq=self.fmax,
                    verbose=False,
                )
        else:
            raise RuntimeError("mne.io.Raw object is not initialized")

    def _get_qc_data(self) -> Tuple[np.ndarray, List[str], mne.Info]:
        """Band-pass filter the Raw to the configured frequency range when needed"""
        data     = self.raw.get_data(picks="eeg")
        eeg_chs  = data.shape[0]
        ch_names = self.raw.ch_names[:eeg_chs]
        info     = self.raw.info

        return data, ch_names, info

    @staticmethod
    def _sliding_windows(
        n_samples: int,
        sfreq: float,
        window_sec: float,
        step_sec: float,
    ) -> List[Tuple[int, int]]:
        """
        Generate (start, stop) indices for sliding windows

        If the recording is shorter than the window size, return a single
        window covering the full data
        Discard last chunk if it doesn't have a proper window
        """
        window_size = int(round(window_sec * sfreq))
        step_size   = int(round(step_sec   * sfreq))

        if window_size <= 0: raise ValueError("window_sec must be positive and non-zero")
        if step_size <= 0  : raise ValueError("step_sec must be positive and non-zero")

        if n_samples <= window_size: return [(0, n_samples)]

        windows: List[Tuple[int, int]] = []
        start = 0
        while start + window_size <= n_samples:
            stop = start + window_size
            windows.append((start, stop))
            start += step_size

        print(
            f"Using {len(windows)} windows of length ~{window_sec:.2f} s "
            f"with step ~{step_sec:.2f} s"
        )

        return windows

    @staticmethod
    def _normalize_data(data: np.ndarray) -> np.ndarray:
        """Normalize each channel to zero mean and unit variance independently"""
        means           = np.mean(data, axis=1, keepdims=True)
        stds            = np.std(data, axis=1, ddof=1, keepdims=True)
        stds[stds == 0] = 1.0 # security 
        normed          = (data - means) / stds
        return normed

    @staticmethod
    def _l2(x1: np.ndarray, x2: np.ndarray) -> float:
        """Euclidean distance (mean)"""
        return float(np.sqrt(np.mean((x1 - x2) ** 2)))

    @staticmethod
    def _gaussian(x: float, sigma: float) -> float:
        """Evaluate a Gaussian kernel value for a distance using standard deviation sigma"""
        return float(
            np.exp(-(x ** 2) / (2.0 * sigma ** 2))
            / (sigma * np.sqrt(2.0 * np.pi))
        )

    def _distance_matrix(self, x: np.ndarray) -> np.ndarray:
        """Compute pairwise distances between channel using the provided time series"""
        
        n_channels = x.shape[0]
        d = np.zeros((n_channels, n_channels), dtype=float)

        for i in range(n_channels):
            for j in range(i, n_channels):
                d_ij = self._l2(x[i], x[j])
                d[i, j] = d_ij
                d[j, i] = d_ij

        return d

    def _gauss_affinity(
        self,
        d: np.ndarray,
        sigma: float,
    ) -> np.ndarray:
        """
        Build a row-normalized affinity matrix from a distance matrix

        Parameters
        ----------
        d : np.ndarray
            Distance matrix (arbitrary units)
        sigma : float
            Gaussian kernel scale (same units as d)

        Returns
        -------
        a : np.ndarray
            Row-normalized affinity matrix
        """
        n = d.shape[0]
        a = np.zeros((n, n), dtype=float)

        for i in range(n):
            for j in range(i, n):
                a[i, j] = self._gaussian(d[i, j], sigma)
                a[j, i] = a[i, j]

        for i in range(n):
            denom = np.sum(a[i, :])
            if denom > 0:
                a[i, :] /= denom

        return a

    def _windowed_analysis(self) -> List[np.ndarray]:
        """
        Evaluate correlation, distance, affinity, and bridging score for the EEG data in a windowed fashion
        Currently the window dimension is collapsed via median
        """
        sigma  = self.sigma  # sigma used in gaussian affinity

        # Evaluate correlation, distance, affinity, and bridging score matrice per window
        # Collapse the window dimension via median
        correlations = []
        distances    = []
        affinities   = []
        scores       = []

        valid_windows = 0

        for start, stop in self.windows:
            segment = self.data[:, start:stop]
            if segment.shape[1] < 2:
                continue
            else:
                norm_segment  = self._normalize_data(segment) # Normalize the windows independently
                corr_mat_seg  = np.corrcoef(norm_segment)
                dist_mat_seg  = self._distance_matrix(norm_segment)
                aff_mat_seg   = self._gauss_affinity(dist_mat_seg, sigma=sigma)
                score_mat_seg = corr_mat_seg * aff_mat_seg

                correlations.append(corr_mat_seg)
                distances.append(dist_mat_seg)
                affinities.append(aff_mat_seg)
                scores.append(score_mat_seg)

                valid_windows += 1

        correlations = np.array(correlations)
        distances    = np.array(distances)
        affinities   = np.array(affinities)
        scores       = np.array(scores)

        if valid_windows == 0:
            raise RuntimeError("No valid windows found for analysis")
        else:
            corr_mat  = np.median(correlations, axis=0)
            dist_mat  = np.median(distances,    axis=0)
            aff_mat   = np.median(affinities,   axis=0)
            score_mat = np.median(scores,       axis=0)

        return [corr_mat, dist_mat, aff_mat, score_mat]

    @staticmethod
    def _connected_groups_from_upper_mat(A: np.ndarray) -> List[List[int]]:
        """Convert an upper triangular matrix into a list of connected groups"""
        from collections import deque

        # Input matrix must be square
        if A.ndim != 2 or A.shape[0] != A.shape[1]:
            raise ValueError("A must be a square 2D array")

        # Make the upper triangular matrix symmetric
        adj = ((A > 0) | (A.T > 0))

        n = adj.shape[0]
        visited = np.zeros(n, dtype=bool)
        groups = []

        # Loop over the rows
        for start in range(n):
            if visited[start]:
                continue # If you have treated this row before, skip it

            unique_group = []
            q = deque([start])     # Initialize conteiner elements to explore
            visited[start] = True  # Set row as visited to avoid repetition 

            while q:
                u = q.popleft()                   # Take first element
                unique_group.append(u)
                neighbors = np.nonzero(adj[u])[0] # Find all neighbors v of u
                # Explore neighbors
                for v in neighbors:
                    if not visited[v]:
                        visited[v] = True
                        q.append(v)               # Set non visited neighbor for exploration

            unique_group.sort()
            groups.append(unique_group)

        # Filter only groups with more than one connected node
        groups = [x for x in groups if len(x) > 1]

        return groups

    # Plotting utilities ------------------------------------------------------

    def _plot_matrices(self) -> None:
        """Plot useful matrices for the bridging analysis"""
        x_labels = self.ch_names

        nrows = 2
        ncols = 3
        fig, ax = plt.subplots(2, 3, figsize=(ncols * 5, nrows * 5))
        ax = ax.ravel()
        # Correlation
        ax[0].set_title("Windowed correlation")
        cax0 = ax[0].imshow(self.corr_mat, vmin=-1, vmax=1)
        fig.colorbar(cax0, ax=ax[0], fraction=0.0475, pad=0.01)
        ax[0].set_xticks(np.arange(len(x_labels)))
        ax[0].set_xticklabels(x_labels, rotation=-45)
        ax[0].set_yticks(np.arange(len(x_labels)))
        ax[0].set_yticklabels(x_labels)
        # Electrical distance
        ax[1].set_title("Windowed distance [a.u.]")
        cax1 = ax[1].imshow(self.dist_mat)
        fig.colorbar(cax1, ax=ax[1], fraction=0.0475, pad=0.01)
        ax[1].set_xticks(np.arange(len(x_labels)))
        ax[1].set_xticklabels(x_labels, rotation=-45)
        ax[1].set_yticks(np.arange(len(x_labels)))
        ax[1].set_yticklabels(x_labels)
        # Affinity
        ax[2].set_title("Windowed affinity [gaussian]")
        cax2 = ax[2].imshow(self.aff_mat, vmin=0, vmax=1)
        fig.colorbar(cax2, ax=ax[2], fraction=0.0475, pad=0.01)
        ax[2].set_xticks(np.arange(len(x_labels)))
        ax[2].set_xticklabels(x_labels, rotation=-45)
        ax[2].set_yticks(np.arange(len(x_labels)))
        ax[2].set_yticklabels(x_labels)
        # Bridging score
        ax[3].set_title("Windowed bridge score [corr x aff]")
        cax3 = ax[3].imshow(self.score_mat, vmin=-1, vmax=1)
        fig.colorbar(cax3, ax=ax[3], fraction=0.0475, pad=0.01)
        ax[3].set_xticks(np.arange(len(x_labels)))
        ax[3].set_xticklabels(x_labels, rotation=-45)
        ax[3].set_yticks(np.arange(len(x_labels)))
        ax[3].set_yticklabels(x_labels)
        # Mask (binary adjacency)
        ax[4].set_title("Binary adjacency mask")
        cax4 = ax[4].imshow(self.mask_bridge_score, vmin=0, vmax=1)
        fig.colorbar(cax4, ax=ax[4], fraction=0.0475, pad=0.01)
        ax[4].set_xticks(np.arange(len(x_labels)))
        ax[4].set_xticklabels(x_labels, rotation=-45)
        ax[4].set_yticks(np.arange(len(x_labels)))
        ax[4].set_yticklabels(x_labels)
        # Print groups
        ax[5].set_axis_off()
        ax[5].set_title("Candidate groups", fontsize=12)
        y_position = 0.9  # Starting y-position (top of the axes)
        for group in self.groups:
            ax[5].text(0.1, y_position, [self.ch_names[x] for x in group], transform=ax[5].transAxes, fontsize=12)
            y_position -= 0.1  # Move down for the next item
        
        plt.tight_layout()
        plt.show()

    @staticmethod
    def _plot_group_topomap(
        info: mne.Info,
        groups: List[List[int]],
        ax: plt.Axes,
        title: str,
    ) -> None:
        """Plot a topomap where each candidate group has its own color"""
        # Use all EEG channels for the topomap
        picks = mne.pick_types(info, eeg=True, exclude=())
        sub_info = mne.pick_info(info, sel=picks)
        ch_names = [info["ch_names"][idx] for idx in picks]

        n_channels = len(picks)
        group_values = np.zeros(n_channels, dtype=int)

        # Map global channel indices to local indices in `picks`
        pick_index_map = {ch_idx: local_idx for local_idx, ch_idx in enumerate(picks)}

        for group_id, group_indices in enumerate(groups, start=1):
            for ch_idx in group_indices:
                local_idx = pick_index_map.get(ch_idx)
                if local_idx is not None:
                    group_values[local_idx] = group_id

        n_groups = len(groups)

        # Base color for non-group channels + one color per group
        base_color = 'white'
        default_cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
        if len(default_cycle) < n_groups:
            repeats = int(np.ceil(n_groups / max(len(default_cycle), 1)))
            default_cycle = (default_cycle * repeats)[:n_groups]

        color_list = [base_color] + default_cycle[:n_groups]
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

    def _plot_bridged_candidates(self, figure=None, axes=None) -> None:
        """Plot the final candidate groups on the topoplot"""
        import matplotlib.pyplot as plt
        from matplotlib.gridspec import GridSpec
        import numpy as np
        
        bbox = {}
        if isinstance(axes, dict):
            for k in ("left", "bottom", "right", "top", "wspace", "hspace"):
                if k in axes:
                    bbox[k] = float(axes[k])
        elif isinstance(axes, (list, tuple)) and len(axes) == 4:
            l, b, r, t = [float(v) for v in axes]
            bbox = dict(left=l, bottom=b, right=r, top=t)

        if isinstance(figure, int):
            fig2 = plt.figure(figure)
        elif figure is not None:
            fig2 = figure  # assume Figure object
        else:
            fig2 = None

        # create figure if none given
        if fig2 is None:
            fig2, axs2 = plt.subplots(1, 1, figsize=(5, 5))
        else:
            # use provided figure and make one axes inside bbox (if any)
            if bbox:
                gs_kwargs = {}
                for k in ("left", "bottom", "right", "top", "wspace", "hspace"):
                    if k in bbox:
                        gs_kwargs[k] = bbox[k]
                gs = GridSpec(1, 1, figure=fig2, **gs_kwargs)
                axs2 = fig2.add_subplot(gs[0, 0])
            else:
                axs2 = fig2.add_subplot(1, 1, 1)
        
        
        self._plot_group_topomap(
            info=self.raw.info,
            groups=self.groups,
            ax=axs2,
            title=f"Bridged candidate groups: filtered [{self.fmin}-{self.fmax}] Hz",
        )
        
        plt.tight_layout()
        
        if self.show_extra:
            plt.show()
            
        return fig2
        
    def run_bridge_analysis(self, plot=True, figure=None, axes=None) -> List[List[str]]:
        """Run windowed bridge analysis on the normalized EEG data"""
        if self.verbose:
            print("[BridgingChecker] Running windowed bridge analysis..")

        deltaT = self.window_sec # window of EEG to consider for windowed analysis

        # Define EEG window start and end points
        sfreq = float(self.info["sfreq"])
        n_channels, n_samples = self.data.shape

        self.windows = self._sliding_windows(
            n_samples=n_samples,
            sfreq=sfreq,
            window_sec=deltaT, # window length in seconds
            step_sec=deltaT,   # step to consider between subsequent window starting points
        )

        self.corr_mat, self.dist_mat, self.aff_mat, self.score_mat = self._windowed_analysis()

        # Generate a mask using the bridging score threshold in input
        mask_bridge_score      = self.score_mat >= self.bridge_score_threshold # mask
        self.mask_bridge_score = np.triu(mask_bridge_score, k=1)     # mask upper off-diagonal 
        self.groups            = self._connected_groups_from_upper_mat(self.mask_bridge_score)


        if plot and self.show_extra:
            self._plot_matrices()


        if plot and self.info is not None and self.ch_names is not None:
            self.bridging_checker_fig = self._plot_bridged_candidates(
                figure=(figure if figure is not None else self._figure_target),
                axes=(axes if axes is not None else self._axes_bbox),
            )
            
            if self.save_path is not None:
                self.bridging_checker_fig.savefig(os.path.join(self.save_path, "bridged_candidates.png"), bbox_inches="tight")
                self.bridging_checker_fig.savefig(os.path.join(self.save_path, "bridged_candidates.svg"), bbox_inches="tight")
                
        return self.groups
