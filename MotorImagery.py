"""
EEGMotorImagery: class

TODO
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import re, os
import numpy as np
import mne
from mne.io import BaseRaw
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.font_manager import FontProperties
from mpl_toolkits.axes_grid1 import make_axes_locatable

from BCI2000Tools.Electrodes import ChannelSet

from scipy.signal import hilbert


class EEGMotorImagery:
    """
    Docstring for EEGMotorImagery
    """
    def __init__(
        self,
        raw    : BaseRaw,
        ch_set : ChannelSet,
        nEpochs: int = 6, 
        duration_task: float = 10., 
        skip: float = 1., 
        resolution: float = 1.,
        freq_bands: List = [4.0, 8.0, 13.0, 31.0],
        nSim: int = 2999,
        strict: bool = False,
        copy: bool = True,
        verbose: bool = False,
        save_path: str = None,
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
        
        self._evaluate_motorimagery_paradigm(swap=False)
        self._generate_epochs()
        self._generate_psds()
        self._grab_left_right_electrodes()
        
        self._run_stat_tests()
        
        # Plot distributions for a single comparison
        # fig, axs = self._plot_test_distributions("left_vs_left_rest", transf="r2")
        
        # Plot both left and right distributions
        plots = self._plot_all_test_distributions(transf="r2")
        
        # Plot topomaps left vs right across bands
        fig, axs = self._plot_lateralized_topomaps()
        
        # Plot log-pvalues 
        (mi_left, ax_left), (mi_right, ax_right) = self._plot_band_pvalues_bootstrap_separate()
        
        # Plot PSDs per channel
        if self.strict: 
            chs_to_plot = ['fc3', 'fc4', 'c3', 'c4', 'cp3', 'cp4']
        else:
            chs_to_plot = ['fc3', 'fc4', 'c3', 'c4', 'p3',  'p4' ]
        
        for ch in chs_to_plot:
            try:
                fig, (ax_psd, ax_r2) = self._plot_channel_psd(ch)
            except:
                pass
            
        self._plot_paradigm_timeline()
        self._plot_band_effect(ci=95)
        
        
        
        
    
    @staticmethod
    def step_down_events(signal, time=None, tol=0.0):
        """TODO"""
        if time is None: time = np.arange(signal.size)
        else:
            time = np.asarray(time)
            if time.shape != signal.shape:
                raise RuntimeError("`time` and `signal` must have the same shape.")

        diff = np.diff(signal)
        step_indices = np.where(diff < -tol)[0] + 1  # +1 to get index of new (lower) value
        step_times = time[step_indices]
        if step_times.size >= 2: durations = np.diff(step_times)
        else: durations = np.array([], dtype=float)

        # Discard it if before 2 seconds given the current paradigm
        while step_times[0] < 2000:
            step_times = step_times[1:]
            durations  = durations[1:]

        # Grab the offset of instruction
        onset  = step_times[0::2]
        offset = step_times[1::2]
        return onset, offset, durations
    
    @staticmethod
    def find_step_intervals(signal, threshold=0.):
        """TODO"""
        active = signal > threshold
        changes = np.diff(active.astype(int))
        onsets = np.where(changes == 1)[0] + 1
        offsets = np.where(changes == -1)[0] + 1
        
        # Discard it if before 2 seconds given the current paradigm
        while onsets[0] < 2000: onsets = onsets[1:]        
        
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
        """Convert p to -log10(p) for a compact 'evidence' score"""
        p = float(np.clip(p, 1e-300, 1.0))
        #return -np.log10(p)
        return float(-np.log(p))
            
    @staticmethod
    def bands_from_edges(band_edges: List[float]) -> List[Tuple[float, float]]:
        """
        Convert band edge list to consecutive (low, high) bands

        Example:
            [4, 8, 13, 31] -> [(4, 7), (8, 12), (13, 31)]
        """
        edges = list(map(float, band_edges))
        if len(edges) < 2:
            raise ValueError("band_edges must contain at least two values.")
        if any(np.diff(edges) <= 0):
            raise ValueError("band_edges must be strictly increasing.")
        
        edges = list(zip(edges[:-1], edges[1:]))
        edges = [(x,y-1) for x,y in edges]
        
        return edges

    @staticmethod
    def band_key(low_f: float, high_f: float) -> str:
        """Format a canonical band key used by this class"""
        return f"{low_f:g}-{high_f:g}Hz"

    @staticmethod
    def build_custom_cmap(colors: List[str]) -> matplotlib.colormaps:
        """TODO"""
        
        c1, c2, c3 = colors
        
        colors = [
            (0.0, c1),  # Color at -1
            (0.5, c2),  # Color at 0
            (1.0, c3),  # Color at 1
        ]  

        return LinearSegmentedColormap.from_list("custom_cmap", colors)
     
    def _evaluate_motorimagery_paradigm(self, swap=True):
        """TODO"""
        nEpochs       = self.nEpochs
        duration_task = self.duration_task
        skip          = self.skip
        
        stimCode    = None
        presentDisp = None
        
        stimCode = self.raw.get_data(picks='StimulusCode').ravel()

        try:
            presentDisp = self.raw.get_data(picks='PresentationDisplayed').ravel()
        except:
            pass
        
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
        
        # Force the rest after right hand to have code = 4
        for i,c in enumerate(code):
            if (i+1) % 4 == 0: code[i] = 4
            
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
                     
        for i,c in enumerate(code):
            onset_.append(onset[i])
            duration_.append(duration[i])
            if   c == 1: description_.append( "left"  )
            elif c == 2: description_.append( "right" )
            elif c == 3: 
                if swap: description_.append( "right_rest" )
                else:    description_.append( "left_rest"  )
            elif c == 4: 
                if swap: description_.append( "left_rest"  )
                else:    description_.append( "right_rest" )

        # Skip the first `skip` seconds of each segment
        # Convert onset_ to seconds
        onset_ = [ t / self.raw.info['sfreq'] + skip for t in onset_ ]    
        
        # Trial number annotations
        delta = (duration_task - skip) / nEpochs
        
        onset_number = []
        duration_number = []
        description_number = []
        
        k = 0
        for i,t in enumerate(onset_):
            if description_[i] == 'left': k += 1
            for j in range(nEpochs):
                onset_number.append( t + delta * j )
                duration_number.append( delta )
                description_number.append( f"{description_[i]}_{k}" )
        
        onset_       = onset_bad       + onset_       + onset_number
        duration_    = duration_bad    + duration_    + duration_number
        description_ = description_bad + description_ + description_number
                 
        # Set annotation to mne.io.Raw
        my_annot = mne.Annotations(onset=onset_, duration=duration_, description=description_)
        self.raw.set_annotations(my_annot)
    
    def _make_epochs_batch(self, event_id: int):
        """TODO""" 
        
        tmin = 0
        tmax = (self.duration_task - self.skip) / self.nEpochs
        
        events_from_annot, event_dict = mne.events_from_annotations(self.raw, verbose=False)

        # Create epochs from RAW data based on the specified event ID and time window
        epochs_ = mne.Epochs(
            self.raw,
            events_from_annot,
            event_id=event_id,
            tmin=tmin,
            tmax=tmax,
            baseline=None,
            preload=True,
            verbose=False,
        )
        # (number of epochs, channel, t)
        return epochs_

    def _generate_epochs(self):
        """TODO"""
        
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
    ):
        """TODO"""
        
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

        # Compute PSD using Welch's method
        psd_ = self.epochs_dict[ batch ].compute_psd(
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
        ).get_data()

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

        return psd_

    def _generate_psds(self):
        """TODO"""
        
        fs   = self.raw.info['sfreq'   ]
        fmin = self.raw.info['highpass']
        fmax = self.raw.info['lowpass' ]
        
        secPerSegment = 1.0 / self.resolution
        secOverlap    = secPerSegment / 2.0
        nPerSegment   = int( secPerSegment * fs)
        nOverlap      = int( secOverlap    * fs)
        
        psds_dict = {}
        psds_epochs_dict = {}
        
        for key in self.epochs_dict.keys():
            try:
                # Epoch-averaged (for your stats pipeline)
                psds_dict[ key ] = self._make_psds_batch(
                    batch=key,
                    resolution=self.resolution,
                    fmin=fmin,
                    fmax=fmax,
                    nPerSegment=nPerSegment,
                    nOverlap=nOverlap,
                    aggregate_epochs=True,
                )

                # Per-epoch (for within-trial dynamics plots)
                psds_epochs_dict[key] = self._make_psds_batch(
                    batch=key,
                    resolution=self.resolution,
                    fmin=fmin,
                    fmax=fmax,
                    nPerSegment=nPerSegment,
                    nOverlap=nOverlap,
                    aggregate_epochs=False,
                )
            except Exception:
                pass
        
        
        if len(list(psds_dict.keys())) == 0: 
            raise RuntimeError("There are no available PSDs")
        
        self.psds_dict = psds_dict
        self.psds_epochs_dict = psds_epochs_dict
    
    def _grab_left_right_electrodes(self):
        """TODO"""
        
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
        """ Compute point-biserial correlation between features and a binary label"""
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
            "neg_log10_p": self.neg_p(p),
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
            "neg_log10_p": self.neg_p(p),
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

        # Local dB conversion as 10 * log10(X * 1e12)
        def _to_db(x: np.ndarray) -> np.ndarray:
            x = np.asarray(x)
            return 10.0 * np.log10(x * 1e12)

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

        # Infer frequency axis from RAW info
        fs = self.raw.info["sfreq"]
        fmin = float(self.raw.info.get("highpass", 0.0) or 0.0)
        fmax = float(self.raw.info.get("lowpass", fs / 2.0) or (fs / 2.0))
        
        resolution = self.resolution  # assuming this exists on the class
        bins = np.arange(fmin, fmax + resolution, resolution)

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
            band_idx = np.where((bins >= low_f) & (bins <= high_f))[0]
            if band_idx.size == 0:
                raise ValueError(f"No frequency bins fall inside band [{low_f}, {high_f}) Hz")
            start = band_idx[0 ]
            stop  = band_idx[-1]
            
            x_task = np.mean(x_task_all[:, :, start:stop], axis=2)
            x_ctrl = np.mean(x_ctrl_all[:, :, start:stop], axis=2)
            
            # Convert to dB to match original analysis expectations
            x_task_db = _to_db(x_task)
            x_ctrl_db = _to_db(x_ctrl)
            
            # Stack trials: task first, control second
            x = np.concatenate([x_ctrl_db, x_task_db], axis=0)

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
                "x_task": x_task_db,
                "x_control": x_ctrl_db,
                "effect_per_channel": effect_per_ch,
                "permutation": perm,
                "bootstrap": boot,
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
        verbose: bool = True,
    ) -> Dict[str, Any]:

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

        self.stats_results = results
        return results
    
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
            ax.axvline(float(null_val), color="red", label=f"$H_0$: $\Delta$ = {float(null_val):.1f}")

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
    ):
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
            
            fig.savefig(os.path.join(self.save_path, f"stat_distribution_{key}.png"), bbox_inches="tight")
            fig.savefig(os.path.join(self.save_path, f"stat_distribution_{key}.svg"), bbox_inches="tight")
            
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
    ):
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
            fig.savefig(os.path.join(self.save_path, "topoplot.png"), bbox_inches="tight")
            fig.savefig(os.path.join(self.save_path, "topoplot.svg"), bbox_inches="tight")
        
        return fig, axs_rows

    def _plot_frequency_bands(self, ax=None, ylim=None, fontsize=12, fraction=0.13):
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
    ):
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
            # Fallback: reconstruct bins from RAW info
            fs = float(self.raw.info["sfreq"])
            fmin = float(self.raw.info.get("highpass", 0.0) or 0.0)
            fmax = float(self.raw.info.get("lowpass", fs / 2.0) or (fs / 2.0))
            resolution = float(self.resolution)
            freqs = np.arange(fmin, fmax + resolution, resolution)

        if freqs.shape[0] != n_bins:
            raise RuntimeError(
                f"Frequency bins length ({freqs.shape[0]}) does not match PSD bins ({n_bins})"
            )

        # Slice PSDs for the selected channel and convert to dB
        def _to_db(x: np.ndarray) -> np.ndarray:
            """Convert PSDs to dB using 10 * log10(x * 1e12)"""
            x = np.asarray(x)
            return 10.0 * np.log10(x * 1e12)

        # PSDs per condition: shape (blocks, bins)
        x_left        = psds_left[:,       ch_idx, :]
        x_right       = psds_right[:,      ch_idx, :]
        x_left_rest   = psds_left_rest[:,  ch_idx, :]
        x_right_rest  = psds_right_rest[:, ch_idx, :]

        x_left_db        = _to_db(x_left)
        x_right_db       = _to_db(x_right)
        x_left_rest_db   = _to_db(x_left_rest)
        x_right_rest_db  = _to_db(x_right_rest)

        # Global min/max across all conditions for reasonable y-limits
        x_all_db = np.vstack([x_left_db, x_right_db, x_left_rest_db, x_right_rest_db])
        min_val = float(np.min(x_all_db))
        max_val = float(np.max(x_all_db))

        min_y = int(min_val - np.abs(min_val) * 0.05)
        max_y = int(max_val + np.abs(max_val) * 0.02)

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

        for y in _multiples_of_n(min_y, max_y, 5):
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
        ax1.set_ylabel("[dB]", loc="top")
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
            fig.savefig(os.path.join(self.save_path, f"psd_{ch_name}.png"), bbox_inches="tight")
            fig.savefig(os.path.join(self.save_path, f"psd_{ch_name}.svg"), bbox_inches="tight")

        return fig, (ax1, ax2)

    def _plot_band_pvalues_bootstrap_separate(
        self,
        transf: str = "r2",
        results: Optional[Dict[str, Any]] = None,
        figsize: Optional[Tuple[float, float]] = None,
    ):
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
            fig_left.savefig(os.path.join(self.save_path, "pvalue_left.png"), bbox_inches="tight")
            fig_left.savefig(os.path.join(self.save_path, "pvalue_left.svg"), bbox_inches="tight")
            
            fig_right.savefig(os.path.join(self.save_path, "pvalue_right.png"), bbox_inches="tight")
            fig_right.savefig(os.path.join(self.save_path, "pvalue_right.svg"), bbox_inches="tight")

        return (fig_left, ax_left), (fig_right, ax_right)





    # 1) Paradigm sanity: timeline plot + counts
    def _summarize_annotations(self) -> Dict[str, int]:
        """
        Count occurrences of each annotation description.
        Useful to sanity-check the paradigm parsing.
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
    ):
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
        except Exception:
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
            fig.savefig(
                os.path.join(self.save_path, "paradigm_timeline.png"),
                bbox_inches="tight",
            )
            fig.savefig(
                os.path.join(self.save_path, "paradigm_timeline.svg"),
                bbox_inches="tight",
            )

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
        figsize: Tuple[float, float] = (8.0, 4.0),
    ):
        """
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
            key=lambda s: float(s.split("-")[0]) if "-" in s else 0.0
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

        # Left $\Delta$
        ax.errorbar(
            obs_l, y - 0.12,
            xerr=[obs_l - lo_l, hi_l - obs_l],
            fmt="o", capsize=3, label="Left hand (vs rest)"
        )
        # Right $\Delta$
        ax.errorbar(
            obs_r, y + 0.12,
            xerr=[obs_r - lo_r, hi_r - obs_r],
            fmt="o", capsize=3, label="Right hand (vs rest)"
        )

        ax.axvline(0.0, linestyle="--", linewidth=1, color='grey', label=r'$\Delta = 0$')
        ax.set_yticks(y)
        ax.set_yticklabels([bk.replace("Hz", " Hz") for bk in band_keys])
        ax.invert_yaxis()  # first bands at the top
        ax.set_xlabel(r"Observed $\Delta$", loc='right')
        ax.set_title("Band-wise observed value with 95% CI", loc="left", fontsize=10)
        ax.legend(frameon=False)
        fig.tight_layout()


        # Explain $\Delta$ at the bottom of the figure
        delta_note1 = (
            r"$\Delta$ = $\sum$ ipsilateral signed $r^2$ "
            r"$-$ $\sum$ contralateral signed $r^2$"
        )
        fig.subplots_adjust(bottom=0.2)
        fig.text(0.01, 0.05, delta_note1, ha="left", va="bottom", fontsize=9)
        delta_note2 = r"Computed from band-averaged PSD (task vs rest) using target channels"
        fig.text(0.01, 0.02, delta_note2, ha="left", va="bottom", fontsize=9)

        if self.save_path is not None:
            fig.savefig(os.path.join(self.save_path, "band_effect.png"), bbox_inches="tight")
            fig.savefig(os.path.join(self.save_path, "band_effect.svg"), bbox_inches="tight")

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
        in the Epochs' channel order.
        """
        if not hasattr(self, "raw"):
            raise RuntimeError("Missing Raw.")
        # Use raw channel order (Epochs inherit it)
        chs = [c.lower() for c in self.raw.info["ch_names"]]
        name_set = set([n.lower() for n in names_lower])
        return [i for i, c in enumerate(chs) if c in name_set]

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

        # Frequency axis consistent with your Welch setup
        fs = float(self.raw.info["sfreq"])
        fmin = float(self.raw.info.get("highpass", 0.0) or 0.0)
        fmax = float(self.raw.info.get("lowpass", fs / 2.0) or (fs / 2.0))
        resolution = float(self.resolution)

        freqs = np.arange(fmin, fmax + resolution, resolution)
        if freqs.size > n_bins:
            freqs = freqs[:n_bins]
        elif freqs.size < n_bins:
            # Fallback: interpolate to expected length (rare, but avoids hard crashes)
            freqs = np.linspace(fmin, fmax, n_bins)

        low_f, high_f = float(band[0]), float(band[1])
        band_idx = np.where((freqs >= low_f) & (freqs <= high_f))[0]
        if band_idx.size == 0:
            raise ValueError(f"No frequency bins fall inside band [{low_f}, {high_f}] Hz")

        start = int(band_idx[0])
        stop = int(band_idx[-1]) + 1  # inclusive

        # Mean across channels and frequency bins => bandpower per epoch
        bp = np.mean(psd[:, :, start:stop], axis=(1, 2))
        return bp

    def _plot_within_trial_bandpower(
        self,
        task_prefix: str,
        rest_prefix: str,
        band: Tuple[float, float],
        roi: str = "contra",
        ci: float = 95.0,
        figsize: Tuple[float, float] = (10.0, 4.0),
    ):
        """
        Plot within-trial bandpower across epoch segments, with a CI across trials.

        roi:
        - "contra": contralateral motor channels
        - "ipsi": ipsilateral motor channels (via montage symmetry)
        - "both": both hemispheres (all MI channels)
        """
        if not hasattr(self, "epochs_dict"):
            raise RuntimeError("No epochs_dict. Run _generate_epochs() first")

        # Decide contralateral hemisphere from task
        task_prefix_l = task_prefix.lower().rstrip("_")
        contra_set    = set([c.lower() for c in (self.chRight if task_prefix_l.startswith("left") else self.chLeft)])
        ipsi_set      = set([c.lower() for c in (self.chLeft if task_prefix_l.startswith("left") else self.chRight)])
        both_set      = set([c.lower() for c in self.ch_motor_imagery])

        if roi == "contra": picks = self._resolve_picks_from_names(list(contra_set))
        elif roi == "ipsi": picks = self._resolve_picks_from_names(list(ipsi_set  ))
        else:               picks = self._resolve_picks_from_names(list(both_set  ))

        task_keys = self._trial_keys_for_prefix(task_prefix)
        rest_keys = self._trial_keys_for_prefix(rest_prefix)
        n_trials = min(len(task_keys), len(rest_keys))
        if n_trials == 0:
            raise RuntimeError(f"No trials found for {task_prefix=} or {rest_prefix=}")

        # Each key contains nEpochs epochs (segments). We will average within each segment index across trials.
        delta = (self.duration_task - self.skip) / float(self.nEpochs)
        t = np.arange(self.nEpochs) * delta + (delta / 2.0)

        task_mat = []
        rest_mat = []

        for i in range(n_trials):
            # Bandpower per segment (epoch) in that trial
            bp_task = self._bandpower_envelope(task_keys[i], band=band, picks=picks)
            bp_rest = self._bandpower_envelope(rest_keys[i], band=band, picks=picks)

            # Defensive: if something weird happened, truncate to nEpochs
            bp_task = bp_task[: self.nEpochs]
            bp_rest = bp_rest[: self.nEpochs]

            if bp_task.size == self.nEpochs and bp_rest.size == self.nEpochs:
                task_mat.append(bp_task)
                rest_mat.append(bp_rest)

        task_mat = np.asarray(task_mat, dtype=float)  # (n_trials, nEpochs)
        rest_mat = np.asarray(rest_mat, dtype=float)

        if task_mat.size == 0:
            raise RuntimeError("No usable trials after filtering/truncation.")

        # ERD/ERS style: percent change relative to rest
        denom = np.mean(rest_mat, axis=0) + 1e-12
        rel = 100.0 * (np.mean(task_mat, axis=0) - np.mean(rest_mat, axis=0)) / denom

        # CI across trials on the relative metric (simple percentile across trial means)
        rel_trials = 100.0 * (task_mat - rest_mat) / (rest_mat + 1e-12)
        lo, hi = np.percentile(rel_trials, [(100 - ci) / 2.0, 100 - (100 - ci) / 2.0], axis=0)

        fig, ax = plt.subplots(1, 1, figsize=figsize)
        ax.plot(t, rel, marker="o")
        ax.fill_between(t, lo, hi, alpha=0.2)
        ax.axhline(0.0, linestyle="--", linewidth=1)

        ax.set_xlabel("Time within trial [s]")
        ax.set_ylabel("Relative change vs rest [%]")
        ax.set_title(f"{task_prefix} vs {rest_prefix} | {band[0]:g}-{band[1]:g} Hz | ROI={roi}", loc="left")
        ax.grid(True, linestyle=":", alpha=0.4)
        fig.tight_layout()

        if self.save_path is not None:
            fname = f"within_trial_{task_prefix}_{band[0]:g}-{band[1]:g}Hz_{roi}.png".replace(" ", "")
            fig.savefig(os.path.join(self.save_path, fname), bbox_inches="tight")

        return fig, ax



    # 4) Laterality index over time (the thing reviewers love)
    def _plot_within_trial_laterality_index(
        self,
        task_prefix: str,
        rest_prefix: str,
        band: Tuple[float, float],
        ci: float = 95.0,
        figsize: Tuple[float, float] = (10.0, 4.0),
    ):
        """
        Laterality index (contra - ipsi) / (contra + ipsi) within trial.
        Computed on bandpower envelopes per segment.
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
            fname = f"laterality_{task_prefix}_{band[0]:g}-{band[1]:g}Hz.png".replace(" ", "")
            fig.savefig(os.path.join(self.save_path, fname), bbox_inches="tight")

        return fig, ax




