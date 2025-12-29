"""
EEGPreprocessorConfig: class
EEGPreprocessing     : class

Standalone EEG preprocessing class operating directly on an MNE Raw object

Pipeline (in this order):
    1) Re-referencing           (ChannelSet.RerefMatrix)
    2) Spatial filter           (ChannelSet.SLAP or similar)
    3) Notch filter             (Raw.notch_filter)
    4) Band-pass filter         (Raw.filter, Raw.info updated automatically)
    5) PREP bad-channel detect  (pyprep.NoisyChannels)
    6) Interpolation of bads    (Raw.interpolate_bads)
    7) Manual BAD segment marks (Raw.plot + Annotations)

This class assumes:
    * `raw` is an MNE Raw with EEG channels defined
    * BCI2000Tools.Electrodes.ChannelSet is available on the path
"""

from __future__ import annotations

from dataclasses import dataclass
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple, Union, Mapping, Iterable, Iterator, Literal, Callable

import numpy as np
import mne
from mne.io import BaseRaw
import matplotlib.pyplot as plt
from pyprep.find_noisy_channels import NoisyChannels
from BCI2000Tools.Electrodes import ChannelSet

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

@dataclass
class EEGPreprocessorConfig(OrderedDict[StepName, Dict[str, Any]]):
    """Ordered mapping of preprocessing steps to parameter dictionaries"""
    
    def __init__(
        self,
        steps: Optional[Iterable[Tuple[StepName, Mapping[str, Any]]]] = None,
    ) -> None:
        super().__init__()
        if steps is not None:
            for name, params in steps:
                self.add_step(name, params)

    def add_step(self, name: StepName, params: Mapping[str, Any]) -> None:
        """Add a preprocessing step and preserve insertion order for execution."""
        self[name] = dict(params)

    def iter_steps(self) -> Iterator[Tuple[StepName, Dict[str, Any]]]:
        """Yield steps in execution order as (name, params) pairs."""
        yield from self.items()

    """
    # Notch
    run_notch   : bool                                = False
    notch_freqs : Optional[Union[float, List[float]]] = None
    notch_kwargs: Dict[str, Any]                      = field(default_factory=dict)
    
    # Band-pass
    run_bandpass   : bool            = False 
    l_freq         : Optional[float] = None
    h_freq         : Optional[float] = None
    bandpass_kwargs: Dict[str, Any]  = field(default_factory=dict)

    # PREP
    run_prep        : bool = False
    prep_correlation: bool = True
    prep_deviation  : bool = True
    prep_hf_noise   : bool = True
    prep_nan_flat   : bool = True
    prep_ransac     : bool = True

    # Interpolation
    run_interpolation      : bool = False
    reset_bads_after_interp: bool = True

    # Re-reference
    run_rereference: bool = False
    reref_channels : Optional[Union[str, List[str]]] = None

    # Spatial filter
    run_spatialfilter  : bool = False
    spatial_exclude    : Optional[List[str]] = None

    # Manual annotation
    run_annotation       : bool = False
    plot                 : bool = True

    random_state: int = 83092
    """
    
class EEGPreprocessor:
    """
    High-level preprocessing wrapper for an MNE Raw object
    
    These are all the steps that can be performed:
        1) Notch filter                  (Raw.notch_filter)       [optional]
        2) Band-pass filter              (Raw.filter)             [optional]
        3) PREP bad-channel detection    (NoisyChannels)          [optional]
        4) Interpolate bad channels      (Raw.interpolate_bads)   [optional]
        5) Re-reference                  (ChannelSet.RerefMatrix) [optional]
        6) patial filter                 (ChannelSet.SLAP)        [optional]
        7) Manual BAD segment annotation (Raw.plot)               [optional]

    Provides a configurable sequence of filters, channel cleaning, and annotation-based rejection.
    """

    def __init__(
        self,
        raw    : BaseRaw,
        ch_set : ChannelSet,
        config : Optional[EEGPreprocessorConfig | Mapping[StepName, Mapping[str, Any]]] = None,
        copy   : bool = True,
        montage_type: Optional[str] = None,
        conv_dict: Optional[Mapping[str, str]] = None,
        verbose: bool = False,
    ) -> None:
        """Initialize the preprocessor with an MNE Raw object, montage info, and a preprocessing configuration"""
        if not isinstance(raw, BaseRaw):
            raise TypeError("`raw` must be an instance of mne.io.BaseRaw.")

        # Work on a copy if requested
        self.raw: BaseRaw = raw.copy() if copy else raw
        self.raw.load_data()

        self.ch_set = ch_set
        
        self._montage_type = montage_type
        self._conv_dict = conv_dict or eeg_dict.stand1020_to_egi

        if config is None:
            self.config: EEGPreprocessorConfig = EEGPreprocessorConfig()
        elif isinstance(config, EEGPreprocessorConfig):
            self.config = config
        else:
            # Accept any mapping, preserving insertion order (dicts are ordered in modern Python).
            self.config = EEGPreprocessorConfig(list(config.items()))
        self.history: Dict[str, Any] = {}
        
        self.verbose = verbose

        self._STEP_DISPATCH: Dict[StepName, Callable[[Mapping[str, Any]], None]] = {
            "rereference"  : self._apply_rereference,
            "spatialfilter": self._apply_spatialfilter,
            "notch"        : self._apply_notch_filter,
            "bandpass"     : self._apply_bandpass_filter,
            "prep"         : self._apply_prep,
            "interpolation": self._apply_interpolation_bad_channels,
            "annotation"   : self._apply_annotation,
        }

        
        # Cache the original montage so to rebuild coordinates
        # after channel remapping (rereference, spatial filters, etc.)
        self._orig_montage     = None
        self._orig_ch_pos      = {}
        self._orig_coord_frame = "head"

        try:              montage = raw.get_montage()
        except Exception: montage = None

        if montage is not None:
            pos_dict = montage.get_positions()
            # pos_dict typically contains: 'ch_pos', 'coord_frame', 'nasion', etc.
            self._orig_montage     = montage
            self._orig_ch_pos      = dict(pos_dict.get("ch_pos", {}))
            self._orig_coord_frame = pos_dict.get("coord_frame", "head")

        # Bookkeeping: EEG picks and ChannelSet
        self._eeg_picks = mne.pick_types(self.raw.info, eeg=True, exclude=())
        if self._eeg_picks.size == 0:
            raise RuntimeError("No EEG channels found in Raw.info.")

        # Initial bookkeeping
        self.history["initial_bads"]     = list(self.raw.info.get("bads", []))
        self.history["initial_highpass"] = self.raw.info.get("highpass", None)
        self.history["initial_lowpass"]  = self.raw.info.get("lowpass", None)

    # Public API --------------------------------------------------------------
    def run(self) -> Tuple[BaseRaw, Dict[str, Any]]:
        """Run preprocessing steps in the order specified by the configuration"""
        for step_name, params in self.config.iter_steps():
            if step_name not in self._STEP_DISPATCH:
                raise KeyError(f"Unknown preprocessing step: {step_name!r}.")
            # Allow disabling a step explicitly by setting it to None/False.
            if params is None or params is False:
                continue
            if not isinstance(params, Mapping):
                raise TypeError(
                    f"Parameters for step {step_name!r} must be a mapping, got {type(params)}."
                )
            self._STEP_DISPATCH[step_name](params)
        return self.raw, self.history

    # Helpers -----------------------------------------------------------------
    
    def _infer_montage_type(self, n_ch: int) -> Optional[str]:
        """Infer the montage type string from the number of EEG channels"""
        if n_ch in (21, 24):
            return "DSI_24"
        if n_ch == 32:
            return "GTEC_32"
        if n_ch == 64:
            return "EGI_64"
        if n_ch == 128:
            return "EGI_128"
        return None
    
    def _make_montage_like_importer(
        self,
        ch_names: List[str],
    ) -> Optional[mne.channels.DigMontage]:
        """
        Recreate montage selection logic used in EEGRawImporter._make_raw_with_montage
        Returns a DigMontage that matches the provided channel names
        """
        montage_type = self._montage_type or self._infer_montage_type(len(ch_names))
        if montage_type is None:
            return None

        if montage_type in ["DSI_24", "GTEC_32"]:
            montage = mne.channels.make_standard_montage("standard_1020")
            is_egi = False
        elif montage_type in ["EGI_64", "EGI_128"]:
            montage = mne.channels.make_standard_montage("GSN-HydroCel-129")
            is_egi = True
        else:
            return None

        # Mirror RawImporter behavior: compare in lower-case space
        montage.ch_names = [x.lower() for x in montage.ch_names]

        idx = []
        kept_names = []
        for ch in ch_names:
            ch_l = ch.lower()
            try:
                if is_egi:
                    mapped = self._conv_dict[ch_l]
                    idx.append(montage.ch_names.index(mapped))
                else:
                    idx.append(montage.ch_names.index(ch_l))
                kept_names.append(ch)
            except Exception:
                # If a label can’t be mapped/found, we silently skip it.
                # This avoids blowing up for derived/odd labels.
                continue

        if not idx:
            return None

        # Reduce dig to fiducials + selected channels
        montage.ch_names = kept_names
        montage.dig = montage.dig[0:3] + [montage.dig[i + 3] for i in idx]

        # Return the montage object; caller will apply it to Raw
        return montage
    
    def _reapply_montage_after_channel_change(self, new_labels: List[str]) -> None:
        """Rebuild montage after channel remapping"""
        # Try importer-style rebuild
        montage = self._make_montage_like_importer(new_labels)
        if montage is not None:
            self.raw.set_montage(montage)
            return

        # Fallback to original cached coordinates
        ch_pos = {
            name: self._orig_ch_pos[name]
            for name in new_labels
            if name in self._orig_ch_pos
        }

        if not ch_pos:
            return

        new_montage = mne.channels.make_dig_montage(
            ch_pos=ch_pos,
            coord_frame=self._orig_coord_frame,
        )
        self.raw.set_montage(new_montage)

    def _apply_notch_filter(self, params: Mapping[str, Any]) -> None:
        """Apply notch filtering to EEG channels using Raw.notch_filter"""
        notch_freqs = params.get('freqs')
        notch_kwargs: Dict[str, Any] = dict(params.get('kwargs', {}))


        if notch_freqs is None:
            raise ValueError("[Notch filter] Provide at least one frequency value")

        nyq_fs = self.raw.info["sfreq"] / 2.0

        # If a scalar is passed (e.g. 50 or 60), create harmonics up to Nyquist
        if isinstance(notch_freqs, (int, float)):
            base = float(notch_freqs)
            freqs = np.arange(base, nyq_fs, base)
        else:
            # If they passed a custom list, just use it as-is
            freqs = np.array(notch_freqs, dtype=float)

        self.raw.notch_filter(
            freqs=freqs,
            picks=self._eeg_picks,
            **notch_kwargs,
            verbose=self.verbose,
        )
        self.history["notch_filter"] = {
            "freqs" : freqs,
            "kwargs": notch_kwargs.copy(),
        }
        
        print(f"[EEGPreprocessor] Notch filter at {freqs}")

    def _apply_bandpass_filter(self, params: Mapping[str, Any]) -> None:
        """Apply band-pass (or high-/low-pass) to EEG channels with Raw.filter"""
        l_freq = params.get('l_freq')
        h_freq = params.get('h_freq')
        bandpass_kwargs: Dict[str, Any] = dict(params.get('kwargs', {}))

        
        if l_freq is None and h_freq is None:
            raise ValueError("[Bandpass filter] Provide at least one frequency value")
        
        self.raw.filter(
            l_freq=l_freq,
            h_freq=h_freq,
            picks=self._eeg_picks,
            **bandpass_kwargs,
            verbose=self.verbose,
        )

        self.history["bandpass_filter"] = {
            "l_freq"       : l_freq,
            "h_freq"       : h_freq,
            "kwargs"       : bandpass_kwargs.copy(),
            "info_highpass": self.raw.info.get("highpass", None),
            "info_lowpass" : self.raw.info.get("lowpass" , None),
        }
        
        print(f"[EEGPreprocessor] Bandpass filter at [{l_freq}-{h_freq}] Hz")
        
    def _apply_prep(self, params: Mapping[str, Any]) -> None:
        """
        Apply PREP-like bad-channel detection using pyprep.NoisyChannels
        More details about methods and thresholds can be found here:
        https://pyprep.readthedocs.io/en/latest/generated/pyprep.NoisyChannels.html
        """

        random_state     = int( params.get('random_state', 83092))
        prep_correlation = bool(params.get('correlation',  True))
        prep_deviation   = bool(params.get('deviation',    True))
        prep_hf_noise    = bool(params.get('hf_noise',     True))
        prep_nan_flat    = bool(params.get('nan_flat',     True))
        prep_ransac      = bool(params.get('ransac',       True))

        if not any(
            [
                prep_correlation,
                prep_deviation,
                prep_hf_noise,
                prep_nan_flat,
                prep_ransac,
            ]
        ):
            raise ValueError("[PREP] Provide at least one method")

        # First PREP pass (non-RANSAC)
        nc = NoisyChannels(self.raw, do_detrend=False, random_state=random_state)

        if prep_correlation:
            # Identifies bad channels by correlation
            # Correlations of channels split into 1 s windows
            nc.find_bad_by_correlation(
                correlation_secs=1.0,
                correlation_threshold=0.4,
                frac_bad=0.01,
            )

        if prep_deviation:
            # Identifies bad channels by deviation
            # Z-scoring method to find high deviations above threshold
            nc.find_bad_by_deviation(deviation_threshold=5.0)

        if prep_hf_noise:
            # Identifies bad channels by high-frequency noise
            # Ratio of amplitudes between >50Hz and overall above threshold
            nc.find_bad_by_hfnoise(HF_zscore_threshold=5.0)

        if prep_nan_flat:
            # Identifies bad channels by NaN or flat signals
            # Standard deviation or its median absolute deviation from the median (MAD) are below threshold
            nc.find_bad_by_nan_flat()

        bad_dict = nc.get_bads(as_dict=True)
        bad_names: List[str] = []

        if prep_correlation: bad_names += bad_dict["bad_by_correlation"]
        if prep_deviation:   bad_names += bad_dict["bad_by_deviation"]
        if prep_hf_noise:    bad_names += bad_dict["bad_by_hf_noise"]
        if prep_nan_flat:    bad_names += bad_dict["bad_by_nan"] + bad_dict["bad_by_flat"]

        # Merge with any existing bads
        old_bads = set(self.raw.info.get("bads", []))
        merged_bads = sorted(old_bads.union(set(bad_names)))
        self.raw.info["bads"] = merged_bads

        # Second PREP pass using RANSAC
        if prep_ransac:
            # Identifies bad channels by RANSAC
            # Random sample consensus approach to predict what the signal should be for each channel 
            # based on the signals and spatial locations of other currently-good channels
            nc_r = NoisyChannels(self.raw, do_detrend=False)
            nc_r.find_bad_by_ransac(
                n_samples=50,
                sample_prop=0.25,
                corr_thresh=0.75,
                frac_bad=0.4,
                corr_window_secs=5.0,
                channel_wise=False,
                max_chunk_size=None,
            )
            bad_dict_r = nc_r.get_bads(as_dict=True)
            merged_bads = sorted(
                set(self.raw.info["bads"]).union(
                    set(bad_dict_r["bad_by_ransac"])
                )
            )
            self.raw.info["bads"] = merged_bads

        # Fraction of bad EEG channels
        n_eeg    = self.raw.get_data(picks="eeg").shape[0]
        frac_bad = len(self.raw.info["bads"]) / n_eeg if n_eeg else 0.0

        self.history["prep"] = {
            "bads_after_prep": list(self.raw.info["bads"]),
            "n_eeg"          : n_eeg,
            "frac_bad"       : frac_bad,
        }
        
    def _apply_interpolation_bad_channels(self, params: Mapping[str, Any]) -> None:
        """Interpolate bad channels using Raw.interpolate_bads"""
        
        before = list(self.raw.info["bads"])
        if not before:
            # If the list is empty, do nothing
            return

        self.raw.interpolate_bads(
            reset_bads=bool(params.get('reset_bads', params.get('reset_bads_after_interp', True)))
        )
        after = list(self.raw.info["bads"])

        self.history["interpolation"] = {
            "bads_before": before,
            "bads_after": after,
        }
        
    def _update_raw_with_new_eeg(
        self,
        signal_new: np.ndarray,
        new_ch_set: ChannelSet,
        step_name: str,
        matrix: np.ndarray,
    ) -> None:
        """
        Replace the EEG part of self.raw with `signal_new` and update `self.ch_set`

        This keeps:
        - The same sampling frequency, measurement date, etc
        - All non-EEG channels (e.g., stim) unchanged

        If the number of EEG channels changes (e.g., due to a spatial filter that
        excludes channels), we construct a new Raw for the EEG channels and then
        re-attach the non-EEG channels
        """
        old_raw       = self.raw
        old_info      = old_raw.info
        old_eeg_picks = self._eeg_picks
        n_old_eeg     = old_eeg_picks.size

        n_new_eeg, n_times = signal_new.shape
        new_labels         = list(new_ch_set.get_labels())
        if n_new_eeg != len(new_labels):
            raise ValueError(
                f"[{step_name}] Mismatch between signal_new channels "
                f"({n_new_eeg}) and new_ch_set labels ({len(new_labels)})."
            )
        if n_times != old_raw.n_times:
            raise ValueError(
                f"[{step_name}] Mismatch in number of samples "
                f"(signal_new={n_times}, raw={old_raw.n_times})."
            )

        # Indices of non-EEG channels to preserve
        all_picks = np.arange(old_raw.info["nchan"])
        non_eeg_picks = np.setdiff1d(all_picks, old_eeg_picks)

        if n_new_eeg == n_old_eeg:
            # Case 1: same number of EEG channels -> modify in place
            # Replace EEG data
            old_raw._data[old_eeg_picks, :] = signal_new

            # Build a mapping {old_name -> new_name} for the EEG channels
            old_names = list(old_info["ch_names"])
            mapping = {
                old_names[idx]: new_label
                for idx, new_label in zip(old_eeg_picks, new_labels)
            }

            # Use the official MNE API to rename channels (updates info["chs"]
            # and info["ch_names"] internally; direct assignment is forbidden)
            old_raw.rename_channels(mapping)

            new_raw = old_raw

        else:
            # Case 2: different number of EEG channels -> build new RawArray
            eeg_info = mne.create_info(
                ch_names=new_labels,
                sfreq=old_info["sfreq"],
                ch_types="eeg",
            )
            # Preserve some global metadata that still makes sense
            eeg_info["line_freq"] = old_info.get("line_freq", None)
            eeg_info["meas_date"] = old_info.get("meas_date", None)

            raw_eeg = mne.io.RawArray(
                signal_new,
                eeg_info,
                first_samp=old_raw.first_samp,
                verbose=self.verbose,
            )

            # Re-attach non-EEG channels if any
            if non_eeg_picks.size:
                raw_other = old_raw.copy().pick(non_eeg_picks)
                raw_eeg.add_channels([raw_other])

            new_raw = raw_eeg

        # Update internal state
        self.raw = new_raw
        self.ch_set = new_ch_set

        # Recompute EEG picks based on updated Raw
        self._eeg_picks = mne.pick_types(self.raw.info, eeg=True, exclude=())

        # Log transformation in history
        self.history[step_name] = {
            "matrix_shape": matrix.shape,
            "n_eeg_old"   : n_old_eeg,
            "n_eeg_new"   : n_new_eeg,
            "ch_labels"   : new_labels,
        }
        
        # Rebuild montage so coordinates follow channel name changes
        try:
            self._reapply_montage_after_channel_change(new_labels)
        except Exception as exc:
            if self.verbose:
                print(
                    f"[{step_name}] Warning: failed to update montage after "
                    f"channel change: {exc}"
                )

    def _apply_rereference(self, params: Mapping[str, Any]) -> None:
        """Apply re-referencing using ChannelSet.RerefMatrix and update Raw"""

        # Compute rereferencing matrix from the ChannelSet
        channels = params.get('channels')
        if channels is None:
            raise ValueError("[Re-reference] 'channels' must be provided")

        m = np.array(self.ch_set.RerefMatrix(channels))

        # New ChannelSet describing the rereferenced channels
        new_ch_set = self.ch_set.copy().spfilt(m)

        # Apply reref matrix to EEG data: m has shape (n_old_eeg, n_new_eeg)
        eeg_data = self.raw.get_data(picks="eeg")   # (n_old_eeg, n_times)
        signal_new = m.T @ eeg_data                 # (n_new_eeg, n_times)

        # Update Raw + ch_set using the shared helper
        self._update_raw_with_new_eeg(
            signal_new=signal_new,
            new_ch_set=new_ch_set,
            step_name="rereference",
            matrix=m,
        )

        if self.verbose:
            print("[EEGPreprocessor] Applied re-reference.")

    def _apply_spatialfilter(self, params: Mapping[str, Any]) -> None:
        """Apply spatial filter (e.g., SLAP) and update Raw"""

        # Compute spatial filter matrix from the ChannelSet
        exclude = params.get('exclude')

        if exclude:
            m = np.array(self.ch_set.SLAP(exclude=exclude))
        else:
            m = np.array(self.ch_set.SLAP())

        # New ChannelSet describing the spatially filtered channels
        new_ch_set = self.ch_set.copy().spfilt(m)

        # Apply spatial filter: m has shape (n_old_eeg, n_new_eeg)
        eeg_data = self.raw.get_data(picks="eeg")   # (n_old_eeg, n_times)
        signal_new = m.T @ eeg_data                 # (n_new_eeg, n_times)

        # Update Raw + ch_set using the shared helper
        self._update_raw_with_new_eeg(
            signal_new=signal_new,
            new_ch_set=new_ch_set,
            step_name="spatial_filter",
            matrix=m,
        )

        if self.verbose:
            print("[EEGPreprocessor] Applied spatial filter.")

    def _apply_annotation(self, params: Mapping[str, Any]) -> None:
        """Launch an interactive Raw plot to mark BAD_region annotations and record bad file time percentage"""
        plot = bool(params.get("plot", True))

        # Initialize an empty Annotations object with a 'BAD_region' label
        region_name = "BAD_region"
        annot       = mne.Annotations([0], [0], [region_name])
        # Set the annotations to the RAW object
        self.raw.set_annotations(annot)
        # Open an interactive plot of the RAW data for visual inspection and marking
        if plot:
            self.raw.plot(block=True)
        
        # Evaluate BAD regions
        # Extract annotations from the RAW object
        annot = self.raw.annotations
        # Identify durations of annotations matching the specified label
        bad_onset    = annot.onset[   np.where(annot.description == region_name)]
        bad_duration = annot.duration[np.where(annot.description == region_name)]

        fileTime = self.raw.get_data(picks='eeg').shape[1] / self.raw.info['sfreq']

        # Percent of file classified as BAD
        total_percent = np.sum(bad_duration) / fileTime * 100

        if plot:
            fig,ax = plt.subplots(1,1, figsize=(8,2))
            # Plot the whole experiment timeline as a bar
            ax.barh(0, fileTime, color="lightgray", edgecolor="black")
            # Highlight the bad segments in red
            for onset, duration in zip(bad_onset, bad_duration):
                ax.barh(0, duration, left=onset, color='red', edgecolor="black")
            # Set labels and title
            ax.set_yticks([])
            ax.set_xlim(0, fileTime)
            ax.set_xlabel("Time [s]")
            # Remove the top and side spines
            ax.spines["top"  ].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["left" ].set_visible(False)

            # Add a text box at the end of the bar
            text_str = f"Bad Segments: {total_percent:.2f}%"
            ax.text(
                fileTime,
                0.5,
                text_str,
                va="center",
                ha="right",
                bbox=dict(facecolor="white", edgecolor="black"),
            )
            plt.show()
        
