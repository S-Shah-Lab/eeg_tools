"""
EEGRawImporter: class

This script is a standalone utility that imports a .dat or .fif file
Generates a mne.io.Raw object with a montage and, if specified, stimulus channels
"""


import os
from typing import Any, Optional, List, Dict, Tuple

import mne
import numpy as np
from BCI2000Tools.FileReader import bcistream
from BCI2000Tools.Electrodes import ChannelSet
import helper.eeg_dict as eeg_dict


class EEGRawImporter:
    """Import EEG recordings from BCI2000 .dat files into MNE Raw objects"""

    def __init__(
        self,
        path_to_file: str,
        helper_dir  : str  = None,
        keep_stim   : bool = False,
        verbose     : bool = False,
    ) -> None:
        """Initialize the importer and load the input file into instance attributes"""
        self.path_to_file = path_to_file   # Path to input file
        self.helper_dir   = helper_dir     # Path to folder with helpers
        self.keep_stim    = keep_stim      # Allows the mne.io.Raw object to be created with or without stim channels
        self.verbose      = verbose        # Print additional information to the terminal
        
        # Automatically import the file at class initialization
        self._import_file() 



    def _read_bci2000_stream(self) -> Dict[str, Any]:
        """Decode the BCI2000 stream and return signal, states, sampling rate, and metadata"""
        # Read BCI2000 stream
        b              = bcistream(self.path_to_file)
        signal, states = b.decode()
        signal         = np.array(signal)
        fs             = float(b.samplingrate())
        ch_names       = list(b.params["ChannelNames"])
        n_channels     = signal.shape[0]
        
        year, month, day = b.params["StorageTime"].split("T")[0].split("-")
        date_test = f"{year}-{month}-{day}"
        
        return {
            "signal"     : signal,
            "states"     : states,
            "fs"         : fs,
            "ch_names"   : ch_names,
            "n_channels" : n_channels,
            "date_test"  : date_test,
        }
        
    def _resolve_path(self, fname: str) -> Dict[str, Any]:
        """Resolve a helper-file path relative to helper_dir or the current working directory"""
        try:
            path_dummy = os.path.join(self.helper_dir, fname)
            if os.path.isfile(path_dummy):
                return path_dummy
        except:
            try:
                print(f"Couldn't find the file at {self.helper_dir}")
                path_dummy = os.path.join('./', fname)
                if os.path.isfile(path_dummy):
                    return path_dummy
            except:
                print(f"Couldn't find the file at './'")
                raise RuntimeError("File couldn't be located")
                
    def _resolve_montage(self) -> Dict[str, Any]:
        """Infer montage type and associated location file from the channel count and names"""
        n_channels = self.stream['n_channels']
        ch_names   = self.stream['ch_names'  ]
        
        montage = {"montage_type": None,
                   "ch_info"     : None, 
                   "signal"      : None,
                   "ch_names"    : None,
                   }
        
        # Decide montage type & location file
        if n_channels in (21, 24):
            montage["montage_type"] = "DSI_24"
            montage["ch_info"]      = self._resolve_path("DSI24_location.txt")
            # Drop aux / trigger channels that are specific to this montage
            keep_idx = [
                i
                for i, ch in enumerate(ch_names)
                if ch not in ["X1", "X2", "X3", "TRG"]
            ]
            montage["signal"]   = self.stream["signal"][keep_idx]
            montage["ch_names"] = list(np.array(self.stream["ch_names"])[keep_idx])
            return montage

        elif n_channels == 32:
            montage["montage_type"] = "GTEC_32"
            montage["ch_info"]      = self._resolve_path("GTEC32_location.txt")
            montage["signal"]       = self.stream["signal"  ]
            montage["ch_names"]     = self.stream["ch_names"]
            return montage
            
        elif n_channels == 128:
            
            _is_egi_64 = False
            
            # Evaluate whether the file has flat channels, this might be by design
            stds = np.std(self.stream["signal"], axis=1)
            flat_chs = np.sum(stds < 0.01)
            if flat_chs >= 20: 
                _is_egi_64 = True
                
            if _is_egi_64:
                montage["montage_type"] = "EGI_64"
                montage["ch_info"]      = self._resolve_path("EGI64_location.txt")
                # Drop channels that are meant to be empty
                keep_idx = eeg_dict.id_ch_64_keep
                montage["signal"]       = self.stream["signal"][keep_idx]
                montage["ch_names"]     = list(np.array(self.stream["ch_names"])[keep_idx])
                return montage
                
            else:
                montage["montage_type"] = "EGI_128"
                montage["ch_info"]      = self._resolve_path("EGI128_location.txt")
                montage["signal"]       = self.stream["signal"  ]
                montage["ch_names"]     = self.stream["ch_names"]
                return montage

        else:
            raise ValueError(
                f"Unknown montage with {n_channels} channels. Cannot determine montage type"
            )
        
    @staticmethod
    def _make_raw_with_montage(
        signal: np.ndarray,
        fs: float,
        ch_names: List[str],
        montage_type: str,
        conv_dict: Optional[dict] = None,
    ) -> mne.io.RawArray:
        """Build an MNE RawArray with an appropriate montage"""
        info = mne.create_info(ch_names=ch_names, sfreq=fs, ch_types="eeg")
        raw  = mne.io.RawArray(signal, info, verbose=False)

        if montage_type in ["DSI_24", "GTEC_32"]:
            montage = mne.channels.make_standard_montage("standard_1020")
        elif montage_type in ["EGI_64", "EGI_128"]:
            montage = mne.channels.make_standard_montage("GSN-HydroCel-129")
        else:
            raise ValueError(f"Unknown montage_type '{montage_type}'")

        # MNE montages use lower-case names internally
        montage.ch_names = [x.lower() for x in montage.ch_names]

        idx: List[int] = []
        for ch in ch_names:
            if montage_type in ["EGI_64", "EGI_128"]:
                if conv_dict is None:
                    raise ValueError("conv_dict must be provided for EGI montages")
                mapped = conv_dict[ch.lower()]
                idx.append(montage.ch_names.index(mapped))
            else:
                idx.append(montage.ch_names.index(ch.lower()))

        montage.ch_names = ch_names
        montage.dig      = montage.dig[0:3] + [montage.dig[i + 3] for i in idx]

        raw.set_montage(montage)
        return raw
        
    def _add_stim_to_raw(self) -> None:
        """Append BCI2000 state vectors as MNE stim channels onto self.raw"""
        fs_stim   = self.raw.info["sfreq"]
        states    = self.stream["states"]
        info_stim = mne.create_info(
            ch_names=[x for x in states.keys()], sfreq=fs_stim, ch_types="stim"
        )
        stim = mne.io.RawArray(
            [x[0] for x in states.values()], info_stim, first_samp=0, verbose=False
        )
        self.raw.add_channels([stim])
        
    def _import_dat(self) -> None:
        """Import a BCI2000 .dat file and populate self.raw with montage and optional stim channels"""
        if self.verbose: print(f"[EEGRawImporter] Importing .dat file: {self.path_to_file}")

        stream       = self._read_bci2000_stream()
        self.stream  = stream
        
        print(f"Data with shape: {self.stream['signal'].shape}")
        
        montage      = self._resolve_montage()
        self.montage = montage
        
        print(f"Montage found: {self.montage['montage_type']}")
        
        ch_set       = ChannelSet(self.montage['ch_info'])
        self.ch_set  = ch_set
        
        
        raw = self._make_raw_with_montage(
            signal=self.montage["signal"] * 1e-6,
            fs=self.stream["fs"],
            ch_names=self.ch_set.get_labels(),
            montage_type=self.montage["montage_type"],
            conv_dict=eeg_dict.stand1020_to_egi,
        )
        self.raw = raw
        
        
        if self.keep_stim:
            self._add_stim_to_raw()
         
    def _import_file(self) -> None:
        """Dispatch import based on file extension and populate self.raw"""
        # Allowed file types
        allowed_file_type = [".dat"]
        
        # Extract extension = file type
        _, ext = os.path.splitext(self.path_to_file)
        ext    = ext.lower()

        # Use appropriate method based on file type
        if   ext == ".dat": self._import_dat()
        else: raise ValueError(f"Unsupported file type '{ext}'. Must be {allowed_file_type}")
        
