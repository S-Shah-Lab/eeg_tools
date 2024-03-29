import os
import numpy as np
from BCI2000Tools.FileReader import bcistream
from BCI2000Tools.Electrodes import *
from BCI2000Tools.Plotting import *
import mne 
import matplotlib.pyplot as plt
import matplotlib 
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable
from pyprep.prep_pipeline import PrepPipeline, NoisyChannels

import eeg_dict # Contains dictionaries and libraries for electrodes locations 

class EEG():
    """
    A class for loading and preprocessing EEG data files.

    This class provides methods to import EEG data from various file formats,
    apply preprocessing steps, and extract relevant features for further analysis.
    It uses the MNE-Python library for handling EEG data information and preprocessing.
    It uses the BCI2000Tools library for reading the original EEG data and handle different montages.
    Other libraries are also used such as: pyprep, numpy, os, and matplotlib

    Attributes:
        None

    Methods:
        create_folder
        import_file_dat
        evaluate_mi_paradigm
        import_file_fif
        save_RAW
        filter_data
        spatial_filter
        find_ch_index
        make_RAW
        make_RAW_stim
        make_PREP
        mark_BAD_region
        evaluate_BAD_region
        make_annotation_MI
        change_StimulusCode
        expand_onset
        generate_onsets
        make_montage
        find_ch_central
        find_ch_left
        find_ch_right
        find_ch_circle
        find_ch_symmetry
        interpolate
        make_epochs
        make_psd
        convert_dB
    """

    def __init__(self):
        pass

    def clean_path(self, path=None):
        """
        Cleans the path string to avoid errors.

        Args:
            path (str, optional): The directory path where some file is located.
                                  If None, the current working directory is assumed.
        """
        # If path is None, use an empty string (assuming current directory)
        # Else, ensure path ends with a '/'
        if path is None:
            path = ''
        elif not path.endswith('/'):
            path += '/'
        return path


    def create_folder(self, path=None, folder_name=None, verbose=False):
        """
        Creates a folder at a specified path.

        Args:
            path (str, optional): The directory path where the folder will be created. If None, it defaults to the current working directory.
            folder_name (str): The name of the folder to be created. If None, no folder will be created.
            verbose (bool, optional): If True, prints additional messages about the folder creation process. Defaults to False.

        Returns: 
            str: path to folder to be used later when saving files
        """
        # Clean path string
        path = self.clean_path(path)
        # Check if the folder does not already exist at the location
        if not os.path.exists(path + folder_name):
            # Create the folder at the specified location
            os.makedirs(path + folder_name)
            # If verbose mode is enabled, print a success message
            if verbose: 
                print(f"Folder '{folder_name}' created successfully at {path}")
        else:
            # If the folder already exists, print a message 
            print(f"Folder '{folder_name}' already exists at {path}")

        return path + folder_name


    def import_file_dat(self, path=None, file_name=None, verbose=True):
        """
        Imports and processes EEG data from a .dat file.

        Args:
            path (str, optional): The file path leading to the .dat file
            file_name (str): The name of the .dat file to be imported.
            verbose (bool, optional): If True, prints detailed information

        Returns:
            tuple: A tuple containing the processed signal data, state information, sampling rate, channel names, block size, and montage type
        """
        # Clean path string
        path = self.clean_path(path)
        # Load the .dat file 
        b = bcistream(path + file_name)
        signal, states = b.decode()
        signal = np.array(signal)
        
        # Retrieve additional parameters from the file
        fs = b.samplingrate()  # Sampling rate
        ch_names = b.params['ChannelNames']  # Channel names
        blockSize = b.params.SampleBlockSize  # Block size used in data acquisition

        # Extract stimulus-related information
        StimulusCode = states['StimulusCode']  # Stimulus codes
        StimulusBegin = states['StimulusBegin']  # Onsets of stimuli
        # Extract file time on tape 
        fileTime = signal.shape[1] / fs

        # Set montage type based on number of channels detected
        if   signal.shape[0] == 24:  montage_type = 'DSI_24'
        elif signal.shape[0] == 32:  montage_type = 'GTEC_32'
        elif signal.shape[0] == 128: montage_type = 'EGI_128'
        else: print(f'WARNING: I don not know this montage with {signal.shape[0]} channels!')

        # Reject some channels for 'DSI_24'
        if montage_type == 'DSI_24':
            # Example: Removing unwanted channels
            chKeep_idx = [i for i, ch in enumerate(ch_names) if ch not in ['X1', 'X2', 'X3', 'TRG']]
            signal = signal[chKeep_idx]
            ch_names = np.array(ch_names)[chKeep_idx].tolist()

        if verbose: 
            # Summary includes channel info, sampling rate, and time details
            print(f'\nEEG channels: {signal.shape[0]} Total ticks: {signal.shape[1]}')
            print(f'Each tick corresponds to [s]: {1/fs}')
            print(f'Sampling rate [Hz]: {fs} ~~~ Time on file [s]: {fileTime}')
            print(f'Montage Detected: {montage_type}')
            print(f'Signal range: [{np.min(signal)}, {np.mean(signal)}, {np.max(signal)}]')
            print(f'StimulusCode: {np.unique(StimulusCode, return_counts=True)}\n')

        return signal, states, fs, ch_names, blockSize, montage_type


    def evaluate_mi_paradigm(self, signal=None, states=None, fs=None, blockSize=None, verbose=True):
        """
        Evaluate information related to a specific motor imagery (MI) paradigm.

        Args:
            signal (numpy array): The time-series signal 
            states (BCI2000Tools.Container.Bunch): object containing EEG file information such as stimuli
            fs (float): sampling frequency
            blockSize (float): size of the block used when EEG file was recorded
            verbose (bool, optional): If True, prints detailed information

            path (str, optional): The file path leading to the .dat file
            file_name (str): The name of the .dat file to be imported.
            

        Returns:
            tuple: A tuple containing the file length in seconds, the number of blocks recorded, the trials per block, 
                   the initial gap before the first block is seconds, the duration of the cue in seconds, and the duration of each task in seconds
        """

        # Define paradigm constants
        _nBlocks=8
        _trialsPerBlock=4
        _initialSec=2
        _stimSec=3
        _taskSec=10

        # Check if the block size is a perfect divisor of the sampling frequency, this will ensure every block to be saved
        if fs % blockSize == 0:
            print(f'Block size [{blockSize}] fits perfectly in sample frequency [{fs/blockSize}]!')
        else: 
            # Warn if there's a mismatch, potentially leading to data loss
            print(f'WARNING: Block size DOES NOT fit perfectly in sample frequency!')
            # Detailed information on the potential data loss
            print(f'    Losing { ( _initialSec * fs ) % blockSize } samples from initial gap')
            print(f'    Losing { ( _stimSec * fs ) % blockSize } samples from cues')
            print(f'    Losing { ( _taskSec * fs ) % blockSize } samples from tasks')

        # Adjust the timings based on potential data loss
        _initialSec = _initialSec - ((_initialSec*fs)%blockSize) / fs
        _stimSec = _stimSec - ((_stimSec*fs)%blockSize) / fs
        _taskSec = _taskSec - ((_taskSec*fs)%blockSize) / fs

        # Calculate the total duration of the file based on experiment design (theoretical, if all trials are saved)
        totalFileTime = _initialSec + _nBlocks*(_trialsPerBlock*(_stimSec + _taskSec))

        # Extract saved file time 
        fileTime = signal.shape[1] / fs

        # If verbose, print a detailed summary of the .dat file content
        if verbose: 
            print(f'This file contains {round(fileTime / totalFileTime * 100, 2)}% of MI paradigm')
            # Additional calculations for recorded blocks and trials
            netTime = fileTime - _initialSec
            blockTime = _trialsPerBlock*(_stimSec + _taskSec)
            recordedBlocks = int(round(netTime / blockTime, 3))
            print(f'    Number of full blocks: { recordedBlocks }')
            leftOver = netTime - recordedBlocks * blockTime
            print(f'    Additional trials: { abs(leftOver) // (_stimSec + _taskSec) }')

        return fileTime, _nBlocks, _trialsPerBlock, _initialSec, _stimSec, _taskSec


    def import_file_fif(self, path=None, file_name=None):
        """
        Imports EEG data from a .fif file using MNE-Python.

        Args:
            path (str, optional): The directory path where the .fif file is located.
                                  If None, current working directory is assumed 
            file_name (str): The name of the .fif file to be imported.

        Returns:
            tuple: A tuple containing the raw data object (`RAW`), the electrode montage (`montage`), and the sampling frequency (`fs`) extracted from the file.
        """
        # Clean path string
        path = self.clean_path(path)
        # Read the .fif file into a RAW object with data preloaded
        RAW = mne.io.read_raw(path + file_name, preload=True) 
        # Retrieve the montage used in the recording from the RAW object
        montage = RAW.get_montage()
        # Extract the sampling frequency from the RAW object's information
        fs = RAW.info['sfreq']
        return RAW, montage, fs


    def save_RAW(self, RAW=None, path=None, file_name=None, label=None):
        """
        Saves an EEG data object to a .fif file with the option to include a label in the filename. 
        With `overwrite=True` any existing file with the same name will be replaced without warning.

        Args:
            RAW (mne.io.Raw): The MNE-Python Raw object containing EEG data to be saved.
            path (str, optional): The directory path where the .fif file will be saved. If None, current working directory is assumed
            file_name (str): The base name for the .fif file
            label (str, optional): An optional label to be appended to the file name

        Returns:
            None: 
        """
        # Clean path string
        path = self.clean_path(path)
        # Clean label string
        if label is None: 
            label = ''
        # Saves the RAW data to the specified file, overwriting any existing file
        RAW.save(path + file_name + label + '.fif', overwrite=True)


    def filter_data(self, signal=None, fs=None, l_freq=None, h_freq=None):
        """
        This function uses MNE-Python's filter_data method to apply a Finite Impulse Response (FIR) bandpass filter to the given signal.
        The function converts the input signal to `float64` to ensure compatibility with the MNE-Python filtering function

        Args:
            signal (numpy array): The time-series signal to be filtered
            fs (float): The sampling frequency of the signal in Hz
            l_freq (float): The lower frequency bound of the filter in Hz
            h_freq (float): The upper frequency bound of the filter in Hz

        Returns:
            array-like: The filtered signal, which is the same shape as the input signal.
        """
        # Convert the signal to float64 for processing and apply the FIR bandpass filter
        return mne.filter.filter_data(signal.astype('float64'), fs, l_freq, h_freq, verbose=False)


    def spatial_filter(self, sfilt=None, ch_set=None, signal=None, flag_ch=None, verbose=True):
        """
        Applies a spatial filter to an EEG signal to enhance signal quality.

        Args:
            sfilt (str): Specifies the type of spatial filter to apply. Accepts 'SLAP'
                         for Spatial Laplacian filtering or 'REF' for re-referencing.
            ch_set (BCI2000Tools.Electrodes.ChannelSet): An object representing the set of channels/electrodes, from BCI2000Tools.Electrodes
            signal (np.array): The EEG signal data 
            flag_ch (str or list, optional): A str with channel names separated by spaces or a list of channel names, they are used differently based on the spatial filter selected
            verbose (bool, optional): If True, displays a graphical representation of the spatial filter matrix applied to the channels.

        Returns:
            tuple: A tuple containing the spatially filtered signal 
        """
        # Apply a Spatial Laplacian filter if specified (SLAP)
        if sfilt == 'SLAP':
            if flag_ch:
                m = np.array(ch_set.SLAP(exclude=flag_ch))
            else:
                m = np.array(ch_set.SLAP())

        # Apply a Re-reference filter if specified (REF)
        elif sfilt == 'REF':
            m = np.array(ch_set.RerefMatrix(flag_ch))

        # Create a new channel set object with the spatial filter applied
        new_ch_set = ch_set.copy().spfilt(m)

        # Apply the spatial filter matrix to the signal
        signalNew = m.T @ signal

        # If verbose, display the filtering matrix with labels for channels
        if verbose: 
            imagesc(m, x=ch_set.get_labels(), y=new_ch_set.get_labels(), colorbar=True)

        return signalNew, new_ch_set


    def find_ch_index(self, ch_set=None, ch_name=None):
        """
        Finds and returns the indices of specified channel(s) within a channel set.

        Args:
            ch_set (BCI2000Tools.Electrodes.ChannelSet): The channel set containing channel labels, from BCI2000Tools.Electrodes
            ch_name (str or list): A string or list of strings specifying the channel(s) to find (examples 'Cz', 'Cz C3 C4')

        Returns:
            List of indices for the specified channel(s), or None if a channel is the current reference.
        """
        # Delegate to ch_set's find_labels method to locate channel indices
        return ch_set.find_labels(ch_name)


    def make_RAW(self, signal=None, fs=None, ch_names=None):
        """
        Constructs an MNE RAW object from signal data, sampling frequency, and channel names.

        Args:
            signal (numpy array): The EEG signal data as a 2D numpy array (channels x time points).
            fs (float): Sampling frequency of the signal in Hz
            ch_names (BCI2000Tools.Electrodes.ChannelSet): List of channel names corresponding to the signal rows

        Returns:
            mne.io.Raw: An MNE Raw object containing the EEG data
        """
        # Create MNE info object specifying EEG data characteristics
        info = mne.create_info(ch_names=ch_names, sfreq=fs, ch_types="eeg")
        # Initialize Raw object with the signal and info, make sure time-series are in uV
        RAW = mne.io.RawArray(signal, info, verbose=False) # uV
        return RAW


    def make_RAW_stim(self, RAW=None, states=None):
        """
        Adds stimulation channel data to an existing MNE RAW object.

        Args:
            RAW (mne.io.Raw): The original MNE RAW object containing EEG data.
            states (BCI2000Tools.Container.Bunch): Dictionary where keys are stim channel names and values are the data arrays.
        """
        # Retrieve sampling frequency from the original RAW object
        fs = RAW.info['sfreq']
        # Create MNE info object for stimulation channels
        info = mne.create_info(ch_names=[x for x in states.keys()], sfreq=fs, ch_types='stim')
        # Create RawArray for stimulation data
        stim = mne.io.RawArray([x[0] for x in states.values()], info, first_samp=0, verbose=False)
        # Add stim channels to the original RAW object
        RAW.add_channels([stim])


    def make_PREP(self, RAW=None, isSNR=False, isCorrelation=False, isDeviation=False, isHfNoise=False, isNanFlat=False, isRansac=False):
        """
        Identifies noisy channels in an MNE Raw object using various criteria and marks them as bad.

        Args:
            RAW (mne.io.Raw): The original MNE RAW object containing EEG data.RAW: The MNE Raw object to be processed.
            isSNR (bool): If True, identifies bad channels by signal-to-noise ratio.
            isCorrelation (bool): If True, identifies bad channels by correlation.
            isDeviation (bool): If True, identifies bad channels by deviation.
            isHfNoise (bool): If True, identifies bad channels by high-frequency noise.
            isNanFlat (bool): If True, identifies bad channels by NaN or flat signals.
            isRansac (bool): If True, identifies bad channels using RANSAC.
        """
        # Initialize NoisyChannels with the RAW object
        NC = NoisyChannels(RAW, do_detrend=False)

        # Apply different criteria based on the function parameters to find bad channels
        if isSNR: NC.find_bad_by_SNR()
        if isCorrelation: NC.find_bad_by_correlation(correlation_secs=1.0, correlation_threshold=0.4, frac_bad=0.01)
        if isDeviation: NC.find_bad_by_deviation(deviation_threshold=5.0)
        if isHfNoise: NC.find_bad_by_hfnoise(HF_zscore_threshold=5.0)
        if isNanFlat: NC.find_bad_by_nan_flat()

        # Collect names of identified bad channels
        ch_names_bad = []
        bad_dict = NC.get_bads(as_dict=True)
        if isSNR: ch_names_bad += bad_dict['bad_by_SNR']
        if isCorrelation: ch_names_bad += bad_dict['bad_by_correlation']
        if isDeviation: ch_names_bad += bad_dict['bad_by_deviation']
        if isHfNoise: ch_names_bad += bad_dict['bad_by_hf_noise']
        if isNanFlat: ch_names_bad += bad_dict['bad_by_nan'] + bad_dict['bad_by_flat']
        
        # Mark identified bad channels in the RAW object
        RAW.info["bads"].extend(np.unique(ch_names_bad).tolist())
        print(f'Before RANSAC: {RAW.info["bads"]}')

        # RANSAC requires BAD channels by other methods to be identified (optimal use)
        # Re-initialize NoisyChannels with the RAW object and the new bad channels
        NC = NoisyChannels(RAW, do_detrend=False)
        if isRansac: NC.find_bad_by_ransac(n_samples=50, sample_prop=0.25, corr_thresh=0.75, frac_bad=0.4, corr_window_secs=5.0, channel_wise=False, max_chunk_size=None)

        # Collect names of identified bad channels
        bad_dict = NC.get_bads(as_dict=True)
        if isRansac: ch_names_bad += bad_dict['bad_by_ransac']

        # Mark identified bad channels in the RAW object
        RAW.info["bads"] = []
        RAW.info["bads"].extend(np.unique(ch_names_bad).tolist())
        print(f'After RANSAC: {RAW.info["bads"]}')


    def mark_BAD_region(self, RAW=None, block=None):
        """
        Opens an interactive plot for visually identifying and marking bad regions in the EEG data.

        Args:
            RAW (mne.io.Raw): The original MNE RAW object containing EEG data.
            block (bool): Determines if the plot should block execution until closed. If True, execution is halted until the plot is manually closed
        """
        # Initialize an empty Annotations object with a 'BAD_region' label
        annot = mne.Annotations([0], [0], ['BAD_region'])
        # Set the annotations to the RAW object
        RAW.set_annotations(annot)
        # Inform the user that bad regions should now be marked visually
        print(f'\n --> Mark BAD regions (visually)')
        # Open an interactive plot of the RAW data for visual inspection and marking
        RAW.plot(block=block)


    def evaluate_BAD_region(self, RAW=None, label='BAD_region', max_duration=418.):
        """
        Evaluates and summarizes bad regions in the EEG data based on annotations.

        Args:
            RAW (mne.io.Raw): The original MNE RAW object containing EEG data and annotations.
            label (str): The label used to identify regions of interest in the annotations. Defaults to 'BAD_region'.
            max_duration (float): The maximum duration expected for the data, used to calculate the percentage of data marked as bad. Defaults to 418 seconds but it should be the total file time
        """
        # Extract annotations from the RAW object
        annot = RAW.annotations
        # Identify durations of annotations matching the specified label
        bad_regions_id = annot.duration[np.where(annot.description == label)]
        # Print a summary of bad sections, their total duration, and the percentage of the maximum duration
        print(f" --> {label}: {len(bad_regions_id)} sections, ~{round(sum(bad_regions_id),1)} s [{round(sum(bad_regions_id)/max_duration*100,1)}%] --> Bad channels: {RAW.info['bads']}")


    def make_annotation_MI(self, RAW=None, fs=None, nBlocks=None, trialsPerBlock=None, initialSec=None, stimSec=None, taskSec=None, rejectSec=None, nSplit=None, fileTime=None):
        """
        Generate annotations on a mne.io.Raw object for the motor imagery paradigm.

        Args: 
            RAW (mne.io.Raw): The original MNE RAW object containing EEG data.
            fs (float): EEG sampling frequency.
            nBlocks (int): Number of blocks in the MI paradigm.
            trialsPerBlock (int): Number of trials (left, rest after left, right, rest after right) per block.
            initialSec (float): Length of initial gap before the first cue in seconds.
            stimSec (float): Length of cues in seconds.
            taskSec (float): Length of tasks in seconds.
            rejectSec (float): Length of rejected window after the cue in seconds (Used in analysis).
            nSplit (int): Number of windows to consider for each trial, determine the number of splits for the task EEG signal.
            fileTime (float): Length of total file in seconds.

        Returns: 
            mne.io.Raw: The original MNE RAW object with annotations added.
        """

        '''
        # This is a brute force method (takes time as it check for the onset of each StimulusCode), it's the ideal one for variable paradigms
        # Since the paradigm is always the same, we know the target locations of the ticks we need to modify so we use an alternative method (faster)
        # Modify StimulusCode to differentiate resting periods after moving LEFT vs RIGHT
        new_StimulusCode = RAW['StimulusCode'][0][0].copy()
        change=False
        for i in range(len(RAW['StimulusCode'][0][0])):
            if RAW['StimulusCode'][0][0][i]==1: change=False
            if RAW['StimulusCode'][0][0][i]==2: change=True
            if RAW['StimulusCode'][0][0][i]==3 and change: new_StimulusCode[i]=4
        RAW['StimulusCode'][0][0] = new_StimulusCode
        '''

        # Convert times in seconds to number of samples with sampling frequency (note: the words Samples and Ticks are used interchangeably in this function)
        initialTicks = int(initialSec * fs) # initial length in Ticks
        stimTicks = int(stimSec * fs) # stimulus length in Ticks (cue)
        taskTicks = int(taskSec * fs) # task length in Ticks (performance)
        rejectTicks = int(rejectSec * fs) # rejection length in Ticks (after cue, subtracted from performance)

        # This is the alternative method (The faster method starts here)
        new_StimulusCode = RAW['StimulusCode'][0][0].copy()

        def change_StimulusCode(list_=None, nBlocks=None, trialsPerBlock=None, initialTicks=None, stimTicks=None, taskTicks=None, posInBlock=None, newStimCode=None):
            """
            Changes the StimulusCode associated to the rest-after-right trials to differentiate them from the rest-after-left trials.

            Args: 
                list_ (numpy array): Array with StimulusCodes from mne.io.Raw.
                nBlocks (int): Number of blocks in the MI paradigm.
                trialsPerBlock (int): Number of trials (left, rest after left, right, rest after right) per block.
                initialTicks (int): Number of samples in initial gap before the first cue.
                stimTicks (int): Number of samples in cues.
                taskTicks (int): Number of samples in tasks.
                posInBlock (int): Position in a block of the trial whose StimulusCode will be changed.
                newStimCode (int): New StimulusCode to use. 

            Returns: 
                None
            """

            # Change the StimulsCode in specific locations
            for i in range(nBlocks):
                # Determines the starting samples
                start = initialTicks + (stimTicks + taskTicks) * (posInBlock-1 + trialsPerBlock*i)
                # Determines the ending samples
                end = start + stimTicks
                try: 
                    # Overwrite the StimulusCode
                    list_[start:end] = [newStimCode]*stimTicks
                except ValueError:
                    break

        # Changing rest-after-right trials (4th position with code 3) to newStimCode 4 to differentiate them from rest-after-left trials
        change_StimulusCode(new_StimulusCode, nBlocks, trialsPerBlock, initialTicks, stimTicks, taskTicks, posInBlock=4, newStimCode=4)

        # Plot the StimulusCode before and after the change 
        fig, (ax,ax1) = plt.subplots(nrows=1, ncols=2, figsize=(12,4))
        ax.plot(new_StimulusCode, label='new_StimulusCode')
        ax.plot(RAW['StimulusCode'][0][0], label='StimulusCode')
        ax.plot(RAW['StimulusBegin'][0][0], label='StimulusBegin')
        ax.legend()
        ax.set_xlabel('Ticks')
        ax1.plot(new_StimulusCode, label='new_StimulusCode')
        ax1.plot(RAW['StimulusCode'][0][0], label='StimulusCode')
        ax1.plot(RAW['StimulusBegin'][0][0], label='StimulusBegin')
        for i in range(1,19):
            ax1.axvline(i*fs, lw=1, ls='--', color='grey', alpha=0.5)
        ax1.set_xlim(0,54*fs)
        ax1.set_xlabel('Ticks')
        plt.legend()
        ax2 = ax1.twiny()
        ax2.plot(new_StimulusCode[::int(fs)], label='new_StimulusCode', alpha=0)
        ax2.set_xlabel('Time [s]', color='red')  # Set the label for the second x-axis
        ax2.set_xlim(0,54)
        ax2.tick_params(axis='x', labelcolor='red') 
        plt.show()

        # Print information regarding the MI paradigm
        print(f'\n~~~~~~~~ ANNOTATIONS ~~~~~~~~')
        print(f'File length: {len(new_StimulusCode)/fs} s or {len(new_StimulusCode)} ticks')
        print(f'File length (remove initial pause): {(len(new_StimulusCode)-initialTicks)/fs} s or {len(new_StimulusCode)-initialTicks} ticks')
        print(f'Block length: {(len(new_StimulusCode)-initialTicks)/fs/nBlocks} s or {(len(new_StimulusCode)-initialTicks)/nBlocks} ticks')
        print(f'Command length: {(len(new_StimulusCode)-initialTicks)/fs/nBlocks/trialsPerBlock} s or {(len(new_StimulusCode)-initialTicks)/nBlocks/trialsPerBlock} ticks')
        print(f'\tStim length: {(stimTicks)/fs} s or {stimTicks} ticks')
        print(f'\tTask length: {(taskTicks)/fs} s or {taskTicks} ticks')

        def expand_onset(onset=None, nSplit=None, taskSec=None, rejectSec=None):
            """
            Create a list of onset timing for the annotations of the epochs withing a specific trial, given the initial onset of the trial

            Args:
                onset (float): Time at which the task section of a trial starts.
                nSplit (int): Number of splits to use for the task window.
                taskSec (float): Length of task is seconds.
                rejectSec (float): Length of reject after cue in seconds.

            Returns:
                list: contains the unique onset timing for annotations on the mne.io.Raw
            """
            if nSplit > 1: 
                for x in onset: 
                    expand = np.linspace(x, x+(taskSec-rejectSec), nSplit+1)
                    onset = onset + list(expand[:-1])
            return list(np.unique(onset))

        def generate_onsets(start=None, end=None, stimSec=None, taskSec=None, trialsPerBlock=None, nSplit=None, rejectSec=None):
            """
            Create a list of onset timing and durations for the annotations of the epochs withing a specific trial
            
            Args:
                start (float): Time start of a specific trial.
                end (float): Time end of that specific trial. Generally fileTime which identifies the end of the file (no more trials after that).
                stimSec (float): Length of cues in seconds.
                taskSec (float): Length of tasks in seconds.
                trialsPerBlock (int): Number of trials (left, rest after left, right, rest after right) per block.
                nSplit (int): Number of windows to consider for each trial, determine the number of splits for the task EEG signal.
                rejectSec (float): Length of rejected window after the cue in seconds (Used in analysis).
    
            Returns:
                (list, list): list of onsets, list of duration of each epoch
            """
            onset = [x+rejectSec for x in np.arange(start, end, (stimSec+taskSec)*trialsPerBlock)] # in seconds [5,57,109,161,...] without rejectSec for example
            onset = expand_onset(onset, nSplit, taskSec, rejectSec)
            duration = [(taskSec-rejectSec)/nSplit for x in np.arange(len(onset))] # in seconds
            return onset, duration

        # annotate left hand task
        onset1, duration1 = generate_onsets(initialSec + taskSec*0 + stimSec*1, fileTime, stimSec, taskSec, trialsPerBlock, nSplit, rejectSec) # in seconds [5,57,109,161,...] without rejectSec
        description1 = ['left' for x in onset1]
        #print(len(onset1), len(duration1), len(description1))

        # annotate right hand task
        onset2, duration2 = generate_onsets(initialSec + taskSec*2 + stimSec*3, fileTime, stimSec, taskSec, trialsPerBlock, nSplit, rejectSec) # in seconds [31,83,135,187,...] without rejectSec
        description2 = ['right' for x in onset2]
        #print(len(onset2), len(duration2), len(description2))

        # annotate left hand rest
        onset3, duration3 = generate_onsets(initialSec + taskSec*1 + stimSec*2, fileTime, stimSec, taskSec, trialsPerBlock, nSplit, rejectSec) # in seconds [18,70,122,174,...] without rejectSec
        description3 = ['left_rest' for x in onset3]
        #print(len(onset3), len(duration3), len(description3))

        # annotate right hand rest
        onset4, duration4 = generate_onsets(initialSec + taskSec*3 + stimSec*4, fileTime, stimSec, taskSec, trialsPerBlock, nSplit, rejectSec) # in seconds [44,96,148,200,...] without rejectSec
        description4 = ['right_rest' for x in onset4]
        #print(len(onset4), len(duration4), len(description4))

        # annotate cue regions
        onset5 = [x for x in np.arange(initialSec, fileTime, stimSec+taskSec)] # in seconds [2,15,28,41,...]
        duration5 = [stimSec + rejectSec for x in onset5]
        description5 = ['cue']*len(duration5)
        #print(len(onset5), len(duration5), len(description5))

        # annotate BAD regions
        # because there are already existing 'BAD_region' annotations
        annot = RAW.annotations
        annot.duration[np.where(annot.description=='BAD_region')]
        onset6 = [0] + list(annot.onset[np.where(annot.description=='BAD_region')]) # in seconds
        duration6 = [initialSec] + list(annot.duration[np.where(annot.description=='BAD_region')])
        description6 = ['BAD_region']*len(duration6) # initial region
        #print(len(onset6), len(duration6), len(description6))

        # Combine all lists 
        onset = onset1 + onset2 + onset3 + onset4 + onset5 + onset6
        duration = duration1 + duration2 + duration3 + duration4 + duration5 + duration6
        description = description1 + description2 + description3 + description4 + description5 + description6

        # Generate information for specific epochs within each trial (left_1, left_2, ...)
        for ith in range(nBlocks):
            onset += onset1[nSplit*ith:nSplit*(ith+1)]
            duration += duration1[nSplit*ith:nSplit*(ith+1)]
            description += [f'left_{int(ith+1)}']*len(onset1[nSplit*ith:nSplit*(ith+1)])
            
            onset += onset2[nSplit*ith:nSplit*(ith+1)]
            duration += duration2[nSplit*ith:nSplit*(ith+1)]
            description += [f'right_{int(ith+1)}']*len(onset2[nSplit*ith:nSplit*(ith+1)])
            
            onset += onset3[nSplit*ith:nSplit*(ith+1)]
            duration += duration3[nSplit*ith:nSplit*(ith+1)]
            description += [f'left_rest_{int(ith+1)}']*len(onset3[nSplit*ith:nSplit*(ith+1)])
            
            onset += onset4[nSplit*ith:nSplit*(ith+1)]
            duration += duration4[nSplit*ith:nSplit*(ith+1)]
            description += [f'right_rest_{int(ith+1)}']*len(onset4[nSplit*ith:nSplit*(ith+1)])

        # Set annotation to mne.io.Raw
        my_annot = mne.Annotations(onset=onset, duration=duration, description=description)
        RAW.set_annotations(my_annot)
        #print(RAW.annotations)

        # Plot annotations using MNE
        events_from_annot, event_dict = mne.events_from_annotations(RAW)
        fig = mne.viz.plot_events(events_from_annot, sfreq=fs, first_samp=RAW.first_samp, event_id=event_dict)
        fig.subplots_adjust(right=0.7)
        plt.show()

        return RAW
    

    def make_montage(self, montage_type=None, ch_to_show=None, conv_dict=None, verbose=False):
        """
        Creates and plots a montage for EEG data based on the specified montage type and channels.

        Args:
            montage_type: The type of montage to create. Supported types  []'DSI_24', 'GTEC_32', 'EGI_128']
            ch_to_show: List of channels to be displayed in the montage plot.
            conv_dict: Optional dictionary for converting channel names from the montage names to custom names.

        Returns:
            montage: The MNE Montage object created based on the specified parameters.
        """
        # Select the appropriate standard montage based on the specified type
        if montage_type in ['DSI_24', 'GTEC_32']: 
            montage = mne.channels.make_standard_montage('standard_1020')
        elif montage_type == 'EGI_128': 
            montage = mne.channels.make_standard_montage('GSN-HydroCel-129')
        
        idx = []
        # Determine indices for the channels to show, using conversion dictionary if provided
        for ch in ch_to_show:
            if montage_type == 'EGI_128': idx.append(montage.ch_names.index(conv_dict[ch]))
            else: idx.append(montage.ch_names.index(ch))
        
        # Update montage to only include specified channels
        montage.ch_names = ch_to_show
        montage.dig = montage.dig[0:3] + [montage.dig[x + 3] for x in idx]
        # Plot the montage
        if verbose: 
            montage.plot()
        return montage


    def find_ch_central(self, ch_location=None, ch_list=None):
        """
        Identifies central electrodes from a list of electrode locations.

        Args:
            ch_location (list of tuples): List of tuples with electrode names and their X, Y coordinates.
            ch_list (list): Optional list of electrode names to filter through.

        Returns:
            list: A list of names of central electrodes (those with an X coordinate = 0) 
        """
        # Default to using all electrodes if no specific list is provided
        if not ch_list: 
            ch_list = [x[0] for x in ch_location]
        # Return names of electrodes that are central and in the target list
        return [x[0] for x in ch_location if (x[1]==0 and x[0] in ch_list)]


    def find_ch_left(self, ch_location=None, ch_list=None):
        """
        Identifies left-sided electrodes from a list of electrode locations.

        Args:
            ch_location (list of tuples): List of tuples with electrode names and their X, Y coordinates.
            ch_list (list): Optional list of electrode names to filter through.

        Returns:
            list: A list of names of left-sided electrodes (those with an X coordinate less than 0) 
        """
        # Use all electrodes if no specific list is provided
        if not ch_list: 
            ch_list = [x[0] for x in ch_location]
        # Return names of electrodes that are on the left side and in the target list
        return [x[0] for x in ch_location if (x[1]<0 and x[0] in ch_list)]


    def find_ch_right(self, ch_location=None, ch_list=None):
        """
        Identifies right-sided electrodes from a list of electrode locations.

        Args:
            ch_location (list of tuples): List detailing each electrode's name and its X, Y coordinates.
            ch_list (list): Optional. A list of specific electrode names to consider. If not provided, all electrodes are considered.

        Returns:
            list: Names of electrodes located on the right side (X coordinate greater than 0),
        """
        # Use all electrodes if no specific list is provided
        if not ch_list: 
            ch_list = [x[0] for x in ch_location]
        # Return the names of right-sided electrodes from the targeted list
        return [x[0] for x in ch_location if (x[1]>0 and x[0] in ch_list)]


    def find_ch_circle(self, ch_location=None, ch_list=None, radius=None):
        """
        Identifies electrodes within a specified circle radius.

        Args:
            ch_location (list of tuples): Electrode details with each tuple containing the electrode's name and X, Y coordinates.
            ch_list (list): Optional. Specific electrodes to consider. If not provided, all electrodes are considered.
            radius (float): The radius of the circle within which to find electrodes.

        Returns:
            list: Names of electrodes located within the specified circle radius,
        """
        # Default to using all electrodes if no specific list is provided
        if not ch_list: 
            ch_list = [x[0] for x in ch_location]
        circle = []  # Initialize list to hold electrodes within the circle
        # Iterate through each electrode to check if it falls within the specified radius
        for x in ch_location:
            if x[0] in ch_list:
                # Check if the electrode is within the circle using the Pythagorean theorem
                if np.sqrt(x[1]**2 + x[2]**2) <= radius: 
                    circle.append(x[0])  # Add electrode name to the list if within the circle
        return circle


    def low(self, x=None):
        """
        Convert the input string or list or numpy array to its lowercase version of itself.

        Args:
        - input_string (str or list or numpy array): The string or list or numpy array of strings to be converted to lowercase.

        Returns:
        - str or list or numpy array: The lowercase version of the input.
        """
        # Convert the string to lowercase using the .lower() method
        if type(x)==str: x = x.lower()
        elif type(x)==list: x = [y.lower() for y in x]
        elif type(x)==np.ndarray: x = np.array([y.lower() for y in x])
        return x


    def find_ch_symmetry(self, ch_location=None, ch_list=None):
        """
        Identifies pairs of electrodes that are symmetric about the Y-axis.

        Args:
            ch_location (list of tuples): Each tuple contains an electrode's name and its X, Y coordinates.
            ch_list (list): Optional. A list of specific electrodes to consider for symmetry pairing. 
                            If not provided, all electrodes are considered.

        Returns:
            dict: A dictionary where each key-value pair represents a pair of electrode names 
        """
        # Default to using all electrodes if no specific list is provided
        if not ch_list: 
            ch_list = [x[0] for x in ch_location]
        symmetry = {}  # Initialize dictionary to hold symmetric electrode pairs
        # Check each electrode against all others for symmetry
        for x in ch_location:
            if x[0].lower() in [ch.lower() for ch in ch_list]:
                ch1, x1, y1 = x  # Current electrode and its coordinates
                for y in ch_location:
                    ch2, x2, y2 = y  # Potential symmetric electrode and its coordinates
                    # Check for symmetry about the Y-axis
                    if x1 == -x2 and y1 == y2:
                        symmetry[ch1] = ch2  # Record symmetric pair
        return symmetry


    def interpolate(self, RAW=None, reset_bads=True):
        """
        Interpolates bad channels in an M/EEG recording. RAW must have a set montage.

        Args:
            RAW: The Raw object containing the data and channel information.
            reset_bads (bool): If True, clears the list of bad channels after interpolation. 
                               If False, the list of bad channels remains unchanged.

        Returns:
            list: The list of channels that were marked as bad before interpolation.
        """
        # Display the bad channels to be interpolated
        print(f"BAD CHANNELS to be interpolated: {RAW.info['bads']}")
        old_ch_bad = RAW.info['bads']  # Store the current list of bad channels

        # Perform interpolation, with the option to reset the list of bad channels
        RAW.interpolate_bads(reset_bads=reset_bads)

        # Provide feedback based on the reset_bads argument
        if reset_bads: 
            print(f"RAW.info['bads'] have been modified")
        else:
            print(f"RAW.info['bads'] have not been modified")

        return old_ch_bad  # Return the original list of bad channels


    def make_epochs(self, RAW=None, tmin=None, tmax=None, event_id=None, events_from_annot=None, verbose=False):
        """
        Segments the Raw M/EEG data into epochs based on specified event identifiers.

        Args:
            RAW: The Raw object containing the data to be epoched.
            tmin (float): Start time before event onset (in seconds).
            tmax (float): End time after event onset (in seconds).
            event_id (int): The ID of the event around which epochs are created.
            events_from_annot (numpy array): Events array extracted from annotations in the Raw data.
            verbose (bool): If True, prints detailed information about the epoching process.

        Returns:
            mne.Epochs: The epochs created from the Raw data for the specified event ID.
        """
        expected_epochs_per_type = sum([1 for x in events_from_annot if x[2]==event_id])  # Calculate expected epochs for the event type
        if verbose: 
            print(f"Expected {expected_epochs_per_type} epochs {tmax-tmin}s-long (per type)")
        
        # Create epochs from RAW data based on the specified event ID and time window
        epochs_ = mne.Epochs(RAW, events_from_annot, event_id=event_id, tmin=tmin, tmax=tmax, baseline=None, preload=True, verbose=False)
        # Note: Index 0 in shape is the number of epochs, index 1 is channel numbers, index 2 is EEG values at segment time t
        # Calculate the actual number of epochs created
        n_epochs = epochs_.get_data(picks='eeg').shape[0]
        if verbose: 
            print(f"Summary: {n_epochs}/{expected_epochs_per_type} total epochs")

        return epochs_


    def make_psd(self, epochs=None, fs=None, resolution=None, tmin=None, tmax=None, fmin=None, fmax=None, nPerSegment=None, nOverlap=None, aggregate=True, verbose=False):
        """
        Computes the Power Spectral Density (PSD) of given epochs using the Welch method.

        Args:
            epochs: The epochs from which to compute the PSD.
            fs (float): The sampling frequency of the data.
            resolution (float): The frequency resolution of the PSD (in Hz).
            tmin (float): Start time (in seconds) to consider for the PSD computation.
            tmax (float): End time (in seconds) to consider for the PSD computation.
            fmin (float): Minimum frequency (in Hz) to include in the PSD.
            fmax (float): Maximum frequency (in Hz) to include in the PSD.
            nPerSegment (int): Number of data points per segment used in the Welch method.
            nOverlap (int): Number of points of overlap between segments.
            aggregate (bool): If True, aggregates the PSD across all epochs and segments (segments first).
            verbose (bool): If True, prints additional information about the PSD computation process.

        Returns:
            ndarray: The computed PSD values.
        """
        nfft = int(fs / resolution)  # Number of FFT points
        effective_window = (1 / resolution)  # Effective window length in seconds
        expected_segments = int(((tmax - tmin) - effective_window) / (effective_window - nOverlap / fs)) + 1  # Expected number of segments
        expected_bins = int((fmax - fmin) / resolution + 1)  # Expected number of frequency bins

        # Compute PSD using Welch's method
        psd_ = epochs.compute_psd(method='welch', 
                                  fmin=fmin, fmax=fmax, 
                                  tmin=tmin, tmax=tmax,
                                  n_fft=nfft, 
                                  n_overlap=nOverlap,
                                  n_per_seg=nPerSegment, 
                                  average=None,
                                  window='hann', 
                                  output='power', 
                                  verbose=verbose).get_data()

        # Verbose output for PSD computation details
        if verbose: 
            print(f"Expected w/ resolution {resolution} [Hz/bin]: ")
            print(f"  - Eff_Window Length [s]: {effective_window}")
            print(f"  - Epochs: {epochs.get_data(picks='eeg').shape[0]} -> {tmax-tmin} s-long (per type)")
            print(f"  - Channels: {epochs.get_data(picks='eeg').shape[1]}")
            print(f"  - Bins: {expected_bins}")
            print(f"  - Expected segments (or periodograms): {expected_segments}")
            print(f"Dimension check: (epoch, ch, bins, segments/periodograms) = {psd_.shape}")
        
        # Aggregate PSD values if requested
        if aggregate: 
            if len(psd_.shape) > 3:
                psd_ = np.mean(psd_, axis=3)  # Average across segments/periodograms
                if verbose: print(f"Aggregate segments/periodograms: (epoch, ch, bins) = {psd_.shape}")
            psd_ = np.mean(psd_, axis=0)  # Average across epochs
            if verbose: print(f"Aggregate epoch-wise: (ch, bins) = {psd_.shape}")
            
        return psd_


    def convert_dB(self, X=None):
        """
        Converts power values from microvolts squared (uV²) to decibels (dB).
        Steps: 
            Converts X from uV² to V²
            Converts to dB using 1V as reference

        Args:
            X: Array of power values in uV².

        Returns:
            Array of power values converted to dB, using 1V² as the reference power level.
        """
        return 10 * np.log10(X * 1e12)


class Stats:
    """
    A class for performing permutation and bootstrap tests on PSDs which are transformed into eta-square values at each electrode.

    This class provides methods to perform statistical tests for motor imagery starting from PSDs.
    An object is built which is able to handle contralateral and ipsilateral electrodes based on the test being performed. 
    Also the frequency band of interest is specified at each object initialization.

    Attributes:
        ch_set (BCI2000Tools.Electrodes.ChannelSet): An object representing the set of channels/electrodes, from BCI2000Tools.Electrodes
        dict_symm (dict): A dictionary containing electrodes names and their symmetric about the X=0 axis in the XY plane
        isContralat (list): List of electrodes in the contralateral hemisphere. The corresponding ipsilateral electrodes are found.
        bins (numpy array): Bins in frequency space. 
        custom_bins (str or list): Bins to consider for a specific frequency band. 
                                   If str: it identifies one of the target frequency bands: theta, alpha, beta
                                   If list: Indices of the first and last bins to consider in a custom frequency band (e.g. [6,10])
        transf (str): Transformation to apply to the PSDs. 

    Methods:
        CalculateEtas2
        Dummy
        CalculateR
        CalculateR2
        Transform
        DifferenceOfSumsR2
        DifferenceOfR2
        FindSymmetric
        Shuffle
        ApproxPermutationTest
        BootstrapTest
        BootstrapResample
        pvalue_interval
        negP
    """

    def __init__(self, ch_set=None, dict_symm=None, isContralat=None, bins=None, custom_bins=None, transf='r2'):
        self.ch_set = ch_set
        self.ch_names = np.array(self.ch_set.get_labels())
        self.dict_symm = dict_symm
        self.isContralat = isContralat
        self.bins = bins
        self.transf = transf

        self.theta_ticks = np.where((self.bins>=4)  & (self.bins<=7 ))[0]
        self.alpha_ticks = np.where((self.bins>=8)  & (self.bins<=12))[0]
        self.beta_ticks  = np.where((self.bins>=13) & (self.bins<=30))[0]
 
        if custom_bins == 'theta': self.custom_ticks = self.theta_ticks
        if custom_bins == 'alpha': self.custom_ticks = self.alpha_ticks
        if custom_bins == 'beta' : self.custom_ticks = self.beta_ticks
        if custom_bins not in ['theta', 'alpha', 'beta']: self.custom_ticks = np.where((self.bins>=custom_bins[0]) & (self.bins<=custom_bins[-1]))[0]

    def CalculateEtas2(self, x=None, isTreatment=None, signed=True):
        """
        Calculates Eta squared (η²), a measure of effect size, for comparing variances between and within groups.
        The ratio between the variance between groups and the total variance is commonly referred to as the "Eta squared (η²)" in statistics. 
        It is a measure of effect size for use in ANOVA (Analysis of Variance) and represents the proportion of the total variance in the dependent variable that is attributable to the variance between groups.
        Here's a breakdown of its components:
            Between-group variance (SSB): Variance due to the difference between the means of the groups.
            Total variance (SST): The sum of the between-group variance (SSB) and within-group variance (SSW).
        Eta squared is a useful measure for understanding the proportion of the overall variance that is explained by the grouping variable, 
        indicating the effect size of the group differences on the dependent variable.
        η² = SSB/SST

        Args:
            x (numpy array): A 3D array with dimensions corresponding to trials, channels, and frequency bins.
            isTreatment (numpy array): A boolean array where True indicates samples belonging to the treatment group and False to the control group
            signed (bool): If True, the calculated η² values are signed based on the direction of the effect (positive if the mean of the treatment group is higher than the control group, and negative otherwise).

        Returns:
            numpy array: The signed Eta squared values for each channel and frequency bin
        """
        # Calculate sample sizes for treatment and control groups
        n1 = np.sum(isTreatment)
        n2 = np.sum(~isTreatment)

        # Separate the data into treatment and control groups
        x1 = x[isTreatment]
        x2 = x[~isTreatment]

        # Compute mean values for each group and the grand mean (aggregate trials)
        mu1 = np.mean(x1, axis=0)
        mu2 = np.mean(x2, axis=0)
        grand_mean = np.mean(x, axis=0)

        # Sum of squares within groups
        ssw = np.sum((x1 - mu1)**2, axis=0) + np.sum((x2 - mu2)**2, axis=0)
        # Sum of squares between groups
        ssb = n1*(mu1 - grand_mean)**2 + n2*(mu2 - grand_mean)**2

        # Compute Eta squared, with an option to sign the result
        eta_squared = ssb / (ssb + ssw)
        if not signed:
            return eta_squared
        else:
            signs = np.where(mu1 - mu2 > 0, -1, 1)  # Determine the direction of the effect
            return eta_squared * signs  # Return signed Eta squared values


    def Dummy(self, y=None):
        """
        Converts a categorical variable to a numerical variable with a dummy trick

        Args:
            y (numpy array): Contains True or False for two classes

        Returns: 
            numpy array: Numerical variable with 0 and 1
        """
        return np.where(y==False, 1, 0)


    def CalculateR(self, x=None, isTreatment=None):
        """
        Calculates the Pearson correlation coefficient (r) between two numerical variables.

        Args:
            x (numpy array): Continuous data with shape (trial, ch)
            isTreatment (numpy array): A binary array where True indicates samples belonging to one group (e.g., treatment) and False to the other (e.g., control).

        Returns:
            numpy array: The Pearson correlation coefficient (r).
        """
        # Initialize the correlation matrix
        r = np.zeros(x.shape[1])
        # Transform the categorical variable into a dummy variable, with task (False) getting higher label
        y = self.Dummy(isTreatment)
        # Calculate the means of x (aggregate trials) and y
        x_mean = np.mean(x, axis=0)
        y_mean = np.mean(y)
        # Calculate the numerator of the correlation coefficient
        numerator = np.sum((x - x_mean) * (y - y_mean)[:, np.newaxis], axis=0)
        # Calculate the denominator of the correlation coefficient
        x_diff_sq = np.sum((x - x_mean) ** 2, axis=0)
        y_diff_sq = np.sum((y - y_mean) ** 2)
        denominator = np.sqrt(x_diff_sq * y_diff_sq)
        # Compute the correlation coefficient
        r = numerator / denominator
        return r


    def CalculateR2(self, x=None, isTreatment=None, signed=True):
        """
        Calculates the squared Pearson correlation coefficient (R^2) between a continuous
        variable and a binary categorical variable, which is used as a measure of effect size.

        Args:
            x (numpy array): It represents the continuous data with shape (trial, ch)
            isTreatment (numpy array): A binary array where True indicates samples belonging to one group (e.g., treatment) and False to the other (e.g., control).
            signed (bool): If True, the result is signed squared R to indicate the direction of the association.
                           If False, the result is the unsigned R^2, which only indicates the strength of the association.

        Returns:
            numpy array: The squared Pearson correlation coefficient (R^2) for each channel
                         and frequency bin. This is optionally signed to reflect the direction of the association.
        """
        # Compute the correlation coefficient
        r = self.CalculateR(x, isTreatment)
        # Return signed or unsigned R^2 based on the signed argument
        if signed:
            return r * abs(r)  # Signed R^2
        else:
            return r * r  # Unsigned R^2


    def Transform(self, x=None, isTreatment=None):
        """
        Applies a specified transformation to the data based on the set transformation type.

        Args:
            x: The data array to be transformed with shape (trial, ch)
            isTreatment: A boolean array indicating treatment group membership for each element in `x`.

        Returns:
            The result of the transformation applied to the data.
            This function supports two types of transformations:
                - 'eta2': Computes the eta squared (η²) statistic, a measure of effect size for the difference between groups.
                - 'r2': Computes the squared Pearson correlation coefficient (R²) to quantify the strength of association.
        """
        if self.transf == 'eta2':
            return self.CalculateEtas2(x=x, isTreatment=isTreatment)
        elif self.transf == 'r2':
            return self.CalculateR2(x=x, isTreatment=isTreatment)


    def DifferenceOfSumsR2(self, x=None, isTreatment=None):
        """
        Calculates the difference between the sums of transformed data for contralateral and ipsilateral electrodes.

        Args:
            x: Data array with shape (trial, ch, bin)
            isTreatment: Boolean array indicating treatment group membership for each element in `x`.

        Returns:
            The difference between the sum of the transformed data for contralateral electrodes and
            the sum for ipsilateral electrodes, based on the specified frequency bins.
        """
        # Average the data (PSDs) within the specified frequency bins (trial, ch, bin) -> (trial, ch)
        x = np.mean(x[:, :, self.custom_ticks[0]:self.custom_ticks[-1]], axis=2)
        # Transform PSDs to dB
        x = 10 * np.log(x * 1e12)
        # Apply specified transformation (e.g., 'eta2' or 'r2') to the data (trial, ch) -> (ch,)
        x = self.Transform(x, isTreatment)
        # Sum the averages for contralateral electrodes
        x1 = x[self.isContralat]
        # Identify and sum the averages for ipsilateral electrodes
        isIpsilat = self.FindSymmetric(isContralat=self.isContralat)
        x2 = x[isIpsilat]
        # Compute and return the difference between the sums for contralateral and ipsilateral electrodes
        return np.sum(x1) - np.sum(x2)


    def DifferenceOfR2(self, x=None, isTreatment=None):
        """
        Calculates the difference in R² values between contralateral and ipsilateral electrodes for topoplot visualization.

        Args:
            x: Data array with shape (trial, ch, bin)
            isTreatment: Boolean array indicating treatment group membership for each element in `x`.

        Returns:
            A numpy array of R² differences suitable for plotting on a topoplot, emphasizing hemispheric differences in brain activity.

        Note:
            - The function assumes that electrode symmetry and hemisphere information are properly defined in `self.isContralat` and
              that methods exist for transforming data (`Transform`), finding symmetric electrodes (`FindSymmetric`), and identifying
              electrode indices (`ch_set.find_labels` and `ch_set.get_labels`).
            - The output array `r2` is initialized to zeros and filled with the computed R² differences for electrodes identified as
              contralateral, with the ipsilateral differences being directly subtracted.
        """
        # Average the data (PSDs) within the specified frequency bins (trial, ch, bin) -> (trial, ch)
        x = np.mean(x[:, :, self.custom_ticks[0]:self.custom_ticks[-1]], axis=2)
        # Transform PSDs to dB
        x = 10 * np.log(x * 1e12)
        # Apply specified transformation (e.g., 'eta2' or 'r2') to the data (trial, ch) -> (ch,)
        x = self.Transform(x, isTreatment)
        # Sum the averages for contralateral electrodes
        x1 = x[self.isContralat]
        # Identify and sum the averages for ipsilateral electrodes
        isIpsilat = self.FindSymmetric(isContralat=self.isContralat)
        x2 = x[isIpsilat]
        # Initialize R² difference array
        r2 = np.zeros(len(self.ch_names))
        # Compute R² differences for matched contralateral and ipsilateral electrodes
        for i, j in zip(self.ch_set.find_labels(np.array(self.ch_set.get_labels())[self.isContralat]), range(len(self.isContralat))):
            r2[i] = x1[j] - x2[j]
        # Return the transformed data
        return x


    def FindSymmetric(self, isContralat=None):
        """
        Identifies symmetric electrode indices in the opposite hemisphere

        Args:
            isContralat: A boolean array indicating electrode positions. True for target electrodes, False otherwise.

        Returns:
            A list of indices corresponding to the symmetric electrodes in the opposite hemisphere.
        """
        ch_symm = []
        # Iterate over target electrodes to find their symmetric counterparts
        for ch in self.ch_names[isContralat]:
            # Append the symmetric channel name based on predefined mapping
            ch_symm.append(self.dict_symm[ch])
        # Convert symmetric channel names to indices for data analysis
        return self.ch_set.find_labels(ch_symm)


    def Shuffle(self, a=None):
        """
        Randomly shuffles the elements of the array `a` in place and returns a reference to the shuffled array.
        
        Args:
            a (numpy array): The array to be shuffled. The shuffling is performed in place, affecting the original array.

        Returns:
            numpy array: A reference to the shuffled array (note that the original array `a` is modified in place).
        """
        np.random.shuffle(a)
        return a


    def ApproxPermutationTest(self, x=None, isTreatment=None, stat=None, nSimulations=1999, plot=False):
        """
        One-sided two-sample approximate permutation test assuming the value of `stat(x,isTreatment)` is expected to be larger under H1 than under H0.

        We call this an "approximate" permutation test because an actual exact permutation test would test *every* permutation exhaustively,
        whereas this one approximates the same distribution by repeated random label reshuffling.

        Note that permutation tests potentially suffer from the Behren's-Fisher problem: a difference-of-means permutation test will perform 
        similarly to a naive (uncorrected) t-test in that regard. To fix this, use `BootstrapTest()` instead.

        Args:
            x: Data array.
            isTreatment: Boolean array indicating treatment group membership.
            stat: Test statistic function 
            nSimulations (int): Number of random permutations for approximating the distribution.
            plot (bool): If True, plots the histogram of the permuted statistics with the observed statistic marked.

        Returns:
            float: P-value estimating the probability of observing the given or more extreme statistic under the null hypothesis.

        This method approximates the distribution of the test statistic under the null hypothesis by randomly shuffling group labels.
        It is termed "approximate" due to relying on a subset of all possible permutations. The function calculates the observed test statistic,
        performs `nSimulations` permutations of the treatment labels, calculates the test statistic for each permutation, and optionally plots
        the distribution of permuted statistics with the observed value. The p-value is calculated as the proportion of permuted statistics
        that are equal to or more extreme than the observed statistic, adjusted for continuity.
        """
        isTreatment = isTreatment.copy()  # Copy to avoid modifying original
        observed = stat(x, isTreatment)  # Calculate observed statistic
        # Perform permutations and calculate p-value
        hist = [stat(x, self.Shuffle(isTreatment)) for _ in range(nSimulations)]
        if plot:  # Optionally plot the distribution of permuted statistics
            plt.hist(hist)
            plt.axvline(observed, color='black')
            plt.show()
        nReached = sum(np.array(hist) >= observed)
        return (0.5 + nReached) / (1.0 + nSimulations)


    def BootstrapTest(self, x=None, isTreatment=None, stat=None, nSimulations=1999, nullHypothesisStatValue=0.0, plot=False):
        """
        Efron & Tibshirani page 215, equation (15.32)

        Again this is equivalent to a one-sided two-sample test and again,
        we assume the value of `stat(x,isTreatment)` is expected to be *larger* under H1 than under H0. However, the math ends up being
        rearranged somewhat to perform the test, so we'll need to specify explicitly the `stat() value that we expect under the
        null hypothesis (and we will be counting the simulation results that go *below* it---however, don't be deceived by this: the
        situation is still the same as in the other tests, in the sense that a bigger effect still means a higher `stat()` value).

        Bootstrap tests avoid the Behren's-Fisher problem that you get with permutation tests: a difference-of-means bootstrap test will perform similarly to a t-test with Welch's correction.

        Args:
            x: Data array.
            isTreatment: Boolean array indicating treatment group membership.
            stat: Test statistic function 
            nSimulations (int): Number of bootstrap samples to generate.
            nullHypothesisStatValue (float): Expected value of the test statistic under the null hypothesis.
            plot (bool): If True, plots the histogram of the bootstrap statistics with observed and null hypothesis values marked.

        Returns:
            float: P-value estimating the probability of observing a test statistic as extreme as or more extreme than the null hypothesis value.
        """
        hist = [stat(self.BootstrapResample(x, isTreatment), isTreatment) for _ in range(nSimulations)]
        if plot:  # Optionally plot the distribution of bootstrap statistics
            plt.hist(hist)
            plt.axvline(nullHypothesisStatValue, color='red')  # Null hypothesis value
            plt.axvline(stat(x, isTreatment), color='black')  # Observed statistic value
            plt.show()
        # Calculate p-value
        nReached = sum(np.array(hist) < nullHypothesisStatValue)
        return (0.5 + nReached) / (1.0 + nSimulations)


    def BootstrapResample(self, a=None, isTreatment=None):
        """
        Performs bootstrap resampling on the array `a`.

        Args:
            a: The array to be resampled. Can be multidimensional.
            isTreatment: An optional boolean array indicating treatment group membership. If provided, resampling is performed separately within each group.

        Returns:
            A resampled array with the same shape as `a`. If `isTreatment` is provided, each group
            defined by `isTreatment` is resampled independently, preserving group sizes.
        """
        if isTreatment is not None:
            isTreatment = isTreatment.ravel()
            # This part only works if a.shape[1] doesn't exist
            #a = a.copy()
            #ar = a.ravel()
            # This part works for any shape of a
            ar = a.copy()

            # Resample each group separately
            ar[isTreatment] = self.BootstrapResample(ar[isTreatment])     # note that in bootstrap resampling, the
            ar[~isTreatment] = self.BootstrapResample(ar[~isTreatment])   # labels don't actually get scrambled
            #return a 
            return ar

        # General case: resample the entire array
        ind = np.random.randint(a.shape[0], size=a.shape[0])
        #return a.flat[ ind ]
        return a[ind]


    def pvalue_interval(self, p=None, N=None):
        """
        Calculates the confidence interval for a proportion.

        Args:
            p: Observed proportion (success rate).
            N: Sample size.

        Returns:
            tuple: Lower bound, observed proportion, and upper bound of the 95% confidence interval for the proportion.
        """
        # Calculate upper and lower bounds of the 95% confidence interval
        p_up = p + 1.96 * np.sqrt(p * (1 - p) / N)
        p_down = p - 1.96 * np.sqrt(p * (1 - p) / N)
        
        # Adjust lower bound if necessary to avoid negative probability
        if p_down <= 0: p_down = 1e-7
        return p_down, p, p_up


    def negP(self, p=None):
        """
        Calculates the negative natural logarithm of a probability.

        Args:
            p: A probability value (0 < p ≤ 1).

        Returns:
            The negative natural logarithm of the probability `p`.
        """
        return -np.log(p)

























class Plotting():
    """
    A class for generating various plots.

    This class provides methods to plot EEG data in various formats,
    It uses the matplotlib library for generating most of the graphs.

    Attributes:
        None

    Methods:
        show_electrode
        plot_frequency_bands
        plot_topomap_L_R
    """

    def __init__(self):
        pass


    def show_electrode(self, ch_location=None, ch_list=None, label=False, color='red', alpha=1, ax=None, alpha_back=0.5):
        """
        Displays electrode positions on a 2D plot.

        Args:
            ch_location (list of tuples): List of tuples containing channel names and their X, Y coordinates.
            ch_list (list): List of channel names to highlight.
            label (bool): If True, display labels for the highlighted channels.
            color (str): Color for the highlighted channels. Default is 'red'.
            alpha (float): Opacity level for the highlighted channels.
            ax (matplotlib.axes.Axes): Axis containing the plot.
            alpha_back (float): Opacity of background channels. 

        Returns:
            None
        """
        # Plot all electrodes in grey with partial opacity
        if ax: ax.scatter([x[1] for x in ch_location], [x[2] for x in ch_location], color='grey', alpha=alpha_back)
        else: plt.scatter([x[1] for x in ch_location], [x[2] for x in ch_location], color='grey', alpha=alpha_back)
        # Exit if no channels to highlight
        if not ch_list: 
            ch_list = [ch[0] for ch in ch_location]
         
        # Highlight and optionally label specified channels
        for i,ch in enumerate(ch_list): 
            y = [[x[1],x[2]] for x in ch_location if x[0]==ch][0]
            if type(color)==list and len(color)>1: 
                if ax: ax.scatter(y[0], y[1], color=color[i], alpha=alpha)
                else: plt.scatter(y[0], y[1], color=color[i], alpha=alpha)
            else: 
                if ax: ax.scatter(y[0], y[1], color=color, alpha=alpha)
                else: plt.scatter(y[0], y[1], color=color, alpha=alpha)
            if label: 
                if ax: ax.text(y[0], y[1], ch)
                else: plt.text(y[0], y[1], ch)


    def plot_frequency_bands(self, ax=None, ylim=None):
        """
        Adds frequency band annotations to a plot.

        Args:
            ax: Matplotlib axis object to which the frequency band lines and labels are added.
            ylim: Tuple of (ymin, ymax) specifying the vertical limits of the plot. Used to position text labels.
        """
        # Frequency band annotations with their upper limit and label position
        bands = {r'$\delta$': [4, 2], r'$\theta$': [8, 5.5], r'$\alpha$': [13, 10], r'$\beta$': [32, 21], r'$\gamma$': [50, 35]}
        for band, [freq, text_pos] in bands.items():
            # Draw vertical line for each band
            if ax: ax.axvline(x=freq, color='grey', linestyle='--', linewidth=1, alpha=0.5)
            else: plt.axvline(x=freq, color='grey', linestyle='--', linewidth=1, alpha=0.5)
            # Place text label if ylim is provided
            if ylim:
                delta = abs(ylim[1] - ylim[0]) * 0.13  # Calculate vertical position for text
                if ax: ax.text(text_pos, ylim[1] - delta, band, horizontalalignment='center')
                else: plt.text(text_pos, ylim[1] - delta, band, horizontalalignment='center')


    def plot_topomap_L_R(self, ax=None, RAW=None, dataL=None, dataR=None, cmap='viridis', vlim=None, masks=None, mask_params=None, text_range=None, text=None):
        """
        Plots left and right EEG topomaps using MNE's plot_topomap function.

        Args:
            ax (list of matplotlib.axes._subplots.AxesSubplot): List of axes on which to draw the topomaps for left, right, and colorbar.
            RAW (mne.io.Raw): Raw MNE object containing the EEG data and channel information.
            dataL (numpy.ndarray): Array containing the data for the left hemisphere topomap.
            dataR (numpy.ndarray): Array containing the data for the right hemisphere topomap.
            cmap (str or matplotlib.colors.Colormap, optional): Colormap to use for both topomaps. Default is viridis
            vlim (tuple of float, optional): Value limits for the colormap.
            masks (list of numpy.ndarray): List of masks to apply to the topomap data.
            mask_params (list of dict): List of dictionaries specifying parameters for each mask.
            text_range (str): Text to display as the title of the topomaps indicating the data range or condition.
            text (bool, optional): Flag to indicate whether additional text labels should be added to the colorbar axis.
        """

        # Plot left hemisphere topomap
        im, cm = mne.viz.plot_topomap(dataL, RAW.info, ch_type='eeg', sensors=True, cmap=cmap, vlim=vlim,
                                      mask=masks[0], mask_params=mask_params[0], show=False, axes=ax[0])
        im, cm = mne.viz.plot_topomap(dataL, RAW.info, ch_type='eeg', sensors=True, cmap=cmap, vlim=vlim,
                                      mask=masks[1], mask_params=mask_params[1], show=False, axes=ax[0])
        ax[0].set_title(f"{text_range} (Left)")

        # Plot right hemisphere topomap
        im, cm = mne.viz.plot_topomap(dataR, RAW.info, ch_type='eeg', sensors=True, cmap=cmap, vlim=vlim,
                                      mask=masks[0], mask_params=mask_params[0], show=False, axes=ax[1])
        im, cm = mne.viz.plot_topomap(dataR, RAW.info, ch_type='eeg', sensors=True, cmap=cmap, vlim=vlim,
                                      mask=masks[2], mask_params=mask_params[1], show=False, axes=ax[1])
        ax[1].set_title(f"{text_range} (Right)")

        # Prepare colorbar axis
        clim = dict(kind='value', lims=[vlim[0], 0, vlim[1]])
        divider = make_axes_locatable(ax[2])
        ax[2].set_yticks([])
        ax[2].set_xticks([])
        ax[2].axis('off')
        cax = divider.append_axes(position='left', size='20%', pad=0.5)

        # Plot colorbar
        mne.viz.plot_brain_colorbar(cax, clim=clim, colormap=cmap, transparent=False, orientation='vertical', label=None)

        # Add optional text to colorbar axis
        if text:
            ax[2].text(0, 0.26, 'Target (.)', color='lime')
            ax[2].text(0, 0.25, 'Target (.)', color='black')
            ax[2].text(0, 0.1, 'Interpolated (x)', color='black')

        # Note: Some lines seem to repeat with slight modifications (e.g., mask parameters). It's assumed these are intentional
        # for demonstration purposes. Ensure this aligns with your actual processing needs.


    def make_cmap(self, cmap, name, n=256):
        """
        Create an alternative cmap
        """
        cmap = LinearSegmentedColormap(name, cmap, n)
        """
        if name not in plt.colormaps():
            try:
                matplotlib.cm.register_cmap(name=name, cmap=cmap)
            except Exception as e:
                print(f"Failed to register colormap '{name}'. Error: {str(e)}")
        """
        return cmap

    # Define the colormap dictionary
    kelvin_i = {
        'red': (
            (0.000, 0.0, 0.0),
            (0.350, 0.0, 0.0),
            (0.500, 1.0, 1.0),
            (0.890, 1.0, 1.0),
            (1.000, 0.5, 0.5),
        ),
        'green': (
            (0.000, 0.0, 0.0),
            (0.125, 0.0, 0.0),
            (0.375, 1.0, 1.0),
            (0.640, 1.0, 1.0),
            (0.910, 0.0, 0.0),
            (1.000, 0.0, 0.0),
        ),
        'blue': (
            (0.000, 0.5, 0.5),
            (0.110, 1.0, 1.0),
            (0.500, 1.0, 1.0),
            (0.650, 0.0, 0.0),
            (1.000, 0.0, 0.0),
        ),
    }
    # Create and register the custom colormap
    kelvin_i_cmap = make_cmap(kelvin_i, 'kelvin_i', 256)


    def make_simple_cmap(self, c1=None, c2=None, c3=None):
        """
        Create a simple 3-color cmap

        Args: 
            c1 (str): Color to set at the lower value
            c2 (str): Color to set in the middle
            c3 (str): Color to set at the higher value 

        Returns: 
            cmap (matplotlib.colors.LinearSegmentedColormap)
        """
        colors = [(0.0, c1),     # Color at -1
                  (0.5, c2),     # Color at 0
                  (1.0, c3)]     # Color at 1

        return LinearSegmentedColormap.from_list('custom_cmap', colors)