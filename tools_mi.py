import os
import numpy as np
from BCI2000Tools.FileReader import bcistream
from BCI2000Tools.Electrodes import *
from BCI2000Tools.Plotting import *
import mne 
import matplotlib.pyplot as plt
from pyprep.prep_pipeline import PrepPipeline, NoisyChannels

import eeg_dict

class tools_mi():
    def __init__(self):
        pass

    def create_folder(self, path=None, folder_name=None, verbose=False):
        """
        Creates a folder at a specified path.

        Args:
            path (str, optional): The directory path where the folder will be created. If None, it defaults to the current working directory.
            folder_name (str): The name of the folder to be created. If None, no folder will be created.
            verbose (bool, optional): If True, prints additional messages about the folder creation process. Defaults to False.

        Returns: 
            None
        """
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


    def import_file_dat(self, path=None, file_name=None, montage_type=None, verbose=True):
        """
        Imports and processes EEG data from a .dat file.

        Args:
            path (str, optional): The file path leading to the .dat file
            file_name (str): The name of the .dat file to be imported.
            montage_type (str): Specifies the electrode montage type ['EGI_128', 'DSI_24', 'GTEC_32']
            verbose (bool, optional): If True, prints detailed information

        Returns:
            tuple: A tuple containing the processed signal data, state information, sampling rate, channel names, total file time, and block size
        """

        # Load the .dat file 
        b = bcistream(path + file_name)
        signal, states = b.decode()
        signal = np.array(signal)
        
        # Retrieve additional parameters from the file
        fs = b.samplingrate()  # Sampling rate
        ch_names = b.params['ChannelNames']  # Channel names
        blockSize = b.params.SampleBlockSize  # Block size used in data acquisition

        # Define paradigm constants
        _nBlocks=8
        _trialsPerBlock=4
        _initialSec=2
        _stimSec=3
        _taskSec=10

        # Check if the block size is a perfect divisor of the sampling frequency, this will ensure every block to be saved
        if fs % blockSize == 0:
            print(f'Block size fits perfectly in sample frequency!')
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

        # Reject some channels for 'DSI_24'
        if montage_type == 'DSI_24':
            # Example: Removing unwanted channels
            chKeep_idx = [i for i, ch in enumerate(ch_names) if ch not in ['X1', 'X2', 'X3', 'TRG']]
            signal = signal[chKeep_idx]
            ch_names = np.array(ch_names)[chKeep_idx].tolist()

        # Extract stimulus-related information
        StimulusCode = states['StimulusCode']  # Stimulus codes
        StimulusBegin = states['StimulusBegin']  # Onsets of stimuli
        # Extract saved file time 
        fileTime = signal.shape[1] / fs

        # If verbose, print a detailed summary of the .dat file content
        if verbose: 
            # Summary includes channel info, sampling rate, and time details
            print(f'\nEEG channels: {signal.shape[0]} Total ticks: {signal.shape[1]}')
            print(f'Each tick corresponds to [s]: {1/fs}')
            print(f'Sampling rate [Hz]: {fs} ~~~ Time on file [s]: {fileTime}')
            print(f'This file contains {round(fileTime / totalFileTime * 100, 2)}% of MI paradigm')
            # Additional calculations for recorded blocks and trials
            netTime = fileTime - _initialSec
            blockTime = _trialsPerBlock*(_stimSec + _taskSec)
            recordedBlocks = int(round(netTime / blockTime, 3))
            print(f'    Number of full blocks: { recordedBlocks }')
            leftOver = netTime - recordedBlocks * blockTime
            print(f'    Additional trials: { abs(leftOver) // (_stimSec + _taskSec) }')
            #print(f'Ch-list: {ch_names}')
            print(f'StimulusCode: {np.unique(StimulusCode, return_counts=True)}')
            print(f'StimulusBegin: {np.unique(StimulusBegin, return_counts=True)}')
            print(f'Signal range: [{np.min(signal)}, {np.mean(signal)}, {np.max(signal)}]\n')

        return signal, states, fs, ch_names, fileTime, blockSize


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
        # Saves the RAW data to the specified file, overwriting any existing file
        RAW.save(path + file_name + label + '.fif', overwrite=True)


    def filter_data(self, signal=None, fs=None, l_freq=None, h_freq=None):
        """
        This function uses MNE-Python's filter_data method to apply a Finite Impulse Response (FIR) bandpass filter to the given signal.
        The function converts the input signal to `float64` to ensure compatibility with the MNE-Python filtering function

        Args:
            signal (array-like): The time-series signal to be filtered
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
            ch_set (object): An object representing the set of channels/electrodes, from BCI2000Tools.Electrodes
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
            ch_set: The channel set containing channel labels, from BCI2000Tools.Electrodes
            ch_name: A string or list of strings specifying the channel(s) to find (examples 'Cz', 'Cz C3 C4')

        Returns:
            List of indices for the specified channel(s), or None if a channel is the current reference.
        """
        # Delegate to ch_set's find_labels method to locate channel indices
        return ch_set.find_labels(ch_name)


    def make_RAW(self, signal=None, fs=None, ch_names=None):
        """
        Constructs an MNE RAW object from signal data, sampling frequency, and channel names.

        Args:
            signal: The EEG signal data as a 2D numpy array (channels x time points).
            fs: Sampling frequency of the signal in Hz
            ch_names: List of channel names corresponding to the signal rows

        Returns:
            An MNE Raw object containing the EEG data
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
            RAW: The original MNE RAW object containing EEG data.
            states: Dictionary where keys are stim channel names and values are the data arrays.

        """
        # Retrieve sampling frequency from the original RAW object
        fs = RAW.info['sfreq']
        # Create MNE info object for stimulation channels
        info = mne.create_info(ch_names=[x for x in states.keys()], sfreq=fs, ch_types='stim')
        # Create RawArray for stimulation data
        stim = mne.io.RawArray([x[0] for x in states.values()], info, first_samp=0, verbose=False)
        # Add stim channels to the original RAW object
        RAW.add_channels([stim])


    def make_PREP(self, RAW, isSNR=False, isCorrelation=False, isDeviation=False, isHfNoise=False, isNanFlat=False, isRansac=False):
        """
        Identifies noisy channels in an MNE Raw object using various criteria and marks them as bad.

        Args:
            RAW: The MNE Raw object to be processed.
            isSNR: If True, identifies bad channels by signal-to-noise ratio.
            isCorrelation: If True, identifies bad channels by correlation.
            isDeviation: If True, identifies bad channels by deviation.
            isHfNoise: If True, identifies bad channels by high-frequency noise.
            isNanFlat: If True, identifies bad channels by NaN or flat signals.
            isRansac: If True, identifies bad channels using RANSAC.

        """
        # Initialize NoisyChannels with the RAW object
        NC = NoisyChannels(RAW, do_detrend=False)
        # Apply different criteria based on the function parameters to find bad channels
        if isSNR: NC.find_bad_by_SNR()
        if isCorrelation: NC.find_bad_by_correlation(correlation_secs=1.0, correlation_threshold=0.4, frac_bad=0.01)
        if isDeviation: NC.find_bad_by_deviation(deviation_threshold=5.0)
        if isHfNoise: NC.find_bad_by_hfnoise(HF_zscore_threshold=5.0)
        if isNanFlat: NC.find_bad_by_nan_flat()
        if isRansac: NC.find_bad_by_ransac(n_samples=50, sample_prop=0.25, corr_thresh=0.75, frac_bad=0.4, corr_window_secs=5.0, channel_wise=False, max_chunk_size=None)

        # Collect names of identified bad channels
        ch_names_bad = []
        bad_dict = NC.get_bads(as_dict=True)
        if isSNR: ch_names_bad += bad_dict['bad_by_SNR']
        if isCorrelation: ch_names_bad += bad_dict['bad_by_correlation']
        if isDeviation: ch_names_bad += bad_dict['bad_by_deviation']
        if isHfNoise: ch_names_bad += bad_dict['bad_by_hf_noise']
        if isNanFlat: ch_names_bad += bad_dict['bad_by_nan'] + bad_dict['bad_by_flat']
        if isRansac: ch_names_bad += bad_dict['bad_by_ransac']

        # Mark identified bad channels in the RAW object
        RAW.info["bads"].extend(np.unique(ch_names_bad).tolist())


    def mark_BAD_region(self, RAW=None, block=None):
        """
        Opens an interactive plot for visually identifying and marking bad regions in the EEG data.

        Args:
            RAW: The MNE Raw object containing the EEG data.
            block: Determines if the plot should block execution until closed. If True, execution is halted until the plot is manually closed

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
            RAW: The MNE Raw object containing the EEG data and annotations.
            label: The label used to identify regions of interest in the annotations. Defaults to 'BAD_region'.
            max_duration: The maximum duration expected for the data, used to calculate the percentage of data marked as bad. Defaults to 418 seconds but it should be the total file time

        """
        # Extract annotations from the RAW object
        annot = RAW.annotations
        # Identify durations of annotations matching the specified label
        bad_regions_id = annot.duration[np.where(annot.description == label)]
        # Print a summary of bad sections, their total duration, and the percentage of the maximum duration
        print(f" --> {label}: {len(bad_regions_id)} sections, ~{round(sum(bad_regions_id),1)} s [{round(sum(bad_regions_id)/max_duration*100,1)}%] --> Bad channels: {RAW.info['bads']}")









    def make_annotation_MI(self, RAW, fs, nBlocks=None, trialsPerBlock=None, initialSec=None, stimSec=None, taskSec=None, rejectSec=None, nSplit=None, fileTime=None):
        '''
        # This is a brute force method (takes time, and it's the ideal one for a new paradigm
        # Since the paradigm is the same for now we know the target locations of the ticks we need to modify so we use an alternative method (faster)
        # Modify StimulusCode to differentiate resting periods after moving LEFT vs RIGHT
        new_StimulusCode = RAW['StimulusCode'][0][0].copy()
        change=False
        for i in range(len(RAW['StimulusCode'][0][0])):
            if RAW['StimulusCode'][0][0][i]==1: change=False
            if RAW['StimulusCode'][0][0][i]==2: change=True
            if RAW['StimulusCode'][0][0][i]==3 and change: new_StimulusCode[i]=4
        RAW['StimulusCode'][0][0] = new_StimulusCode
        '''

        initialTicks = int(initialSec * fs) # initial length in Ticks
        stimTicks = int(stimSec * fs) # stimulus length in Ticks (cue)
        taskTicks = int(taskSec * fs) # task length in Ticks (performance)
        rejectTicks = int(rejectSec * fs) # rejection length in Ticks (after cue, subtracted from performance)

        # This is the alternative method (faster)
        new_StimulusCode = RAW['StimulusCode'][0][0].copy()

        def change_StimulusCode(list_, nBlocks, trialsPerBlock, initialTicks, stimTicks, taskTicks, posInBlock, newStimCode):
            for i in range(nBlocks):
                start = initialTicks + (stimTicks + taskTicks) * (posInBlock-1 + trialsPerBlock*i)
                end = start + stimTicks
                try: 
                    list_[start:end] = [newStimCode]*stimTicks
                except ValueError:
                    break

        # Changing rest-after-right trials (4th position with code 3) to newStimCode 4 to differentiate them from rest-after-left trials
        change_StimulusCode(new_StimulusCode, nBlocks, trialsPerBlock, initialTicks, stimTicks, taskTicks, posInBlock=4, newStimCode=4)

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

        print(f'\n~~~~~~~~ ANNOTATIONS ~~~~~~~~')
        print(f'File length: {len(new_StimulusCode)/fs} s or {len(new_StimulusCode)} ticks')
        print(f'File length (remove initial pause): {(len(new_StimulusCode)-initialTicks)/fs} s or {len(new_StimulusCode)-initialTicks} ticks')
        print(f'Block length: {(len(new_StimulusCode)-initialTicks)/fs/nBlocks} s or {(len(new_StimulusCode)-initialTicks)/nBlocks} ticks')
        print(f'Command length: {(len(new_StimulusCode)-initialTicks)/fs/nBlocks/trialsPerBlock} s or {(len(new_StimulusCode)-initialTicks)/nBlocks/trialsPerBlock} ticks')
        print(f'\tStim length: {(stimTicks)/fs} s or {stimTicks} ticks')
        print(f'\tTask length: {(taskTicks)/fs} s or {taskTicks} ticks')

        def expand_onset(onset, nSplit, taskSec, rejectSec):
            if nSplit > 1: 
                for x in onset: 
                    expand = np.linspace(x, x+(taskSec-rejectSec), nSplit+1)
                    onset = onset + list(expand[:-1])
            return list(np.unique(onset))

        def generate_onsets(start, end, stimSec, taskSec, trialsPerBlock, nSplit, rejectSec):
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

        # annotate left hand pause
        onset3, duration3 = generate_onsets(initialSec + taskSec*1 + stimSec*2, fileTime, stimSec, taskSec, trialsPerBlock, nSplit, rejectSec) # in seconds [18,70,122,174,...] without rejectSec
        description3 = ['left_rest' for x in onset3]
        #print(len(onset3), len(duration3), len(description3))

        # annotate right hand pause
        onset4, duration4 = generate_onsets(initialSec + taskSec*3 + stimSec*4, fileTime, stimSec, taskSec, trialsPerBlock, nSplit, rejectSec) # in seconds [44,96,148,200,...] without rejectSec
        description4 = ['right_rest' for x in onset4]
        #print(len(onset4), len(duration4), len(description4))

        # annotate cue regions inter-trial
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

        onset = onset1 + onset2 + onset3 + onset4 + onset5 + onset6
        duration = duration1 + duration2 + duration3 + duration4 + duration5 + duration6
        description = description1 + description2 + description3 + description4 + description5 + description6

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

        my_annot = mne.Annotations(onset=onset, duration=duration, description=description)
        RAW.set_annotations(my_annot)
        #print(RAW.annotations)

        events_from_annot, event_dict = mne.events_from_annotations(RAW)
        #print(event_dict)
        #print(events_from_annot[:])

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
            if conv_dict: idx.append(montage.ch_names.index(conv_dict[ch]))
            else: idx.append(montage.ch_names.index(ch))
        
        # Update montage to only include specified channels
        montage.ch_names = ch_to_show
        montage.dig = montage.dig[0:3] + [montage.dig[x + 3] for x in idx]
        # Plot the montage
        if verbose: 
            montage.plot()
        return montage



    def show_electrode(self, ch_location=None, ch_list=None, label=False, color='red', alpha=1):
        # Plot locations in a X,Y plane of electrodes of interest, can show labels
        plt.scatter([x[1] for x in ch_location], [x[2] for x in ch_location], color='grey', alpha=0.5)
        if not ch_list: 
            return None
        else: 
            for ch in ch_list: 
                y = [[x[1],x[2]] for x in ch_location if x[0]==ch][0]
                plt.scatter(y[0], y[1], color=color, alpha=alpha)
                if label: plt.text(y[0], y[1], ch)


    def find_ch_central(self, ch_location=None, ch_list=None):
        # If no target list is specified use any electrode
        if not ch_list: 
            ch_list = [x[0] for x in ch_location]
        return [x[0] for x in ch_location if (x[1]==0 and x[0] in ch_list)]


    def find_ch_left(self, ch_location=None, ch_list=None):
        # If no target list is specified use any electrode
        if not ch_list: 
            ch_list = [x[0] for x in ch_location]
        return [x[0] for x in ch_location if (x[1]<0 and x[0] in ch_list)]


    def find_ch_right(self, ch_location=None, ch_list=None):
        # If no target list is specified use any electrode
        if not ch_list: 
            ch_list = [x[0] for x in ch_location]
        return [x[0] for x in ch_location if (x[1]>0 and x[0] in ch_list)]


    def find_ch_circle(self, ch_location=None, ch_list=None, radius=None):
        # Find which electrodes in a list of interest are contained in a circle with given radius
        # If no target list is specified use any electrode
        if not ch_list: 
            ch_list = [x[0] for x in ch_location]
        circle = []
        for x in ch_location:
            if x[0] in ch_list:
                    if np.sqrt(x[1]**2 + x[2]**2) <= radius: circle.append(x[0])
        return circle


    def find_ch_symmetry(self, ch_location=None, ch_list=None):
        # Create a dictionary of symmetries given a list of channel of interest
        # If no target list is specified use any electrode
        if not ch_list: 
            ch_list = [x[0] for x in ch_location]
        symmetry = {}
        for x in ch_location:
            if x[0].lower() in [ch.lower() for ch in ch_list]: 
                ch1 = x[0]
                x1 = x[1]
                y1 = x[2]
                for y in ch_location:
                    if y[0] in ch_list: 
                        ch2 = y[0]
                        x2 = y[1]
                        y2 = y[2]
                        if x1==-x2 and y1==y2: symmetry[ch1] = ch2 
        return symmetry


    def interpolate(self, RAW=None, reset_bads=True):
        # Identify bad channels (previously saved)
        # Requires RAW.set_montage(montage) before interpolating
        print(f"BAD CHANNELS to be interpolated: {RAW.info['bads']}")
        # Interpolation via spline (don't reset information regarding bad channels)
        old_ch_bad = RAW.info['bads']
        RAW.interpolate_bads(reset_bads=reset_bads)
        if reset_bads: 
            print(f"RAW.info['bads'] have been modified")
        elif not reset_bads: 
            print(f"RAW.info['bads'] have not been modified")

        return old_ch_bad


    def make_epochs(self, RAW=None, tmin=None, tmax=None, twindow=None, event_id=None, events_from_annot=None, verbose=False):
        expected_epochs_per_type = sum([1 for x in events_from_annot if x[2]==event_id])  # Calculate total expected number of epochs per type (e.g. LEFT)
        if verbose: print(f"Expected {expected_epochs_per_type} epochs {tmax-tmin}s-long (per type)")
        # Generate Epochs
        # Generate and store epochs for each condition
        epochs_ = mne.Epochs(RAW, events_from_annot, event_id=event_id, tmin=tmin, tmax=tmax, baseline=None, preload=True, verbose=False)
        # Note: Index 0 in shape is the number of epochs, index 1 is channel numbers, index 2 is EEG values at segment time t
        # Print the number of Epochs generated
        n_epochs = epochs_.get_data(picks='eeg').shape[0]  # Number of epochs for the current condition
        if verbose: print(f"Summary: {n_epochs}/{expected_epochs_per_type} total epochs")
        return epochs_


    def make_psd(self, epochs=None, fs=None, resolution=None, tmin=None, tmax=None, fmin=None, fmax=None, nPerSegment=None, nOverlap=None, aggregate=True, verbose=False):
        # Define bin resolution in frequency space and run PSD
        # Resolution defines bin width [Hz]
        nfft = int(fs / resolution)
        effective_window = (1 / resolution)
        expected_segments = int(((tmax - tmin) - effective_window) / (effective_window - nOverlap / fs)) + 1
        expected_bins = int((fmax - fmin) / resolution + 1)
        window = 'hann'
        
        # Welch method
        psd_ = epochs.compute_psd(method='welch', 
                                 fmin=fmin, fmax=fmax, 
                                 tmin=tmin, tmax=tmax, 
                                 n_fft=nfft, 
                                 n_overlap=nOverlap, 
                                 n_per_seg=nPerSegment, 
                                 average=None, 
                                 window=window, 
                                 output='power', 
                                 verbose=verbose).get_data()
        if verbose: 
            print(f"Expected w/ resolution {resolution} [Hz/bin]: ")
            print(f"  - Eff_Window Length [s]: {effective_window}")
            print(f"  - Epochs: {epochs.get_data(picks='eeg').shape[0]} -> {tmax-tmin} s-long (per type)")
            print(f"  - Channels: {epochs.get_data(picks='eeg').shape[1]}")
            print(f"  - Bins: {expected_bins}")
            print(f"  - Expected segments (or periodograms): {expected_segments}")
            print(f"Dimension check: (epoch, ch, bins, segments/periodograms) = {psd_.shape}")
        
        if aggregate: 
            # if average='mean' is defined in compute_psd(), a dimension is lost and no need to aggregate periodograms
            if len(psd_.shape)>3:
                # Aggregate periodograms via mean (Welch)
                psd_ = np.mean(psd_, axis=3)
                if verbose: print(f"Aggregate segments/periodograms: (epoch, ch, bins) = {psd_.shape}")
            # Aggregate PSDs from different epochs epochs via mean
            psd_ = np.mean(psd_, axis=0)
            if verbose: print(f"Aggregate epoch-wise: (ch, bins) = {psd_.shape}")
            
        return psd_



    def CalculateEtas(self, x=None, isTreatment=None, signed=True):
        """
            Calculates the correlation ratio between two arrays X and Y by comparing the variance between groups to the
            variance within group

            The ratio between the variance between groups and the total variance is commonly referred to as the "Eta squared (η²)" in statistics. 
            It is a measure of effect size for use in ANOVA (Analysis of Variance) and represents the proportion of the total variance in the dependent variable that is attributable to the variance between groups.
            Here's a breakdown of its components:
                Between-group variance (SSB): Variance due to the difference between the means of the groups.
                Total variance (SST): The sum of the between-group variance (SSB) and within-group variance (SSW).
            Eta squared is a useful measure for understanding the proportion of the overall variance that is explained by the grouping variable, 
            indicating the effect size of the group differences on the dependent variable.
            
            η² = SSB/SST
        """
        # x contains the psds (trial, ch, bin)
        # isTreatment is labels for task and rest (trial)
        
        # Sample sizes
        n1 = np.sum(isTreatment)
        n2 = np.sum(~isTreatment)
        
        # Samples
        x1 = x[isTreatment]
        x2 = x[~isTreatment]
        # Samples means over the trials
        mu1 = np.mean(x1, axis=0)
        mu2 = np.mean(x2, axis=0)
        grand_mean = np.mean(x, axis=0)
        
        # Sum of squares within groups
        ssw = np.sum((x1 - mu1)**2, axis=0) + np.sum((x2 - mu2)**2, axis=0)
        # Sum of squares between groups
        ssb = n1*(mu1 - grand_mean)**2 + n2*(mu2 - grand_mean)**2
        
        # eta-square
        if not signed: return ssb/(ssb + ssw)
        else: 
            signs = np.where(mu1 - mu2 > 0, 1, -1)
            return ssb/(ssb + ssw) * signs


    def r2(self, x=None, isTreatment=None, signed=True):
        # This is like calculating eta square for an ANOVA
        # Instead the categorical variable (isTreatment) is transformed into a dummy variable (y) with range of choice
        # The choice of the range doesn't matter but the choice of which group has a bigger label changes the sign of r
        r = np.zeros((x.shape[1], x.shape[2]))
        y = np.where(isTreatment==True, 1,0)
        #for ch in range(x.shape[1]):
        #    for b in range(x.shape[2]):
        #        r[ch, b] = np.corrcoef( x[:,ch,b], y )[0,1]
                
        # Calculate means
        x_mean = np.mean(x, axis=0)
        y_mean = np.mean(y)
        # Calculate the numerator
        numerator = np.sum((x - x_mean) * (y - y_mean)[:, np.newaxis, np.newaxis], axis=0)
        # Calculate the denominator
        x_diff_sq = np.sum((x - x_mean) ** 2, axis=0)
        y_diff_sq = np.sum((y - y_mean) ** 2)
        denominator = np.sqrt(x_diff_sq * y_diff_sq)
        # Compute correlation coefficients
        r = numerator / denominator        
        if signed: return r*abs(r)
        else: return r*r


    def Shuffle(self, a=None):
        """
        Shuffle sequence `a` in-place and also return a reference to it.
        This is the helper function at the core of the `ApproxPermutationTest()`.
        """
        np.random.shuffle( a )
        return a



    def ApproxPermutationTest(self, x=None, isTreatment=None, stat=None, nSimulations=1999, plot=False):
        """
        One-sided two-sample approximate permutation test assuming the
        value of `stat(x,isTreatment)` is expected to be larger under H1
        than under H0.

        We call this an "approximate" permutation test because an actual
        exact permutation test would test *every* permutation exhaustively,
        whereas this one approximates the same distribution by repeated
        random label reshuffling.

        Note that permutation tests potentially suffer from the Behren's-
        Fisher problem: a difference-of-means permutation test will perform
        similarly to a naive (uncorrected) t-test in that regard. To fix
        this, use `BootstrapTest()` instead.
        """
        isTreatment = isTreatment.copy() # copy once, then shuffle-in-place many times to save avoid allocation overhead in the simulation loop
        observed = stat(x, isTreatment)
        print(observed)
        hist = []
        for iSimulation in range( nSimulations ):
            hist.append( stat(x, self.Shuffle( isTreatment )) )
        if plot: 
            fig = plt.hist(hist)
            plt.axvline(observed, color='black')
            plt.show()
            
        nReached = sum( np.array(hist) >= observed  )
        return ( 0.5 + nReached ) / ( 1.0 + nSimulations )


    def BootstrapTest(self, x=None, isTreatment=None, stat=None, nSimulations=1999, nullHypothesisStatValue=0.0, plot=False):
        """
        Efron & Tibshirani page 215, equation (15.32)

        Again this is equivalent to a one-sided two-sample test and again,
        we assume the value of `stat(x,isTreatment)` is expected to be
        *larger* under H1 than under H0. However, the math ends up being
        rearranged somewhat to perform the test, so we'll need to
        specify explicitly the `stat() value that we expect under the
        null hypothesis (and we will be counting the simulation results
        that go *below* it---however, don't be deceived by this: the
        situation is still the same as in the other tests, in the sense
        that a bigger effect still means a higher `stat()` value).

        Bootstrap tests avoid the Behren's-Fisher problem that you get with
        permutation tests: a difference-of-means bootstrap test will perform
        similarly to a t-test with Welch's correction.
        """
        hist = []
        for iSimulation in range( nSimulations ):
            hist.append( stat( self.BootstrapResample(x, isTreatment),isTreatment) )
        if plot: 
            plt.hist(hist)
            plt.axvline(nullHypothesisStatValue, color='red')
            plt.axvline(stat( x,isTreatment ), color='black')
            plt.show()
            
        nReached = sum( np.array(hist) < nullHypothesisStatValue )
        return ( 0.5 + nReached ) / ( 1.0 + nSimulations )


    def BootstrapResample(self, a=None, isTreatment=None):
        """
        Sample with replacement from `a` (or from each class of `a`, separately).
        This is the helper function at the core of the `BootstrapTest()`.
        """
        if isTreatment is not None:
            isTreatment = isTreatment.ravel()
            # This part only works if a.shape[1] doesn't exist
            #a = a.copy()
            #ar = a.ravel()
            
            # This part works for any shape of a
            ar = a.copy()

            ar[  isTreatment ] = self.BootstrapResample( ar[  isTreatment ] ) # note that in bootstrap resampling, the
            ar[ ~isTreatment ] = self.BootstrapResample( ar[ ~isTreatment ] ) # labels don't actually get scrambled
            #return a 
            return ar

        ind = np.random.randint( a.shape[0], size=a.shape[0] )
        #return a.flat[ ind ]
        return a[ ind ]


    def convert_dB(self, X=None):
        """
        Converts given X values to decibels (dB), input X is assumed to be in uV² (microVolts square)
        Steps: 
            Converts X from uV² to V²
            Converts to dB using 1V as reference
        """
        return 10*np.log10(X*1e12)


    def plot_frequency_bands(self, ax=None, ylim=None):
        """
        """
        # Frequency band annotations
        bands = {r'$\delta$': [4, 2], r'$\theta$': [8, 5.5], r'$\alpha$': [12, 10], r'$\beta$': [30, 21], r'$\gamma$': [50, 35]}
        for band, [freq,text_pos] in bands.items():
            ax.axvline(x=freq, color='grey', linestyle='--', linewidth=1, alpha=0.5)
            if ylim: 
                delta = abs(ylim[1] - ylim[0])*0.13
                ax.text(text_pos, ylim[1] - delta, band, horizontalalignment='center')


    def pvalue_interval(self, p=None, N=None):
        p_up = p + 1.96*np.sqrt(p*(1-p)/N)
        p_down = p - 1.96*np.sqrt(p*(1-p)/N)
        if p_down<=0: 
            p_down = 1e-7
        return p_down, p, p_up


    def negP(self, p):
        return -np.log(p)







class SumsR2:
    def __init__(self, ch_set=None, dict_symm=None, isContralat=None, bins=None, transf='r2'):
        self.ch_set = ch_set
        self.ch_names = np.array(self.ch_set.get_labels())
        self.dict_symm = dict_symm
        self.isContralat = isContralat
        self.bins = bins
        self.transf = transf
        
    def calculateEtas(self, x=None, isTreatment=None, signed=True):
        """
            Calculates the correlation ratio between two arrays X and Y by comparing the variance between groups to the
            variance within group

            The ratio between the variance between groups and the total variance is commonly referred to as the "Eta squared (η²)" in statistics. 
            It is a measure of effect size for use in ANOVA (Analysis of Variance) and represents the proportion of the total variance in the dependent variable that is attributable to the variance between groups.
            Here's a breakdown of its components:
                Between-group variance (SSB): Variance due to the difference between the means of the groups.
                Total variance (SST): The sum of the between-group variance (SSB) and within-group variance (SSW).
            Eta squared is a useful measure for understanding the proportion of the overall variance that is explained by the grouping variable, 
            indicating the effect size of the group differences on the dependent variable.

            η² = SSB/SST
        """
        # x contains the psds (trial, ch, bin)
        # isTreatment is labels for rest (True) and task (False)
        # Sample sizes
        n1 = np.sum(isTreatment)
        n2 = np.sum(~isTreatment)
        # Samples
        x1 = x[isTreatment]
        x2 = x[~isTreatment]
        # Samples means over the trials
        mu1 = np.mean(x1, axis=0)
        mu2 = np.mean(x2, axis=0)
        grand_mean = np.mean(x, axis=0)
        # Sum of squares within groups
        ssw = np.sum((x1 - mu1)**2, axis=0) + np.sum((x2 - mu2)**2, axis=0)
        # Sum of squares between groups
        ssb = n1*(mu1 - grand_mean)**2 + n2*(mu2 - grand_mean)**2
        # eta-square
        if not signed: return ssb/(ssb + ssw)
        else: 
            signs = np.where(mu1 - mu2 > 0, 1, -1)
            return ssb/(ssb + ssw) * signs 
    
    def calculateR2(self, x=None, isTreatment=None, signed=True):
        # This is like calculating eta square for an ANOVA
        # Instead the categorical variable (isTreatment) is transformed into a dummy variable (y) with range of choice
        # The choice of the range doesn't matter but the choice of which group has a bigger label changes the sign of r
        # Higher value label given to task which has label isTreatment = False
        r = np.zeros((x.shape[1], x.shape[2]))
        y = np.where(isTreatment==True, 1,0)
        # Calculate means
        x_mean = np.mean(x, axis=0)
        y_mean = np.mean(y)
        # Calculate the numerator
        numerator = np.sum((x - x_mean) * (y - y_mean)[:, np.newaxis, np.newaxis], axis=0)
        # Calculate the denominator
        x_diff_sq = np.sum((x - x_mean) ** 2, axis=0)
        y_diff_sq = np.sum((y - y_mean) ** 2)
        denominator = np.sqrt(x_diff_sq * y_diff_sq)
        # Compute correlation coefficients
        r = numerator / denominator
        if signed: return r*abs(r)
        else: return r*r

    def Transform(self, x=None, isTreatment=None):
        if self.transf == 'eta2': return self.calculateEtas(x=x, isTreatment=isTreatment)
        elif self.transf == 'r2': return self.calculateR2(x=x, isTreatment=isTreatment)

    def DifferenceOfSumsR2(self, x=None, isTreatment=None):

        '''
            This is for the statistics to be tested 
            This function doesn't need to track symmetric electrodes since the sum of all electrodes is performed in each hemisphere
        '''
        x = self.Transform(x, isTreatment)
        x = np.mean(x[:,self.bins[0]:self.bins[1]], axis=1)
        x1 = x[ self.isContralat ]
        isIpsilat = self.FindSymmetric(isContralat=self.isContralat)
        x2 = x[ isIpsilat ]
        return np.sum(x1) - np.sum(x2)

    def DifferenceOfR2(self, x=None, isTreatment=None):
        '''
            This is for plotting the R2 on topoplot
            This funciton needs to track the symmetric electrode in ipsilateral hemisphere
        '''
        x = self.Transform(x, isTreatment)
        x = np.mean(x[:,self.bins[0]:self.bins[1]], axis=1)
        x1 = x[ self.isContralat ]
        isIpsilat = self.FindSymmetric(isContralat=self.isContralat)
        x2 = x[ isIpsilat ]
        r2 = np.zeros(len(self.ch_names))
        for i,j in zip( self.ch_set.find_labels(np.array(self.ch_set.get_labels())[self.isContralat]), range(len(self.isContralat))):
            r2[i] = x1[j] - x2[j]
        return x

    def FindSymmetric(self, isContralat=None):
        ch_symm = []
        for ch in self.ch_names[isContralat]:
            # Find symmetric channel of contralateral channel in ipsilateral hemisphere using dict_symm
            ch_symm.append(self.dict_symm[ch])
        # Find index of that symmetric channel using ch_set.find_labels()
        return self.ch_set.find_labels(ch_symm)