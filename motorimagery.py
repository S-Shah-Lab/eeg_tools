#!/usr/bin/env -S python  #


if __name__ == '__main__':
    import argparse 
    parser = argparse.ArgumentParser( description=__doc__, formatter_class=argparse.RawTextHelpFormatter )
    parser.add_argument( '-c', metavar='cleaned',      type=bool,  default=False,     help="has the file been previously cleaned? [True, False]")
    parser.add_argument( '-f', metavar='file_path',    type=str,   default='',        help="path to the file to run scipt on")
    parser.add_argument( '-r', metavar='resolution',   type=int,   default='1',       help="resolution for PSDs [1, 2]")
    opts = parser.parse_args()

import os
import numpy as np
from BCI2000Tools.FileReader import bcistream
from BCI2000Tools.Electrodes import *
from BCI2000Tools.Plotting import *
import mne 
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable
from pyprep.prep_pipeline import PrepPipeline, NoisyChannels

import eeg_dict # Contains dictionaries and libraries for electrodes locations 
import MotorImageryTools as mi # Contains tools for eeg and motor imagery


if __name__ == '__main__':
    EEG = mi.EEG() # Initialize EEG tools
    PLOT = mi.Plotting() # Initialize Plotting tools

    clean_bool = opts.c
    file_path, file_name = os.path.split(opts.f)
    resolution = opts.r
    secPerSegment = 1/resolution
    secOverlap = secPerSegment/2
    
    ch_location = eeg_dict.ch_location

    fmin = 1
    fmax = 40

    # Extract base name from file
    base_name, extension = os.path.splitext(file_name)

    # Create a folder using base name, if folder doesn't exist 
    path_to_folder = EEG.create_folder(path=file_path+'/', folder_name=base_name)

    # Import .dat file
    signal, states, fs, ch_names, blockSize, montage_type = EEG.import_file_dat(file_path+'/', file_name)

    if montage_type == 'DSI_24': ch_info = 'DSI24_location.txt'
    elif montage_type == 'EGI_128': ch_info = 'EGI128_location.txt'
    elif montage_type == 'GTEC_32': ch_info = 'GTEC32_location.txt'

    fileTime, nBlocks, trialsPerBlock, initialSec, stimSec, taskSec = EEG.evaluate_mi_paradigm(signal=signal, states=states, fs=fs, blockSize=blockSize, verbose=True)
    nSplit = 6
    rejectSec = taskSec - 9 # 1 [s]

    tmin = 0
    twindow = (taskSec - rejectSec) # e.g. 9
    tmax = twindow/nSplit # e.g. 1.5

    # File has not been cleaned before, it's a new file
    if not clean_bool:
        # Define initial ChannelSet
        ch_set = ChannelSet(ch_info)

        plt.figure(figsize=(4.5,4.5))
        ch_set.plot()
        plt.text(-1, 1.05, f'Montage', weight='bold')
        split_text = montage_type.split('_')
        plt.text(-1, 0.95, f'{split_text[0]} {split_text[1]} Channels')
        # Save the plot
        plt.savefig(f'{path_to_folder}/montage.png')
        plt.savefig(f'{path_to_folder}/montage.pdf')
        plt.show()


        # Run PREP the first time
        RAW = EEG.make_RAW_with_montage(signal=signal * 1e-6, 
                                        fs=fs, 
                                        ch_names=ch_set.get_labels(), 
                                        montage_type=montage_type, 
                                        conv_dict=eeg_dict.stand1020_to_egi)

        # Run PREP for bad channels
        EEG.make_PREP(RAW, isSNR=True, isDeviation=True, isHfNoise=True, isNanFlat=True, isRansac=True)

        # Bandpass filter
        signalFilter = EEG.filter_data(signal, fs, l_freq=1, h_freq=40)

        # Re-reference
        if montage_type == 'EGI_128':
            # Re-reference to mastoids
            signalFilter, ch_set = EEG.spatial_filter(sfilt='REF', 
                                                      ch_set=ch_set, 
                                                      signal=signalFilter, 
                                                      flag_ch='TP9 TP10', 
                                                      verbose=True)

        # Create RAW with montage
        RAW = EEG.make_RAW_with_montage(signal=signalFilter * 1e-6, 
                                        fs=fs, 
                                        ch_names=ch_set.get_labels(), 
                                        montage_type=montage_type, 
                                        conv_dict=eeg_dict.stand1020_to_egi)

        # Mark BAD regions
        EEG.mark_BAD_region(RAW, block=True)
        # Summary of BAD regions (confirm the marking)
        EEG.evaluate_BAD_region(RAW, max_duration=fileTime)

        # Add Stim to RAW
        EEG.make_RAW_stim(RAW, states)

        # Create annotations
        RAW = EEG.make_annotation_MI(RAW, fs,
                                    nBlocks=nBlocks,
                                    trialsPerBlock=trialsPerBlock,
                                    initialSec=initialSec,
                                    stimSec=stimSec,
                                    taskSec=taskSec,
                                    rejectSec=rejectSec,
                                    nSplit=nSplit,
                                    fileTime=fileTime)

        # Here we can save RAW as .fif
        EEG.save_RAW(RAW=RAW, path=file_path+'/', file_name=base_name, label='')

    else: # if clean_bool: 
        # Here we can import a previously saved .fif file
        RAW, montage, fs = EEG.import_file_fif(path=file_path+'/', file_name=base_name + '.fif')
        ch_set = ChannelSet(RAW.info['ch_names'][:RAW.get_data(picks='eeg').shape[0]])
        # This is manually added here cause when you import a RAW .fif file it doesn't know the location of all EGI channels, it's inconvenient
        if montage_type=='EGI_128':
            ch_set = ChannelSet('EGI128_location.txt')
            # make sure this rereference is the same as the one before saving the RAW .fif, it should be
            m = np.array(ch_set.RerefMatrix('TP9 TP10'))
            # Occurs in place
            ch_set = ch_set.spfilt(m)

    # RAW is saved without interpolation performed on it, in this way RAW.info['bads'] still contains the identified bad channels
    # Interpolate BAD channels
    old_ch_bads = RAW.info['bads']
    if not RAW.info['bads'] == []:
        old_ch_bads = EEG.interpolate(RAW)
        # Is any channel bad after interpolation? 
        EEG.make_PREP(RAW, isSNR=True, isDeviation=True, isHfNoise=True, isNanFlat=True, isRansac=True)
        print(f"Currently bad channels: {RAW.info['bads']}")
        print(f'Old bad channels: {old_ch_bads}')

    #RAW.plot()

    # Summary of any region
    #EEG.evaluate_BAD_region(RAW, 'BAD_region')
    #EEG.evaluate_BAD_region(RAW, 'left_1')

    # Spatial filter with exclusion
    signalSLAP, ch_setSLAP = EEG.spatial_filter(sfilt  = 'SLAP', 
                                                ch_set  = ch_set, 
                                                signal  = RAW.get_data(picks='eeg'), 
                                                flag_ch = eeg_dict.ch_face + eeg_dict.ch_forehead, 
                                                verbose = True)

    # Create RAW after spatial filter (RAW_SL)
    RAW_SL = EEG.make_RAW_with_montage(signal=signalSLAP, 
                                       fs=fs, 
                                       ch_names=ch_setSLAP.get_labels(), 
                                       montage_type=montage_type, 
                                       conv_dict=eeg_dict.stand1020_to_egi)

    # Add Stim to RAW_SL
    EEG.make_RAW_stim(RAW_SL, states)

    # Import annotations
    RAW_SL.set_annotations(RAW.annotations)
    #RAW_SL.plot()

    # Create Epochs and PSDs
    events_from_annot, event_dict = mne.events_from_annotations(RAW_SL)

    def epochs_to_psd(RAW=None, fs=None, event_dict=None, label=None, events_from_annot=None, tmin=None, tmax=None, fmin=None, fmax=None, resolution=None, secPerSegment=None, secOverlap=None, nSkip=[]):
        """
        Generate epochs and psds based on pre-generated annotations

        Args: 
            RAW (mne.io.Raw): Raw object containing annotations and EEG signal.
            fs (float): EEG sampling frequency.
            event_dict (dict): Dictionary with annotation names as keys and event id as values.
            label (str): Initial part of labels to assign to each epoch.
            events_from_annot (numpy ndarray): Array containing [duration in samples, /, event id] for all annotations.
            tmin (float): Initial time of an epoch in seconds.
            tmax (float): Final time of an epoch in seconds. tmax - tmin = Length of an epoch in seconds.
            fmin (float): Min frequency to be considered in PSDs.
            fmax (float): Max frequency to be consdiered in PSDs.
            resolution (float): Bin width in frequency space. 
            secPerSegment (float): Length of segments in PSDs Welch method in seconds.
            secOverlap (float): Length of overlap between segments in PSDs Welch method in seconds.
            nSkip (list): List of Epochs within a trial to skip. E.g. [0,3,4]

        Returns: 
            numpy array: Return the PSDs associated to a specific trial.
        """
        # Generate all things
        psds_ = []
        for i in range(1,9):
            if i not in nSkip: 
                try:
                    # Generate Epochs
                    epochs_ = EEG.make_epochs(RAW, 
                                              tmin=tmin, 
                                              tmax=tmax,  
                                              event_id=event_dict[label+f'{i}'], 
                                              events_from_annot=events_from_annot, verbose=False)

                    # Generate PSDs
                    psds_.append(EEG.make_psd(epochs_, fs=fs, 
                                              resolution=resolution, 
                                              tmin=tmin, tmax=tmax, 
                                              fmin=fmin, fmax=fmax, 
                                              nPerSegment=int(secPerSegment * fs), 
                                              nOverlap=int(secOverlap * fs), 
                                              aggregate=True, verbose=False))
                except KeyError:
                    # Print label of Epoch if not found, PSDs also will not exist
                    print(f'{label}{i} not found')
            else: 
                # Print label of Epoch if being skipped
                print(f'Skipping Epoch {i}')
        return np.stack(psds_)

    # Generate PSDs for each type of trial
    nSkip = []
    psds_left = epochs_to_psd(RAW_SL, fs, event_dict, 'left_', events_from_annot, 
                              tmin=tmin, tmax=tmax,
                              fmin=fmin, fmax=fmax, resolution=resolution, 
                              secPerSegment=secPerSegment, secOverlap=secOverlap, 
                              nSkip=nSkip)

    psds_left_rest = epochs_to_psd(RAW_SL, fs, event_dict, 'left_rest_', events_from_annot, 
                                   tmin=tmin, tmax=tmax,
                                   fmin=fmin, fmax=fmax, resolution=resolution, 
                                   secPerSegment=secPerSegment, secOverlap=secOverlap, 
                                   nSkip=nSkip)

    psds_right = epochs_to_psd(RAW_SL, fs, event_dict, 'right_', events_from_annot, 
                               tmin=tmin, tmax=tmax,
                               fmin=fmin, fmax=fmax, resolution=resolution, 
                               secPerSegment=secPerSegment, secOverlap=secOverlap, 
                               nSkip=nSkip)

    psds_right_rest = epochs_to_psd(RAW_SL, fs, event_dict, 'right_rest_', events_from_annot, 
                                    tmin=tmin, tmax=tmax,
                                    fmin=fmin, fmax=fmax, resolution=resolution, 
                                    secPerSegment=secPerSegment, secOverlap=secOverlap, 
                                    nSkip=nSkip)

    # Find all channels on the left and right hemispheres
    isLeft_ch =  [x for x in eeg_dict.ch_central + eeg_dict.ch_parietal if x in EEG.find_ch_left(eeg_dict.ch_location) ]
    isRight_ch = [x for x in eeg_dict.ch_central + eeg_dict.ch_parietal if x in EEG.find_ch_right(eeg_dict.ch_location)]

    # Convert ch_set channels into an array of True of False based on the ones to consider 
    isLeft =  np.array([True if x in isLeft_ch  else False for x in EEG.low(ch_setSLAP.get_labels())])
    isRight = np.array([True if x in isRight_ch else False for x in EEG.low(ch_setSLAP.get_labels())])

    # Plot the electrodes used in the statistical tests
    plt.figure(figsize=(6,6))
    PLOT.show_electrode(eeg_dict.ch_location, EEG.low(list(np.array(ch_setSLAP.get_labels()))), 
                        label=False, color='grey',   alpha_back=0)
    PLOT.show_electrode(eeg_dict.ch_location, EEG.low(list(np.array(ch_setSLAP.get_labels())[isLeft])), 
                        label=True, color='blue',    alpha_back=0)

    PLOT.show_electrode(eeg_dict.ch_location, EEG.low(list(np.array(ch_setSLAP.get_labels())[isRight])), 
                        label=True, color='magenta', alpha_back=0)
    lim = 1.5
    plt.xlim(-lim, lim)
    plt.ylim(-1.05, 1.05)
    plt.xticks([])
    plt.yticks([])
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['bottom'].set_visible(False)
    plt.gca().spines['left'].set_visible(False)
    plt.text(-lim*0.95, 1*1.05, f'Montage', weight='bold')
    split_text = montage_type.split('_')
    plt.text(-lim*0.65, 1*1.05, f'{split_text[0]} {split_text[1]} Channels')
    plt.axvline(0, lw=1, ls='--', color='grey', alpha=0.25)
    # Show the boundary of the head:
    r0 = 1
    theta = np.arange(0, np.pi, 0.01)
    plt.plot(r0 * np.cos(theta), r0 * np.sin(theta), color='black', lw=1)
    plt.plot(r0 * np.cos(theta), -r0 * np.sin(theta), color='black', lw=1)
    # Save the plot
    plt.savefig(f'{path_to_folder}/target_electrodes.png')
    plt.savefig(f'{path_to_folder}/target_electrodes.pdf')
    plt.show()

    # Generate frequency bins in each frequency band
    bins_ticks = np.arange(fmin, fmax+1, int(resolution))

    # Initialize useful things for the statistical test
    N = 2999
    perm_bool = True # Do Permutation test
    boot_bool = True # Do Bootstrap test

    p_left = []
    p_right = []
    labels = []

    plt.figure(figsize=(12,6))
    # LEFT HAND Statistical Test begins
    x = np.vstack([psds_left_rest, psds_left]) #(trial, ch, bin)
    # Generate labels for rest (True) and task (False)
    isTreatment = np.arange(x.shape[0]) < psds_left_rest.shape[0] #(trial,)
    isContrast = isRight

    # Theta frequency band
    STAT = mi.Stats(ch_set=ch_setSLAP, dict_symm=eeg_dict.dict_symm, isContralat=isContrast, bins=bins_ticks, custom_bins='theta')
    r2_left_theta = STAT.DifferenceOfR2(x, isTreatment)
    if perm_bool:
        plt.subplot(231)
        plt.title(r'Open/Close Left', fontsize=12,weight='bold', loc='left')
        p = STAT.ApproxPermutationTest(x=x, isTreatment=isTreatment, stat=STAT.DifferenceOfSumsR2, nSimulations=N, plot=True)
        p_left.append(p)
        labels.append(r'4-7 Hz (P)')
    if boot_bool:
        plt.subplot(234)
        p = STAT.BootstrapTest(x=x, isTreatment=isTreatment, stat=STAT.DifferenceOfSumsR2, nSimulations=N, nullHypothesisStatValue=0.0, plot=True)
        p_left.append(p)
        labels.append(r'4-7 Hz (B)')

    # Alpha frequency band
    STAT = mi.Stats(ch_set=ch_setSLAP, dict_symm=eeg_dict.dict_symm, isContralat=isContrast, bins=bins_ticks, custom_bins='alpha')
    r2_left_alpha = STAT.DifferenceOfR2(x, isTreatment)
    if perm_bool:
        plt.subplot(232)
        p = STAT.ApproxPermutationTest(x=x, isTreatment=isTreatment, stat=STAT.DifferenceOfSumsR2, nSimulations=N, plot=True)
        p_left.append(p)
        labels.append(r'8-12 Hz (P)')
    if boot_bool:
        plt.subplot(235)
        p = STAT.BootstrapTest(x=x, isTreatment=isTreatment, stat=STAT.DifferenceOfSumsR2, nSimulations=N, nullHypothesisStatValue=0.0, plot=True)
        p_left.append(p)
        labels.append(r'8-12 Hz (B)')

    # Beta frequency band
    STAT = mi.Stats(ch_set=ch_setSLAP, dict_symm=eeg_dict.dict_symm, isContralat=isContrast, bins=bins_ticks, custom_bins='beta')
    r2_left_beta = STAT.DifferenceOfR2(x, isTreatment)
    if perm_bool:
        plt.subplot(233)
        p = STAT.ApproxPermutationTest(x=x, isTreatment=isTreatment, stat=STAT.DifferenceOfSumsR2, nSimulations=N, plot=True)
        p_left.append(p)
        labels.append(r'13-30 Hz (P)')
    if boot_bool:
        plt.subplot(236)
        p = STAT.BootstrapTest(x=x, isTreatment=isTreatment, stat=STAT.DifferenceOfSumsR2, nSimulations=N, nullHypothesisStatValue=0.0, plot=True)
        p_left.append(p)
        labels.append(r'13-30 Hz (B)')
    plt.tight_layout()
    plt.savefig(f'{path_to_folder}/test_left_{resolution}_Hz.png')
    plt.savefig(f'{path_to_folder}/test_left_{resolution}_Hz.pdf')
    plt.show()


    plt.figure(figsize=(12,6))
    # RIGHT HAND Statistical Test begins
    x = np.vstack([psds_right_rest, psds_right]) #(trial, ch, bin)
    # Generate labels for rest (True) and task (False)
    isTreatment = np.arange(x.shape[0]) < psds_right_rest.shape[0] #(trial,)
    isContrast = isLeft

    # Theta frequency band
    STAT = mi.Stats(ch_set=ch_setSLAP, dict_symm=eeg_dict.dict_symm, isContralat=isContrast, bins=bins_ticks, custom_bins='theta')
    r2_right_theta = STAT.DifferenceOfR2(x, isTreatment)
    if perm_bool:
        plt.subplot(231)
        plt.title(r'Open/Close Right', fontsize=12, weight='bold', loc='left')
        p = STAT.ApproxPermutationTest(x=x, isTreatment=isTreatment, stat=STAT.DifferenceOfSumsR2, nSimulations=N, plot=True)
        p_right.append(p)
    if boot_bool:
        plt.subplot(234)
        p = STAT.BootstrapTest(x=x, isTreatment=isTreatment, stat=STAT.DifferenceOfSumsR2, nSimulations=N, nullHypothesisStatValue=0.0, plot=True)
        p_right.append(p)

    # Alpha frequency band
    STAT = mi.Stats(ch_set=ch_setSLAP, dict_symm=eeg_dict.dict_symm, isContralat=isContrast, bins=bins_ticks, custom_bins='alpha')
    r2_right_alpha = STAT.DifferenceOfR2(x, isTreatment)
    if perm_bool:
        plt.subplot(232)
        p = STAT.ApproxPermutationTest(x=x, isTreatment=isTreatment, stat=STAT.DifferenceOfSumsR2, nSimulations=N, plot=True)
        p_right.append(p)
    if boot_bool:
        plt.subplot(235)
        p = STAT.BootstrapTest(x=x, isTreatment=isTreatment, stat=STAT.DifferenceOfSumsR2, nSimulations=N, nullHypothesisStatValue=0.0, plot=True)
        p_right.append(p)

    # Beta frequency band
    STAT = mi.Stats(ch_set=ch_setSLAP, dict_symm=eeg_dict.dict_symm, isContralat=isContrast, bins=bins_ticks, custom_bins='beta')
    r2_right_beta = STAT.DifferenceOfR2(x, isTreatment)
    if perm_bool:
        plt.subplot(233)
        p = STAT.ApproxPermutationTest(x=x, isTreatment=isTreatment, stat=STAT.DifferenceOfSumsR2, nSimulations=N, plot=True)
        p_right.append(p)
    if boot_bool:
        plt.subplot(236)
        p = STAT.BootstrapTest(x=x, isTreatment=isTreatment, stat=STAT.DifferenceOfSumsR2, nSimulations=N, nullHypothesisStatValue=0.0, plot=True)
        p_right.append(p)
    plt.tight_layout()
    plt.savefig(f'{path_to_folder}/test_right_{resolution}_Hz.png')
    plt.savefig(f'{path_to_folder}/test_right_{resolution}_Hz.pdf')
    plt.show()


    # Plot p-values extracted by Statistical tests
    # This is a Left - Right view
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 2.5), sharey=True)
    y_max = 0.5
    y_min = -0.5
    y = np.linspace(start=y_max, stop=y_min, num=len(p_left)+2)
    y = y[1:-1]
    deltay = 0.075
    # Left results
    #---------
    x_values = []
    xUp_values = []
    xDown_values = []
    for p in p_left:
        p_down, p, p_up = STAT.pvalue_interval(p, N+1)
        xUp_values.append(STAT.negP(p_up))
        x_values.append(STAT.negP(p))
        xDown_values.append(STAT.negP(p_down))
    #---------
    ax1.set_title('Open/Close Left', fontsize=12, loc='left', weight='bold')
    ax1.scatter(x_values, y, color='red', marker='o')
    # Add confidence interval on true p
    for i in range(len(y)):
        ax1.fill_betweenx([y[i]-deltay, y[i]+deltay], xDown_values[i], xUp_values[i], color='red', alpha=0.3)
    ax1.hlines(y, 0, x_values, colors='red', lw=1, alpha=0.25)
    ax1.set_xlabel(r'-log(p)')
    ax1.set_xlim(ax1.get_xlim()[::-1])  # Reverse the x-axis for left plot
    ax1.set_xlim(right=0, left=6)
    ax1.set_ylim(y_min, y_max)
    ax1.axvline(STAT.negP(0.05), color='black', lw=1, ls=':', alpha=0.5, label='95% C.L.')
    ax1.axvline(STAT.negP(0.01), color='darkviolet', lw=1, ls=':', alpha=0.5, label='99% C.L.')
    ax1.set_yticks(y)
    ax1.set_yticklabels(labels)
    ax1.legend(loc='upper left')
    # Right results
    #---------
    x_values = []
    xUp_values = []
    xDown_values = []
    for p in p_right:
        p_down, p, p_up = STAT.pvalue_interval(p, N+1)
        xUp_values.append(STAT.negP(p_up))
        x_values.append(STAT.negP(p))
        xDown_values.append(STAT.negP(p_down))
    #---------
    ax2.set_title('Open/Close Right', fontsize=12, loc='right', weight='bold')
    ax2.scatter(x_values, y, color='blue', marker='o')
    # Add confidence interval on true p
    for i in range(len(y)):
        ax2.fill_betweenx([y[i]-deltay, y[i]+deltay], xDown_values[i], xUp_values[i], color='blue', alpha=0.3)
    ax2.hlines(y, 0, x_values, colors='blue', lw=1, alpha=0.25)
    ax2.set_xlabel(r'-log(p)')
    ax2.set_xlim(left=0, right=6)
    ax1.set_ylim(y_min, y_max)
    ax2.axvline(STAT.negP(0.05), color='black', lw=1, ls=':', alpha=0.5)
    ax2.axvline(STAT.negP(0.01), color='darkviolet', lw=1, ls=':', alpha=0.5)
    #--------- 
    plt.subplots_adjust(wspace=0)  # Adjust space between subplots
    plt.savefig(f'{path_to_folder}/pVal_{resolution}_Hz.png')
    plt.savefig(f'{path_to_folder}/pVal_{resolution}_Hz.pdf')
    plt.show()
    

    # Plot topoplots with r2
    # Identify interpolated channels to show on the topomap
    mask = np.array([True if x in old_ch_bads else False for x in ch_setSLAP.get_labels()])
    mask_params1 = dict(marker='X', markersize=7, markerfacecolor='black')
    # Identify target channels to show on the topomap
    mask_right = np.array([True if x in isLeft_ch else False for x in ch_setSLAP.get_labels()])
    mask_left = np.array([True if x in isRight_ch else False for x in ch_setSLAP.get_labels()])
    mask_params2 = dict(marker='o', markersize=4, markerfacecolor='lime', alpha=0.75)
    # Plot
    fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(6, 6))
    # Make colormap for topoplots
    custom_cmap = PLOT.make_simple_cmap('blue', 'white', 'red')
    # Plot topoplots - Theta
    PLOT.plot_topomap_L_R([axes[0,0],axes[0,1],axes[0,2]], RAW_SL, r2_left_theta, r2_right_theta, custom_cmap, (-1,1), [mask,mask_left,mask_right], [mask_params1,mask_params2], '4-7 Hz', True)
    # Plot topoplots - Alpha
    PLOT.plot_topomap_L_R([axes[1,0],axes[1,1],axes[1,2]], RAW_SL, r2_left_alpha, r2_right_alpha, custom_cmap, (-1,1), [mask,mask_left,mask_right], [mask_params1,mask_params2], '8-12 Hz', False)
    # Plot topoplots - Beta
    PLOT.plot_topomap_L_R([axes[2,0],axes[2,1],axes[2,2]], RAW_SL, r2_left_beta, r2_right_beta, custom_cmap, (-1,1), [mask,mask_left,mask_right], [mask_params1,mask_params2], '13-30 Hz', False)
    fig.tight_layout()
    plt.savefig(f'{path_to_folder}/topoR2_{resolution}_Hz.png')
    plt.savefig(f'{path_to_folder}/topoR2_{resolution}_Hz.pdf')
    plt.show()