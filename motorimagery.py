#!/usr/bin/env -S python  #


if __name__ == '__main__':
    import argparse 
    parser = argparse.ArgumentParser( description=__doc__, formatter_class=argparse.RawTextHelpFormatter )
    parser.add_argument( '-c',     metavar='cleaned',    type=bool,  default=False,           help="has the file been previously cleaned? [True, False]")
    parser.add_argument( '-f',     metavar='file_path',  type=str,   default='',              help="path to the file to run scipt on")
    parser.add_argument( '-r',     metavar='resolution', type=int,   default=1,               help="resolution for PSDs [1, 2]")
    parser.add_argument( '-fmin',  metavar='fmin',       type=float, default=1.,              help="min frequency to consider")
    parser.add_argument( '-fmax',  metavar='fmax',       type=float, default=40.,             help="max frequency to consider")
    parser.add_argument( '-fband', metavar='fband',      type=list,  default=[4.,8.,13.,31.], help="frequency band sections of interest")
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

import plot_style
plot_style.set_plot_style()


if __name__ == '__main__':
    EEG = mi.EEG() # Initialize EEG tools
    PLOT = mi.Plotting() # Initialize Plotting tools

    # Handle the input options
    clean_bool = opts.c
    file_path, file_name = os.path.split(opts.f)
    resolution = opts.r
    secPerSegment = 1/resolution
    secOverlap = secPerSegment/2
    fmin = opts.fmin
    fmax = opts.fmax
    freq_band = opts.fband
    
    ch_location = eeg_dict.ch_location

    # Extract base name from file
    base_name, extension = os.path.splitext(file_name)

    # Create a folder using base name, if folder doesn't exist 
    path_to_folder = EEG.create_folder(path=file_path, folder_name=base_name)


    # Import .dat file
    signal, states, fs, ch_names, blockSize, montage_type = EEG.import_file_dat(file_path, file_name)

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

        # Create unfiltered RAW for PREP
        RAW = EEG.make_RAW_with_montage(signal=signal * 1e-6, 
                                        fs=fs, 
                                        ch_names=ch_set.get_labels(), 
                                        montage_type=montage_type, 
                                        conv_dict=eeg_dict.stand1020_to_egi)

        # Run PREP for bad channels
        EEG.make_PREP(RAW, isSNR=False, isDeviation=False, isHfNoise=True, isNanFlat=True, isRansac=False)

        # Bandpass filter
        signalFilter = EEG.filter_data(signal, fs, l_freq=fmin, h_freq=fmax)

        # Re-reference
        if montage_type == 'EGI_128':
            # Re-reference to mastoids
            signalFilter, ch_set = EEG.spatial_filter(sfilt='REF', 
                                                      ch_set=ch_set, 
                                                      signal=signalFilter, 
                                                      flag_ch='tp9 1p10', 
                                                      verbose=True)

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # PLOT ChannelSet after potential re-reference
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        plt.figure(figsize=(7, 6))
        ch_set.plot()
        plt.text(-1.25, 1.05, f'Montage', weight='bold')
        split_text = montage_type.split('_')
        plt.text(-1.25, 0.95, f'{split_text[0]} {split_text[1]} Channels')
        plt.tight_layout()
        # Save the plot
        plt.savefig(f'{path_to_folder}/montage.png', bbox_inches='tight')
        plt.savefig(f'{path_to_folder}/montage.pdf', bbox_inches='tight')
        plt.show()



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
        EEG.save_RAW(RAW=RAW, path=file_path, file_name=base_name, label='')

    else: # if clean_bool: 
        # Here we can import a previously saved .fif file
        RAW, montage, fs = EEG.import_file_fif(path=file_path, file_name=base_name + '.fif')
        #RAW.info['ch_names'] = [x.lower() for x in RAW.info['ch_names']]
        ch_set = ChannelSet([x.lower() for x in RAW.info['ch_names'][:RAW.get_data(picks='eeg').shape[0]]])
        # This is manually added here cause when you import a RAW .fif file it doesn't know the location of all EGI channels, it's inconvenient
        if montage_type=='EGI_128':
            ch_set = ChannelSet('EGI128_location.txt')
            # make sure this rereference is the same as the one before saving the RAW .fif, it should be
            m = np.array(ch_set.RerefMatrix('tp9 tp10'))
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
    isLeft_ch =  [x for x in eeg_dict.ch_central + eeg_dict.ch_parietal + eeg_dict.ch_frontal if x in EEG.find_ch_left(eeg_dict.ch_location)]
    isRight_ch = [x for x in eeg_dict.ch_central + eeg_dict.ch_parietal + eeg_dict.ch_frontal if x in EEG.find_ch_right(eeg_dict.ch_location)]

    # Convert ch_set channels into an array of True of False based on the ones to consider 
    isLeft =  np.array([True if x in isLeft_ch  else False for x in EEG.low(ch_setSLAP.get_labels())])
    isRight = np.array([True if x in isRight_ch else False for x in EEG.low(ch_setSLAP.get_labels())])

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # PLOT the electrodes used in the statistical tests
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    plt.figure(figsize=(6,4))
    # All channels
    PLOT.show_electrode(eeg_dict.ch_location, EEG.low(list(np.array(ch_setSLAP.get_labels()))), 
                        label=False, color='grey',   alpha_back=0, marker='o')
    # Left channels
    PLOT.show_electrode(eeg_dict.ch_location, EEG.low(list(np.array(ch_setSLAP.get_labels())[isLeft])), 
                        label=True, color='blue',    alpha_back=0, marker='o')
    # Right channels
    PLOT.show_electrode(eeg_dict.ch_location, EEG.low(list(np.array(ch_setSLAP.get_labels())[isRight])), 
                        label=True, color='magenta', alpha_back=0, marker='o')
    # Interpolated channels
    PLOT.show_electrode(eeg_dict.ch_location, EEG.low(old_ch_bads), 
                        label=False, color='lime', alpha_back=0, marker='x')
    lim = 1.8
    plt.xlim(-lim, lim)
    plt.ylim(-1.3, 1.3)
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
    r0 = 1.05
    theta = np.arange(0, np.pi, 0.01)
    plt.plot(r0 * np.cos(theta), r0 * np.sin(theta), color='black', lw=1)
    plt.plot(r0 * np.cos(theta), -r0 * np.sin(theta), color='black', lw=1)
    # Save the plot
    plt.savefig(f'{path_to_folder}/target_electrodes.png')
    plt.savefig(f'{path_to_folder}/target_electrodes.pdf')
    plt.show()


    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # STATISTICAL TESTS
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Generate frequency bins in each frequency band
    bins_ticks = np.arange(fmin, fmax+1, int(resolution))

    # Number of frequency bands to consider
    N = len(freq_band)-1

    # Initialize useful things for the statistical test
    nSim = 2999
    perm_bool = True # Do Permutation test
    boot_bool = True # Do Bootstrap test

    r2_left = []
    p_left = []
    labels = []

    # LEFT HAND Statistical Test begins
    fig, axs = plt.subplots(nrows=2, ncols=N, figsize=(int(4*N),6))
    # Generate array of trials, rest followed by task
    x = np.vstack([psds_left_rest, psds_left]) #(trial, ch, bin)
    # Generate labels for rest (True) and task (False)
    isTreatment = np.arange(x.shape[0]) < psds_left_rest.shape[0] #(trial,)
    isContrast = isRight

    for i in range(N):
        STAT = mi.Stats(ch_set=ch_setSLAP, dict_symm=eeg_dict.dict_symm, isContralat=isContrast, bins=bins_ticks, custom_bins=[freq_band[i], freq_band[i+1]])
        r2_left.append(STAT.DifferenceOfR2(x, isTreatment))
        if perm_bool:
            if i == 0: axs[0,i].set_title(r'Open/Close Left', fontsize=12,weight='bold', loc='left')
            axs[0,i].set_title(f'[{freq_band[i]}-{freq_band[i+1]-1}] Hz', fontsize=12, loc='right')
            p = STAT.ApproxPermutationTest(x=x, isTreatment=isTreatment, stat=STAT.DifferenceOfSumsR2, nSimulations=nSim, plot=True, ax=axs[0,i])
            p_left.append(p)
            labels.append(f'{freq_band[i]}-{freq_band[i+1]-1} Hz (P)')
        if boot_bool:
            p = STAT.BootstrapTest(x=x, isTreatment=isTreatment, stat=STAT.DifferenceOfSumsR2, nSimulations=nSim, nullHypothesisStatValue=0.0, plot=True, ax=axs[1,i])
            p_left.append(p)
            labels.append(f'{freq_band[i]}-{freq_band[i+1]-1} Hz (B)')
    plt.tight_layout()
    plt.savefig(f'{path_to_folder}/test_left_{resolution}_Hz.png')
    plt.savefig(f'{path_to_folder}/test_left_{resolution}_Hz.pdf')
    plt.show()


    p_right = []
    r2_right = []

    # RIGHT HAND Statistical Test begins
    fig, axs = plt.subplots(nrows=2, ncols=N, figsize=(int(4*N),6))
    # Generate array of trials, rest followed by task
    x = np.vstack([psds_right_rest, psds_right]) #(trial, ch, bin)
    # Generate labels for rest (True) and task (False)
    isTreatment = np.arange(x.shape[0]) < psds_right_rest.shape[0] #(trial,)
    isContrast = isLeft

    for i in range(N):
        STAT = mi.Stats(ch_set=ch_setSLAP, dict_symm=eeg_dict.dict_symm, isContralat=isContrast, bins=bins_ticks, custom_bins=[freq_band[i], freq_band[i+1]])
        r2_right.append(STAT.DifferenceOfR2(x, isTreatment))
        if perm_bool:
            if i == 0: axs[0,i].set_title(r'Open/Close Right', fontsize=12,weight='bold', loc='left')
            axs[0,i].set_title(f'[{freq_band[i]}-{freq_band[i+1]-1}] Hz', fontsize=12, loc='right')
            p = STAT.ApproxPermutationTest(x=x, isTreatment=isTreatment, stat=STAT.DifferenceOfSumsR2, nSimulations=nSim, plot=True, ax=axs[0,i])
            p_right.append(p)
        if boot_bool:
            p = STAT.BootstrapTest(x=x, isTreatment=isTreatment, stat=STAT.DifferenceOfSumsR2, nSimulations=nSim, nullHypothesisStatValue=0.0, plot=True, ax=axs[1,i])
            p_right.append(p)
    plt.tight_layout()
    plt.savefig(f'{path_to_folder}/test_right_{resolution}_Hz.png')
    plt.savefig(f'{path_to_folder}/test_right_{resolution}_Hz.pdf')
    plt.show()


    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # PLOT p-values extracted by Statistical tests
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # This is a Left - Right view
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3.5), sharey=True)
    y_max = 0.5
    y_min = -0.5
    y = np.linspace(start=y_max, stop=y_min, num=len(p_left)+2)
    y = y[1:-1]
    deltay = 0.05
    # Left results
    #---------
    x_values = []
    xUp_values = []
    xDown_values = []
    for p in p_left:
        p_down, p, p_up = STAT.pvalue_interval(p, nSim+1)
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
        p_down, p, p_up = STAT.pvalue_interval(p, nSim+1)
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
    plt.savefig(f'{path_to_folder}/pVal_{resolution}_Hz.png', bbox_inches='tight')
    plt.savefig(f'{path_to_folder}/pVal_{resolution}_Hz.pdf', bbox_inches='tight')
    plt.show()
    
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # PLOT topoplots with r2
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Identify interpolated channels to show on the topomap
    mask = np.array([True if x in old_ch_bads else False for x in ch_setSLAP.get_labels()])
    mask_params1 = dict(marker='X', markersize=6, markerfacecolor='black')
    # Identify target channels to show on the topomap
    mask_right = np.array([True if x in isLeft_ch else False for x in ch_setSLAP.get_labels()])
    mask_left = np.array([True if x in isRight_ch else False for x in ch_setSLAP.get_labels()])
    mask_params2 = dict(marker='o', markersize=5, markerfacecolor='lime', alpha=0.75)
    # Plot
    fig, axes = plt.subplots(nrows=N, ncols=3, figsize=(6, int(2*N)))
    # Make colormap for topoplots
    custom_cmap = PLOT.make_simple_cmap('blue', 'white', 'red')

    for i in range(N):
        if i==0: 
            PLOT.plot_topomap_L_R([axes[i,0],axes[i,1],axes[i,2]], RAW_SL, r2_left[i], r2_right[i], custom_cmap, (-1,1), 
                                  [mask,mask_left,mask_right], [mask_params1,mask_params2], text=True)
        else: 
            PLOT.plot_topomap_L_R([axes[i,0],axes[i,1],axes[i,2]], RAW_SL, r2_left[i], r2_right[i], custom_cmap, (-1,1), 
                                  [mask,mask_left,mask_right], [mask_params1,mask_params2], text=False)
        
        if p_left[1::2][i] < 0.05:
            axes[i,0].set_title(f'*{freq_band[i]}-{freq_band[i+1]-1} Hz (Left)', fontsize=12)
        else: 
            axes[i,0].set_title(f'{freq_band[i]}-{freq_band[i+1]-1} Hz (Left)', fontsize=12)
            
        if p_right[1::2][i] < 0.05:
            axes[i,1].set_title(f'*{freq_band[i]}-{freq_band[i+1]-1} Hz (Right)', fontsize=12)
        else: 
            axes[i,1].set_title(f'{freq_band[i]}-{freq_band[i+1]-1} Hz (Right)', fontsize=12)

    fig.tight_layout()
    plt.savefig(f'{path_to_folder}/topoR2_{resolution}_Hz.png')
    plt.savefig(f'{path_to_folder}/topoR2_{resolution}_Hz.pdf')
    plt.show()


    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # PLOT channel x-ray
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Generate array of trials, rest followed by task
    x_list = [np.vstack([psds_left_rest, psds_left]),       #(trial, ch, bin)]
              np.vstack([psds_right_rest, psds_right])]     #(trial, ch, bin)]
    # Generate labels for rest (True) and task (False)
    isTreatment_list = [np.arange(x_list[0].shape[0]) < psds_left_rest.shape[0],  #(trial,)
                        np.arange(x_list[1].shape[0]) < psds_right_rest.shape[0]] #(trial,)

    # List of channels to show plots for
    chs = ['c4', 'c3']
    # Find symmetric channel id
    #ch_symm = EEG.find_ch_symmetry(ch_location=eeg_dict.ch_location, ch_list=[ch.lower()])[ch.lower()]
    #ch_symm_idx = ch_set.find_labels(ch_symm)[0]

    for ch in chs:
        # Find channel id
        ch_idx = ch_set.find_labels(ch)[0]
        
        # xlim to use in plots based on frequency range used
        xlim = [STAT.bins[0], STAT.bins[-1]]

        # Masks for target electrodes
        mask = np.array([True if x==ch else False for x in ch_setSLAP.get_labels()])
        mask_params = dict(marker='o', markersize=6, markerfacecolor='lime')

        # Define mid points and width of frequency bands to consider
        x_band = []
        w_band = []
        for i in range(N):
            # Frequency bands mid points
            x_band.append(freq_band[i] + (freq_band[i+1] - freq_band[i]) / 2)
            # Frequency bands width
            w_band.append( freq_band[i+1] - freq_band[i])
            
        # Generate plots for Left and Right trials
        for k, mode in enumerate(['Left', 'Right']):
            # Generate the canvases
            figTopo, axsTopo = plt.subplots(1, N+1, figsize=(int(2*N+1), 2)) # Canvas for topoplots
            figPSD,  axsPSD  = plt.subplots(3, 1,   figsize=(6, int(2*(3)))) # Canvas for PSDs
            figCorr, axsCorr = plt.subplots(N, 1,   figsize=(4, int(2*N)))   # Canvas for correlation coeffs

            # Define which trials to consider
            x = x_list[k]
            isTreatment = isTreatment_list[k]

            # Rest PSD trials
            xs, y_rest, ws = PLOT.plot_psd_at_channel(x=x[ :, [ch_idx], : ][ isTreatment], color='navy', ax=axsPSD[0], freq_band=freq_band, bins=STAT.bins)
            # Use Rest PSD trials to define y limits in dB
            ylim = (int(np.min(y_rest) - np.abs(np.min(y_rest)) - 15), 
                    int(np.max(y_rest) + np.abs(np.max(y_rest)) + 15))\
            # Plot (Rest PSD trials)
            PLOT.plot_frequency_bands(ax=axsPSD[0], ylim=ylim, fontsize=10)
            axsPSD[0].set_xticks([])
            axsPSD[0].set_xlim(xlim)
            axsPSD[0].set_ylim(ylim)
            axsPSD[0].set_ylabel('[dB]', loc='top', fontsize=11)
            axsPSD[0].text(xlim[0]+0.25, ylim[0] + (ylim[1]-ylim[0])*0.025, 'Rest Trials')
        
            # Lists to store information for each frequency band
            ys = []
            r_coeffs = []
            cmap_max=1

            # Generate plots for each frequency band
            for i in range(N):
                idx = np.where((STAT.bins >= freq_band[i]) & (STAT.bins < freq_band[i+1]))[0]
                start_idx = idx[0]
                end_idx = idx[-1]
                x_within_band = np.mean(x[:, :, start_idx:end_idx], axis=2) #(trial, ch)
                x_within_band = EEG.convert_dB(x_within_band) # Transform PSDs to dB
                r_coeff = STAT.CalculateR(x=x_within_band, isTreatment=isTreatment)
                r_coeffs.append(r_coeff)
                ys.append(r_coeff[ [ch_idx] ])

                # Band r topoplot
                mne.viz.plot_topomap(r_coeff, RAW_SL.info, ch_type='eeg', sensors=True, cmap=PLOT.simple_cmap, vlim=(-cmap_max,cmap_max), 
                                     mask=mask, mask_params=mask_params, show=False, axes=axsTopo[i])
                
                # Convert labels of trials into dummy variable (True/Rest = 1, False/Task = 0 by choice)
                dummy = STAT.Dummy(isTreatment)
                PLOT.plot_correlation_psd_groups(x=x_within_band[ :, [ch_idx] ], y=dummy, isTreatment=isTreatment, r=r_coeff[ [ch_idx] ], xlim=ylim, ax=axsCorr[i])
                
                # Show frequency band (topoplot)
                axsTopo[i].set_title(f'[{freq_band[i]}, {freq_band[i+1]-1}] Hz', fontsize=12)
                
                # Show frequency band (correlation)
                axsCorr[i].text(ylim[-1] + np.abs(ylim[-1])*0.05, 1.05, f'[{freq_band[i]}, {freq_band[i+1]-1}] Hz', fontsize=12)
                axsCorr[i].set_xlim(ylim)
                if i != N-1: 
                    axsCorr[i].set_xticks([])
                    
            # Task PSD trials
            xs, y_task, ws = PLOT.plot_psd_at_channel(x=x[ :, [ch_idx], : ][~isTreatment], color='green', ax=axsPSD[1], freq_band=freq_band, bins=STAT.bins)
            # Plot (Task PSD trials)
            PLOT.plot_frequency_bands(ax=axsPSD[1], ylim=None)
            axsPSD[1].set_xticks([])
            axsPSD[1].set_xlim(xlim)
            axsPSD[1].set_ylim(ylim)
            axsPSD[1].set_ylabel('[dB]', loc='top', fontsize=11)
            axsPSD[1].text(xlim[0]+0.25, ylim[0] + (ylim[1]-ylim[0])*0.025, 'Task Trials')

            # Rest - Task PSD (mean) trials
            y_diff = [np.mean(i)-np.mean(j) for i,j in zip(y_rest, y_task)]
            ymax = int(np.max(np.abs(y_diff)) * 2)
            # Plot (Rest - Task PSD (mean) trials)
            for i,y in enumerate(y_diff):
                if i==0: 
                    axsPSD[2].plot([x_band[i]-w_band[i]/2, x_band[i]+w_band[i]/2], [y,y], color='blue', label='Rest - Task')
                else:
                    axsPSD[2].plot([x_band[i]-w_band[i]/2, x_band[i]+w_band[i]/2], [y,y], color='blue')
            axsPSD[2].axhline(0, lw=1, ls='--', color='grey')
            axsPSD[2].set_xlim(xlim)
            axsPSD[2].set_xlabel('$f$ [Hz]', loc='right', fontsize=11)
            ylim_diff = np.max(np.abs(y_diff))*1.5
            axsPSD[2].set_ylim(-ylim_diff, ylim_diff)
            axsPSD[2].set_ylabel('Difference [dB]', fontsize=11, loc='top')
            PLOT.plot_frequency_bands(ax=axsPSD[2], ylim=None)
            # Add r2 on top of PSD difference
            ys =  [x * abs(x) * 10 for x in ys]
            color='magenta'
            axsPSD[2].plot(x_band, ys, '-o', color=color, markersize=4, label=r'r$^{2}$ [$\times$10]')
            axsPSD[2].scatter(x_band, ys, color='black', s=60)
            axsPSD[2].legend(loc='lower right', frameon=True)
            # Add new axis on the right
            #ax = axsPSD[2].twinx()  # instantiate a second axes that shares the same x-axis
            #ax.set_ylabel(r'[$\times$10]  r$^{2}$', color=color)  # we already handled the x-label with ax1
            #ax.tick_params(axis='y', labelcolor=color, color=color)
            #ax.set_ylim(-1,1)

            # Show channel (correlation)
            axsCorr[0].text(ylim[0], 1.35, 'Channel', weight='bold', fontsize=12)
            axsCorr[0].text(ylim[0] + np.abs(ylim[1]-ylim[0])*0.2, 1.35, f'{ch}',fontsize=12)
            axsCorr[0].set_title(r'Coeff. r & r$^{2}$', fontsize=11, loc='right')
            
            # Show channel (PSD)
            axsPSD[0].text(xlim[0], ylim[-1] + np.abs(ylim[-1])*0.05, f'Channel', weight='bold', fontsize=12)
            axsPSD[0].text(xlim[0] + 5.5, ylim[-1] + np.abs(ylim[-1])*0.05, f'{ch}', fontsize=12)
            axsPSD[0].set_title('Trial PSD', fontsize=11, loc='right')
            
            # Add color bar (topoplot)
            clim = dict(kind='value', lims=[-cmap_max,0,cmap_max])
            divider = make_axes_locatable(axsTopo[-1])
            axsTopo[-1].set_yticks([])
            axsTopo[-1].set_xticks([])
            axsTopo[-1].axis('off')
            axsTopo[-1] = divider.append_axes(position='left', size='20%', pad=0.5)
            mne.viz.plot_brain_colorbar(axsTopo[-1], clim=clim, colormap=PLOT.simple_cmap, transparent=False, orientation='vertical', label=None)   
            axsTopo[-1].text(4,  0.5, r'$r$ Coefficients', fontsize=12)
            axsTopo[-1].text(4,  0.3, f'Channel', weight='bold', fontsize=12)
            axsTopo[-1].text(10, 0.3, f'{ch}', fontsize=12)
            axsTopo[-1].text(4,  0.1, f'{mode} trials', fontsize=12)
            
            figCorr.subplots_adjust(hspace=0)
            figPSD.subplots_adjust(hspace=0)

            figTopo.savefig(f'{path_to_folder}/topoR_{ch}_{mode}_{resolution}_Hz.png', bbox_inches='tight')
            figTopo.savefig(f'{path_to_folder}/topoR_{ch}_{mode}_{resolution}_Hz.pdf', bbox_inches='tight')

            figPSD.savefig(f'{path_to_folder}/psd_{ch}_{mode}_{resolution}_Hz.png', bbox_inches='tight')
            figPSD.savefig(f'{path_to_folder}/psd_{ch}_{mode}_{resolution}_Hz.pdf', bbox_inches='tight')

            figCorr.savefig(f'{path_to_folder}/corr_{ch}_{mode}_{resolution}_Hz.png', bbox_inches='tight')
            figCorr.savefig(f'{path_to_folder}/corr_{ch}_{mode}_{resolution}_Hz.pdf', bbox_inches='tight')

            figTopo.show()
            figPSD.show()
            figCorr.show()
