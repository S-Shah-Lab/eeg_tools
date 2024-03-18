#!/usr/bin/env -S python  #


if __name__ == '__main__':
    import argparse 
    parser = argparse.ArgumentParser( description=__doc__, formatter_class=argparse.RawTextHelpFormatter )
    parser.add_argument( '-m', metavar='montage_type', type=str,   default='EGI_128', help="eeg montage used during data aquisition ['DSI_24', 'EGI_128', 'GTEC_32']")
    parser.add_argument( '-c', metavar='cleaned',      type=bool,  default=False,     help="has the file been previously cleaned? [True, False]")
    parser.add_argument( '-f', metavar='file_path',    type=str,   default='',        help="path to the file to run scipt on")
    parser.add_argument( '-r', metavar='resolution',   type=int,   default='1',         help="resolution for PSDs [1, 2]")
    opts = parser.parse_args()

import os
import numpy as np
from BCI2000Tools.FileReader import bcistream
from BCI2000Tools.Electrodes import *
from BCI2000Tools.Plotting import *
import mne 
import matplotlib.pyplot as plt
from pyprep.prep_pipeline import PrepPipeline, NoisyChannels

import eeg_dict # Contains dictionaries and libraries for electrodes locations 
import tools_mi as mi # Contains tools for eeg and motor imagery


if __name__ == '__main__':
    MI = mi.tools_mi()

    montage_type = opts.m
    clean_bool = opts.c
    file_path, file_name = os.path.split(opts.f)
    resolution = opts.r

    if montage_type == 'DSI_24': ch_info = 'DSI24_location.txt'
    elif montage_type == 'EGI_128': ch_info = 'EGI128_location.txt'
    # elif montage_type == 'GTEC_32': ch_info = 'GTEC32_location.txt'

    ch_location = eeg_dict.ch_location

    fmin = 1
    fmax = 40

    # Extract base name from file
    base_name, extension = os.path.splitext(file_name)

    # Create a folder using base name, if folder doesn't exist 
    MI.create_folder(path=file_path+'/', folder_name=base_name)


    # Import .dat file
    signal, states, fs, ch_names, fileTime, blockSize = MI.import_file_dat(file_path+'/', file_name, montage_type)

    nBlocks = 8
    trialsPerBlock = 4
    nSplit = 6
    
    initialSec = 2
    stimSec = 3
    taskSec = 10
    rejectSec = taskSec - 9 # 1 [s]

    initialSec = initialSec - ( (initialSec * fs) % blockSize ) / fs # 1.92 [s]
    stimSec = stimSec - ( (stimSec * fs) % blockSize ) / fs # 2.944 [s]
    taskSec = taskSec - ( (taskSec * fs) % blockSize ) / fs # 9.984 [s]
    rejectSec = taskSec - 9 # 0.984 [s]

    '''
    if montage_type == 'DSI_24' or montage_type == 'EGI_128': 
        nBlocks = 8
        initialSec = 2
        stimSec = 3
        taskSec = 10
        rejectSec = 1
    elif montage_type == 'GTEC_32':
        nBlocks = 8
        initialSec = 1.92
        stimSec = 2.944
        taskSec = 9.984
        rejectSec = 0.984
    '''

    tmin = 0
    twindow = (taskSec - rejectSec) # e.g. 9
    tmax = twindow/nSplit # e.g. 1.5


    # File has not been cleaned before, it's a new file
    if not clean_bool:
        # Define initial ChannelSet
        ch_set = ChannelSet(ch_info)

        # Bandpass filter
        signalFilter = MI.filter_data(signal, fs, l_freq=1, h_freq=40)

        # Re-reference
        if montage_type == 'EGI_128':
            # Re-reference to mastoids
            signalFilter, ch_set = MI.spatial_filter(sfilt='REF', 
                                                     ch_set=ch_set, 
                                                     signal=signalFilter, 
                                                     flag_ch='TP9 TP10', 
                                                     verbose=True)

        # Create RAW
        RAW = MI.make_RAW(signalFilter * 1e-6, fs, ch_set.get_labels())

        # Run PREP for bad channels
        MI.make_PREP(RAW, isSNR=True, isDeviation=True, isHfNoise=True, isNanFlat=True)

        # Mark BAD regions
        MI.mark_BAD_region(RAW, block=True)

        # Summary of BAD regions (confirm the marking)
        MI.evaluate_BAD_region(RAW, max_duration=fileTime)

        # Add Stim to RAW
        MI.make_RAW_stim(RAW, states)

        # Create annotations
        RAW = MI.make_annotation_MI(RAW, fs,
                                    nBlocks=nBlocks,
                                    trialsPerBlock=trialsPerBlock,
                                    initialSec=initialSec,
                                    stimSec=stimSec,
                                    taskSec=taskSec,
                                    rejectSec=rejectSec,
                                    nSplit=nSplit,
                                    fileTime=fileTime)

        # Summary of any region
        #MI.evaluate_BAD_region(RAW, 'BAD_region')
        #MI.evaluate_BAD_region(RAW, 'left_1')

        # Create montage based on channels to show
        #chsetRef.plot()
        if montage_type=='DSI_24' or montage_type=='GTEC_32':
            montage = MI.make_montage(montage_type=montage_type, 
                                      ch_to_show=ch_set.get_labels(), 
                                      conv_dict=None)

        elif montage_type=='EGI_128':
            montage = MI.make_montage(montage_type=montage_type, 
                                      ch_to_show=ch_set.get_labels(), 
                                      conv_dict=eeg_dict.stand1020_to_egi)

        # Assign montage to RAW
        RAW.set_montage(montage)
        #ch_set.plot()

        # Here we can save RAW as .fif
        MI.save_RAW(RAW=RAW, path=file_path+'/', file_name=base_name, label='')

    else: # if clean_bool: 
        # Here we can import a previously saved .fif file
        RAW, montage, fs = MI.import_file_fif(path=file_path+'/', file_name=base_name + '.fif')
        ch_set = ChannelSet(RAW.info['ch_names'][:RAW.get_data(picks='eeg').shape[0]])

    # Interpolate BAD channels
    old_ch_bads = RAW.info['bads']
    if not RAW.info['bads'] == []:
        old_ch_bads = MI.interpolate(RAW)


    # Spatial filter with exclusion
    signalSLAP, ch_setSLAP = MI.spatial_filter(sfilt= 'SLAP', 
                                               ch_set= ch_set, 
                                               signal= RAW.get_data(picks='eeg'), 
                                               flag_ch= eeg_dict.ch_face + eeg_dict.ch_forehead, 
                                               verbose= True)


    # Create RAW after spatial filter
    RAW_SL = MI.make_RAW(signal=signalSLAP, fs=RAW.info['sfreq'], ch_names=ch_setSLAP.get_labels())
    MI.make_RAW_stim(RAW_SL, states)
    RAW_SL.set_annotations(RAW.annotations)
    if montage_type=='DSI_24' or montage_type=='GTEC_32':
        montage = MI.make_montage(montage_type=montage_type, 
                                  ch_to_show=ch_setSLAP.get_labels(), 
                                  conv_dict=None)

    elif montage_type=='EGI_128':
        montage = MI.make_montage(montage_type=montage_type, 
                                  ch_to_show=ch_setSLAP.get_labels(), 
                                  conv_dict=eeg_dict.stand1020_to_egi)
    # Assign montage to RAW_SL
    RAW_SL.set_montage(montage)
    #RAW_SL.plot()


    # Create Epochs and PSDs
    events_from_annot, event_dict = mne.events_from_annotations(RAW_SL)

    def epochs_to_psd(RAW=None, fs=None, event_dict=None, label=None, events_from_annot=None, tmin=None, tmax=None, twindow=None, fmin=None, fmax=None, resolution=None, secPerSegment=1, secOverlap=0.5, nSkip=0):
        psds_ = []
        for i in range(1,9):
            if i>nSkip: 
                try:
                    epochs_ = MI.make_epochs(RAW, 
                                             tmin=tmin, 
                                             tmax=tmax, 
                                             twindow=twindow, 
                                             event_id=event_dict[label+f'{i}'], 
                                             events_from_annot=events_from_annot, verbose=False)

                    psds_.append(MI.make_psd(epochs_, fs=fs, 
                                             resolution=resolution, 
                                             tmin=tmin, tmax=tmax, 
                                             fmin=fmin, fmax=fmax, 
                                             nPerSegment=int(secPerSegment * fs), 
                                             nOverlap=int(secOverlap * fs), 
                                             aggregate=True, verbose=False))
                except KeyError:
                    print(f'{label}{i} not found')
            else: 
                print(f'Skipping epochs {i}')
        return np.stack(psds_)

    nSkip = 0
    psds_left = epochs_to_psd(RAW_SL, fs, event_dict, 'left_', events_from_annot, tmin=tmin, tmax=tmax, twindow=twindow, fmin=fmin, fmax=fmax, resolution=resolution, secPerSegment=1, secOverlap=0.5, nSkip=nSkip)
    psds_left_rest = epochs_to_psd(RAW_SL, fs, event_dict, 'left_rest_', events_from_annot, tmin=tmin, tmax=tmax, twindow=twindow, fmin=fmin, fmax=fmax, resolution=resolution, secPerSegment=1, secOverlap=0.5, nSkip=nSkip)
    psds_right = epochs_to_psd(RAW_SL, fs, event_dict, 'right_', events_from_annot, tmin=tmin, tmax=tmax, twindow=twindow, fmin=fmin, fmax=fmax, resolution=resolution, secPerSegment=1, secOverlap=0.5, nSkip=nSkip)
    psds_right_rest = epochs_to_psd(RAW_SL, fs, event_dict, 'right_rest_', events_from_annot, tmin=tmin, tmax=tmax, twindow=twindow, fmin=fmin, fmax=fmax, resolution=resolution, secPerSegment=1, secOverlap=0.5, nSkip=nSkip)


    # PLOT PSDS ACCORDING TO 


    # Identify Left vs Right electrodes based on montage
    isLeft_ch = [x for x in MI.find_ch_circle(ch_location, radius=0.74) if x in MI.find_ch_left(ch_location)]
    isRight_ch = [x for x in MI.find_ch_circle(ch_location, radius=0.74) if x in MI.find_ch_right(ch_location)]

    isLeft = np.array([True if x in isLeft_ch else False for x in ch_setSLAP.get_labels()])
    isRight = np.array([True if x in isRight_ch else False for x in ch_setSLAP.get_labels()])

    fig = plt.figure(figsize=(8, 4))
    plt.subplot(121)
    left_labels = [x for x, y in zip(ch_setSLAP.get_labels(), isLeft) if y == True]
    MI.show_electrode(ch_location, left_labels, True, 'red')
    plt.subplot(122)
    right_labels = [x for x, y in zip(ch_setSLAP.get_labels(), isRight) if y == True]
    MI.show_electrode(ch_location, right_labels, True, 'blue')
    plt.yticks([])
    plt.subplots_adjust(wspace=0)  # Remove space between plots
    plt.show()



    bins_ticks = np.arange(fmin, fmax+1, int(resolution))
    theta_ticks = np.where((bins_ticks>=4) & (bins_ticks<=7))[0]
    alpha_ticks = np.where((bins_ticks>=8) & (bins_ticks<=12))[0]
    beta_ticks = np.where((bins_ticks>=13) & (bins_ticks<=30))[0]

    # Create statistics to test in nonparametric tests
    N = 1999
    perm_bool = False
    boot_bool = True
    
    p_left = []
    p_right = []
    labels = []

    # LEFT HAND TEST
    x = np.vstack([psds_left_rest, psds_left])
    isTreatment = np.arange(x.shape[0]) < psds_left_rest.shape[0]

    # theta
    T = mi.SumsR2(ch_set=ch_setSLAP, dict_symm=eeg_dict.dict_symm, isContralat=isRight, bins=theta_ticks)
    if perm_bool:
        p = MI.ApproxPermutationTest(x=x, isTreatment=isTreatment, stat=T.DifferenceOfSumsR2, nSimulations=N )
        p_left.append(p)
    if boot_bool:
        p = MI.BootstrapTest(x=x, isTreatment=isTreatment, stat=T.DifferenceOfSumsR2, nSimulations=N, nullHypothesisStatValue=0.0 )
        p_left.append(p)
        
    labels.append(r'4-7 Hz')
    r2_left_theta = T.DifferenceOfR2(x, isTreatment)
    
    # alpha
    T = mi.SumsR2(ch_set=ch_setSLAP, dict_symm=eeg_dict.dict_symm, isContralat=isRight, bins=alpha_ticks)
    if perm_bool:
        p = MI.ApproxPermutationTest(x=x, isTreatment=isTreatment, stat=T.DifferenceOfSumsR2, nSimulations=N )
        p_left.append(p)
    if boot_bool:
        p = MI.BootstrapTest(x=x, isTreatment=isTreatment, stat=T.DifferenceOfSumsR2, nSimulations=N, nullHypothesisStatValue=0.0 )
        p_left.append(p)

    labels.append(r'8-12 Hz')
    r2_left_alpha = T.DifferenceOfR2(x, isTreatment)
    
    # beta
    T = mi.SumsR2(ch_set=ch_setSLAP, dict_symm=eeg_dict.dict_symm, isContralat=isRight, bins=beta_ticks)
    if perm_bool:
        p = MI.ApproxPermutationTest(x=x, isTreatment=isTreatment, stat=T.DifferenceOfSumsR2, nSimulations=N )
        p_left.append(p)
    if boot_bool:
        p = MI.BootstrapTest(x=x, isTreatment=isTreatment, stat=T.DifferenceOfSumsR2, nSimulations=N, nullHypothesisStatValue=0.0 )
        p_left.append(p)

    labels.append(r'13-30 Hz')
    r2_left_beta = T.DifferenceOfR2(x, isTreatment)


    # RIGHT HAND TESTS
    x = np.vstack([psds_right_rest, psds_right])
    isTreatment = np.arange(x.shape[0]) < psds_right_rest.shape[0]

    # theta
    T = mi.SumsR2(ch_set=ch_setSLAP, dict_symm=eeg_dict.dict_symm, isContralat=isLeft, bins=theta_ticks)
    if perm_bool:
        p = MI.ApproxPermutationTest(x=x, isTreatment=isTreatment, stat=T.DifferenceOfSumsR2, nSimulations=N )
        p_right.append(p)
    if boot_bool:
        p = MI.BootstrapTest(x=x, isTreatment=isTreatment, stat=T.DifferenceOfSumsR2, nSimulations=N, nullHypothesisStatValue=0.0 )
        p_right.append(p)
        
    #labels.append(r'4-7 Hz')
    r2_right_theta = T.DifferenceOfR2(x, isTreatment)

    # alpha
    T = mi.SumsR2(ch_set=ch_setSLAP, dict_symm=eeg_dict.dict_symm, isContralat=isLeft, bins=alpha_ticks)
    if perm_bool:
        p = MI.ApproxPermutationTest(x=x, isTreatment=isTreatment, stat=T.DifferenceOfSumsR2, nSimulations=N )
        p_right.append(p)
    if boot_bool:
        p = MI.BootstrapTest(x=x, isTreatment=isTreatment, stat=T.DifferenceOfSumsR2, nSimulations=N, nullHypothesisStatValue=0.0 )
        p_right.append(p)
        
    #labels.append(r'8-12 Hz')
    r2_right_alpha = T.DifferenceOfR2(x, isTreatment)

    # beta
    T = mi.SumsR2(ch_set=ch_setSLAP, dict_symm=eeg_dict.dict_symm, isContralat=isLeft, bins=beta_ticks)
    if perm_bool:
        p = MI.ApproxPermutationTest(x=x, isTreatment=isTreatment, stat=T.DifferenceOfSumsR2, nSimulations=N )
        p_right.append(p)
    if boot_bool:
        p = MI.BootstrapTest(x=x, isTreatment=isTreatment, stat=T.DifferenceOfSumsR2, nSimulations=N, nullHypothesisStatValue=0.0 )
        p_right.append(p)

    #labels.append(r'13-30 Hz')
    r2_right_beta = T.DifferenceOfR2(x, isTreatment)



    # MERGE LEFTvsRIGHT TESTS
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 2), sharey=True)
    y_max = 0.5
    y_min = -0.5
    y = np.linspace(start=y_max, stop=y_min, num=len(p_left)+2)
    y = y[1:-1]
    deltay = 0.075
    # LEFT RESULTS
    #---------
    x_values = []
    xUp_values = []
    xDown_values = []
    for p in p_left:
        p_down, p, p_up = MI.pvalue_interval(p, N+1)
        xUp_values.append(MI.negP(p_up))
        x_values.append(MI.negP(p))
        xDown_values.append(MI.negP(p_down))
    #---------
    ax1.set_title('Open/Close Left')
    ax1.scatter(x_values, y, color='red', marker='.')
    # Add confidence interval on true p
    for i in range(len(y)):
        ax1.fill_betweenx([y[i]-deltay, y[i]+deltay], xDown_values[i], xUp_values[i], color='red', alpha=0.3)
    ax1.hlines(y, 0, x_values, colors='red', lw=1, alpha=0.25)
    ax1.set_xlabel(r'-log($p$)')
    ax1.set_xlim(ax1.get_xlim()[::-1])  # Reverse the x-axis for left plot
    ax1.set_xlim(right=0, left=8)
    ax1.set_ylim(y_min, y_max)
    ax1.axvline(MI.negP(0.05), color='cornflowerblue', lw=1, ls='--', alpha=0.5, label='95% C.L.')
    ax1.axvline(MI.negP(0.01), color='black', lw=1, ls='--', alpha=0.5, label='99% C.L.')
    ax1.set_yticks(y)
    ax1.set_yticklabels(labels)
    ax1.legend(loc='upper left')
    # RIGHT RESULTS
    #---------
    x_values = []
    xUp_values = []
    xDown_values = []
    for p in p_right:
        p_down, p, p_up = MI.pvalue_interval(p, N+1)
        xUp_values.append(MI.negP(p_up))
        x_values.append(MI.negP(p))
        xDown_values.append(MI.negP(p_down))
    #---------
    ax2.set_title('Open/Close Right')
    ax2.scatter(x_values, y, color='blue', marker='.')
    # Add confidence interval on true p
    for i in range(len(y)):
        ax2.fill_betweenx([y[i]-deltay, y[i]+deltay], xDown_values[i], xUp_values[i], color='blue', alpha=0.3)
    ax2.hlines(y, 0, x_values, colors='blue', lw=1, alpha=0.25)
    ax2.set_xlabel(r'-log($p$)')
    ax2.set_xlim(left=0, right=8)
    ax1.set_ylim(y_min, y_max)
    ax2.axvline(MI.negP(0.05), color='cornflowerblue', lw=1, ls='--', alpha=0.5)
    ax2.axvline(MI.negP(0.01), color='black', lw=1, ls='--', alpha=0.5)
    #--------- 
    plt.subplots_adjust(wspace=0)  # Adjust space between subplots
    plt.show()
    