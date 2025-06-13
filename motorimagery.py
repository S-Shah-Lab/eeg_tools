"""
This script runs the major parts of the motor imagery paradigm analysis.

1) It naturally cleans the input file with minimum input from the user (it manually requires you to visually select the BAD regions)

2) It splits the EEG sections into epochs and extracts PSDs using the Welch method.

3) It converts PSDs values from chosen frequency bands into signed-r2 coeffiecients and performs statistical tests using permutation and bootstrap methods to extract p-values

4) It generates a PDF report with some of the main findings.
"""

if __name__ == "__main__":
    import argparse
    from Meta import *

    __version__ = GetRevision()
    if __version__ == "unknown revision":
        __version__ = "N/A"
    else:
        __version__ = __version__.split()[1]

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "-c",
        "--cleaned",
        metavar="cleaned",
        type=bool,
        default=False,
        help="Has the file been previously cleaned? This varaible makes the script skip the cleaning step and import a .fif file instead",
    )
    parser.add_argument(
        "-f",
        "--file_path",
        metavar="file_path",
        type=str,
        default="",
        help="Path to the file to run the scipt on. This is the path to a EEG motorimagery .dat file, if previously cleaned, a .fif file must be present in the same folder",
    )
    parser.add_argument(
        "-r",
        "--resolution",
        metavar="resolution",
        type=int,
        default=1,
        help="Resolution for PSDs [1, 2] Hz. This variable is passed to the function epochs_to_psd(), is used to determine the bins in many plots and is a key component in the names of saved plots",
    )
    parser.add_argument(
        "-m",
        "--fmin",
        metavar="fmin",
        type=float,
        default=1.0,
        help="Min frequency to consider for PSDs. This variable is used by bandpass filter, is used to determine the bins in many plots and passed to the function epochs_to_psd()",
    )
    parser.add_argument(
        "-M",
        "--fmax",
        metavar="fmax",
        type=float,
        default=40.0,
        help="Max frequency to consider for PSDs. This variable is used by bandpass filter, is used to determine the bins in many plots and passed to the function epochs_to_psd()",
    )
    parser.add_argument(
        "-b",
        "--freq_band",
        metavar="freq_band",
        type=list,
        default=[4.0, 8.0, 13.0, 31.0],
        help="Splits the frequency band using these values in continuous intervals. The frequency band is passed to the object STAT as it determines which PSDs bins are aggregated",
    )
    parser.add_argument(
        "-p",
        "--prep",
        metavar="prep",
        type=bool,
        default=True,
        help="Do you want to run prep on the EEG signal? This variable add cleaning steps to the script. If no bad channels are found, no interpolation occurs as well",
    )
    parser.add_argument(
        "-x",
        "--xray",
        metavar="xray",
        type=list,
        default=["c3", "c4"],
        help="Detailed information of the explicitly mentioned channels will be provided and saved",
    )
    parser.add_argument(
        "--pause",
        metavar="pause",
        type=float,
        default=0.5,
        help="Each generated plot will remain visible for the given time in seconds. This variable is passed to matplotlib pause() function",
    )
    parser.add_argument(
        "-P",
        "--pdf",
        metavar="pdf",
        type=bool,
        default=False,
        help="Skips the analysis and plot generation part and only generates the output PDF file. This variable allows to save a lot of time if all the plots and information needed for the PDF already exist",
    )
    parser.add_argument(
        "-N",
        "--nSplit",
        metavar="nSplit",
        type=int,
        default=6,
        help="Determines the number of epochs in each trial, splits the trial in equal segments. This variable is passed to make_annotation_MI() to determine the onset of each epoch and it defines tmax which is later passed to epochs_to_psd()",
    )
    parser.add_argument(
        "-t",
        "--twindow",
        metavar="twindow",
        type=float,
        default=9.0,
        help="Determines the window length that will be analyzed, ending at the end of each trial. Hence, it indirectly determines how much at the beginning of the trial is automatically skipped. This variable defines tmax which is later passed to epochs_to_psd()",
    )
    parser.add_argument(
        "--new_ref",
        metavar="new_ref",
        type=str,
        default="tp9 tp10",
        help="Specifies the channels used in rereferencing for EGI 128 montage, which is currently the only used montage that requires offline rerefencing. The variable is passed to the ChannelSet.RerefMatrix() function, when more than one channel is specified the new reference will be the average of those channels",
    )
    parser.add_argument(
        "--ch_egi",
        metavar="ch_egi",
        type=int,
        default=128,
        help="Specifies the number of channels to use in case the EGI montage is found, this is useful to break the duality of 128 channels or only a subset of 64 channels with the same montage",
    )
    parser.add_argument(
        "-y",
        "--yob",
        metavar="yob",
        type=int,
        default=None,
        help="Year of birth of the subject",
    )

    opts = parser.parse_args()


import os
import numpy as np

# from BCI2000Tools.FileReader import bcistream
from BCI2000Tools.Electrodes import *
from BCI2000Tools.Plotting import *
import mne
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.font_manager import FontProperties

# from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable

# from pyprep.prep_pipeline import PrepPipeline, NoisyChannels

import eeg_dict  # Contains dictionaries and libraries for electrodes locations
import MotorImageryTools as mi  # Contains tools for eeg and motor imagery
import PlotStyle

PlotStyle.set_plot_style()

import PdfReport  # Contains tools for PDF report generation


if __name__ == "__main__":

    # Check if the file exists
    if os.path.isfile(opts.file_path):
        pass
    # If the file doesn't exists print message and exit
    else:
        print(f"File {opts.file_path} not found! Exiting...\n")
        exit(1)

    # Initialize EEG and Plotting classes from MotorImageryTools.py
    EEG = mi.EEG()  # Initialize EEG tools
    PLOT = mi.Plotting()  # Initialize Plotting tools
    QUALITY = mi.SignalQuality()  # Initialize SignalQuality tools

    # Store input options into variables
    # does the imported .dat file have a previously cleaned .fif file?
    clean_bool = opts.cleaned
    # split imported file info into path and name
    file_path, file_name = os.path.split(opts.file_path)
    # PSD frequency resolution
    resolution = opts.resolution
    # how long are the segments in Welch PSD method
    secPerSegment = 1 / resolution
    # how long are the overlapping parts in Welch PSD method
    secOverlap = secPerSegment / 2
    # lowest frequency in PSD spectrum
    fmin = opts.fmin
    # highest frequency in PSD spectrum
    fmax = opts.fmax
    # list w/ frequency boundaries for each
    freq_band = opts.freq_band
    # do you want to run prep?
    prep = opts.prep
    # which channels do you want to take a detailed look at? ['c3', 'c4']
    xray = opts.xray
    # how long do you want to wait for each plot?
    pause = opts.pause
    # do you only want to generate the pdf report?
    pdf_only = opts.pdf
    # determines the number of epochs in each trial, splits the trial in equal segments
    nSplit = opts.nSplit
    # determines the window ending at the end of each trial that will be analyzed, first part is automatically skipped
    twindow = opts.twindow
    # channels used in rereferencing for EGI 128 montage
    new_ref = opts.new_ref
    # number of channels used in case of EGI 128 montage
    ch_egi = opts.ch_egi
    # year of birth of the subject
    yob = opts.yob

    # Extract base name from file
    base_name, extension = os.path.splitext(file_name)
    sub_name = base_name.split("sub-")[1].split("_ses")[0]
    ses_name = base_name.split("ses-")[1].split("_")[0]
    # String used in some plots for labeling
    text_id = (
        "Sub" + f" $\\mathbf{{{sub_name}}}$,  " + "Ses" + f" $\\mathbf{{{ses_name}}}$"
    )
    # Create a folder using base name, if folder doesn't exist
    path_to_folder = EEG.clean_path(
        EEG.create_folder(path=file_path, folder_name=base_name)
    )

    # Import .dat file
    signal, states, fs, ch_names, blockSize, montage_type, date_test = (
        EEG.import_file_dat(file_path, file_name)
    )

    # The montage_type is automatically decided by the dimensions of the signal matrix
    # ch_info contains the channel names for that specific montage
    if montage_type == "DSI_24":
        ch_info = "DSI24_location.txt"  # DSI-24 ch dry
    elif montage_type == "GTEC_32":
        ch_info = "GTEC32_location.txt"  # g.Nautilus 32 ch gel
    elif montage_type == "EGI_128":
        if ch_egi == 64:
            ch_info = "EGI64_location.txt"  # HydroCel GNS 64 ch gel (adapted from HydroCel GNS 128 ch montage)
            montage_type = "EGI_64"
            signal = signal[eeg_dict.id_ch_64_keep]
            ch_names = list(np.array(ch_names)[eeg_dict.id_ch_64_keep])
        else:
            ch_info = "EGI128_location.txt"  # HydroCel GNS 128 ch saline/gel

    # Information used later in the PDF report generation
    plot_folder = path_to_folder
    montage_name = " ".join(montage_type.split("_"))

    # Define paradigm information based on blocksize of the EEG file
    # If fs % blocksize == 0:
    #    the blocksize fits perfectly in the sampling frequency so if a time window has a integer length (e.e 2 seconds) all blocks will be saved
    #    this might not be true for decimal length windows (e.g. 2.4 seconds), but our motorimagery doesn't have any
    fileTime, nBlocks, trialsPerBlock, initialSec, stimSec, taskSec = (
        EEG.evaluate_mi_paradigm(
            signal=signal, states=states, fs=fs, blockSize=blockSize, verbose=True
        )
    )

    # Rejected seconds after end of cue
    # Here we are making sure the task trial length is 9 seconds
    rejectSec = taskSec - twindow  # 1 [s]
    # Min and Max time for each epoch
    tmin = 0
    tmax = twindow / nSplit  # e.g. 1.5

    # Don't just produce the PDF, go ahead with analysis to generate the plots
    if not pdf_only:
        # File has not been cleaned before, it's a new file
        if not clean_bool:
            # Define initial ChannelSet
            # ChannelSet comes from BCI2000Tools, it contains information regarding a specific montage and can be used for transformation of the signal matrix
            ch_set = ChannelSet(ch_info)

            # Create RAW for PREP, signal is unfiltered
            RAW = EEG.make_RAW_with_montage(
                signal=signal * 1e-6,  # uV
                fs=fs,
                ch_names=ch_set.get_labels(),
                montage_type=montage_type,
                conv_dict=eeg_dict.stand1020_to_egi,
            )

            #####################################################################################
            # THIS IS THE PLACE TO INCLUDE QUALITY OF DATA CHECK
            # SAVE THE OUTPUT TO THE FOLDER
            #####################################################################################

            # Channels with PfInterest/PfAll > 1 are bad channels (very high impedence)
            QUALITY.quality_signal_powers(
                signal=signal,
                fs=fs,
                fDC=0.1,
                fLow=1,
                fHigh=40,
                fPowerLine=60,
                plot=True,
                n_components=2,
                ch_names=ch_set.get_labels(),
                eeg_dict=eeg_dict,
                path_to_folder=path_to_folder,
            )
            #####################################################################################
            # THIS IS THE PLACE TO INCLUDE QUALITY OF DATA CHECK
            # SAVE THE OUTPUT TO THE FOLDER
            #####################################################################################

            # Run PREP for bad channels
            # Don't run Ransac at first, it's ideal to run it after some bad channels are identified, if any
            if prep:
                EEG.make_PREP(
                    RAW,
                    isSNR=False,
                    isDeviation=False,
                    isHfNoise=True,
                    isNanFlat=True,
                    isRansac=False,
                )

            # Bandpass filter
            signalFilter = EEG.filter_data(signal, fs, l_freq=fmin, h_freq=fmax)

            # Re-reference only in the case of EGI montage, the other ones we are using are ready to go
            if montage_type == "EGI_128" or montage_type == "EGI_64":
                signalFilter, ch_set = EEG.spatial_filter(
                    sfilt="REF",
                    ch_set=ch_set,
                    signal=signalFilter,
                    flag_ch=new_ref,  # Re-reference to average of mastoids
                    verbose=True,
                )

            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # PLOT ChannelSet after potential re-reference
            # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            # Plot montage using BCI2000Tools
            PLOT.plot_channelset(
                ch_set, montage_type, sub_name, ses_name, pause=pause, figsize=(6, 6)
            )

            # Create RAW with montage, signal is filtered
            RAW = EEG.make_RAW_with_montage(
                signal=signalFilter * 1e-6,  # uV
                fs=fs,
                ch_names=ch_set.get_labels(),
                montage_type=montage_type,
                conv_dict=eeg_dict.stand1020_to_egi,
            )

            # Run PREP for bad channels
            # Run Ransac
            if prep:
                EEG.make_PREP(
                    RAW,
                    isSNR=True,
                    isDeviation=True,
                    isHfNoise=True,
                    isNanFlat=True,
                    isRansac=True,
                )

            # Mark BAD regions (Done visually)
            EEG.mark_BAD_region(RAW, block=True)

            # Summary of BAD regions (confirm the marking)
            EEG.evaluate_BAD_region(RAW, max_duration=fileTime)

            # Grab BAD onset and duration
            annot = RAW.annotations
            bad_onset = annot.onset[np.where(annot.description == "BAD_region")]
            bad_duration = annot.duration[np.where(annot.description == "BAD_region")]
            # Create a figure and axis
            fig, ax = plt.subplots(figsize=(10, 2))
            EEG.visualize_BAD_region(
                max_duration=fileTime,
                annotation_onset=bad_onset,
                annotation_duration=bad_duration,
                color="red",
                ax=ax,
            )
            # Create a bar to visualize bad areas
            plt.savefig(f"{path_to_folder}BAD_segments.png", bbox_inches="tight")
            plt.savefig(f"{path_to_folder}BAD_segments.svg", bbox_inches="tight")
            plt.show(block=False)
            plt.pause(pause)
            plt.close()

            # Add Stimuli to RAW
            EEG.make_RAW_stim(RAW, states)

            # Create annotations on RAW object
            RAW = EEG.make_annotation_MI(
                RAW,
                fs,
                nBlocks=nBlocks,
                trialsPerBlock=trialsPerBlock,
                initialSec=initialSec,
                stimSec=stimSec,
                taskSec=taskSec,
                rejectSec=rejectSec,
                nSplit=nSplit,
                fileTime=fileTime,
            )

            # Save RAW as .fif
            # This file can be imported later to skip all the previous steps
            EEG.save_RAW(RAW=RAW, path=file_path, file_name=base_name, label="")

        else:  # Alternative to `if not clean_bool:`
            # Import a previously saved .fif file
            RAW, montage, fs = EEG.import_file_fif(
                path=file_path, file_name=base_name + ".fif"
            )

            # Grab BAD onset and duration
            annot = RAW.annotations
            bad_onset = annot.onset[np.where(annot.description == "BAD_region")]
            bad_duration = annot.duration[np.where(annot.description == "BAD_region")]
            # Create a figure and axis
            fig, ax = plt.subplots(figsize=(10, 2))
            EEG.visualize_BAD_region(
                max_duration=fileTime,
                annotation_onset=bad_onset,
                annotation_duration=bad_duration,
                color="red",
                ax=ax,
            )
            # Create a bar to visualize bad areas
            plt.savefig(f"{path_to_folder}BAD_segments.png", bbox_inches="tight")
            plt.savefig(f"{path_to_folder}BAD_segments.svg", bbox_inches="tight")
            plt.show(block=False)
            plt.pause(pause)
            plt.close()

            # Define ChannelSet using RAW.info
            # ChannelSet comes from BCI2000Tools, it contains information regarding a specific montage and can be used for transformation of the signal matrix
            ch_set = ChannelSet(
                [
                    x.lower()
                    for x in RAW.info["ch_names"][: RAW.get_data(picks="eeg").shape[0]]
                ]
            )

            # There is a little problem with EGI montage since the location of those electrodes is not known by ChannelSet
            # We need to manually re-reference like we did before to the average of the mastoids
            # Note: only the ChannelSet needs to be re-referenced, the signal matrtix is already ok since it's imported from before
            # This is manually done here
            if montage_type == "EGI_128" or montage_type == "EGI_64":
                # Define ChannelSet
                if ch_egi == 64:
                    ch_set = ChannelSet(
                        "EGI64_location.txt"
                    )  # HydroCel GNS 64 ch gel (adapted from HydroCel GNS 128 ch montage)
                else:
                    ch_set = ChannelSet(
                        "EGI128_location.txt"
                    )  # HydroCel GNS 128 ch saline/gel
                # Re-reference, make sure this is the same as the one done before saving the RAW .fif
                m = np.array(ch_set.RerefMatrix(new_ref))
                # Apply re-reference, occurs in place
                ch_set = ch_set.spfilt(m)

        # RAW was saved without interpolation, RAW.info['bads'] contains the prevously identified bad channels, if any
        # Store previosly identified bad channels
        old_ch_bads = RAW.info["bads"]

        # Interpolate BAD channels, if any
        if not RAW.info["bads"] == []:
            # Interpolate
            old_ch_bads = EEG.interpolate(RAW)

            # Is any channel BAD after interpolation?
            if prep:
                EEG.make_PREP(
                    RAW,
                    isSNR=True,
                    isDeviation=True,
                    isHfNoise=True,
                    isNanFlat=True,
                    isRansac=True,
                )
                print(f"Old bad channels: {old_ch_bads}")
                print(f"Currently bad channels: {RAW.info['bads']}")

        # Spatial filter with exclusion
        signalSLAP, ch_setSLAP = EEG.spatial_filter(
            sfilt="SLAP",
            ch_set=ch_set,
            signal=RAW.get_data(picks="eeg"),
            flag_ch=eeg_dict.ch_face + eeg_dict.ch_forehead,  # to be excluded
            verbose=True,
        )

        # Create RAW after spatial filter (RAW_SL)
        RAW_SL = EEG.make_RAW_with_montage(
            signal=signalSLAP,
            fs=fs,
            ch_names=ch_setSLAP.get_labels(),
            montage_type=montage_type,
            conv_dict=eeg_dict.stand1020_to_egi,
        )

        # Add Stim to RAW_SL
        EEG.make_RAW_stim(RAW_SL, states)

        # Import annotations
        RAW_SL.set_annotations(RAW.annotations)
        # RAW_SL.plot()

        # Create Epochs and PSDs
        events_from_annot, event_dict = mne.events_from_annotations(RAW_SL)

        # Generate PSDs for each type of trial
        nSkip = []
        # Left trials
        psds_left = EEG.epochs_to_psd(
            RAW_SL,
            fs,
            event_dict,
            "left_",
            events_from_annot,
            tmin=tmin,
            tmax=tmax,
            fmin=fmin,
            fmax=fmax,
            resolution=resolution,
            secPerSegment=secPerSegment,
            secOverlap=secOverlap,
            nSkip=nSkip,
        )
        # Rest after Left trials
        psds_left_rest = EEG.epochs_to_psd(
            RAW_SL,
            fs,
            event_dict,
            "left_rest_",
            events_from_annot,
            tmin=tmin,
            tmax=tmax,
            fmin=fmin,
            fmax=fmax,
            resolution=resolution,
            secPerSegment=secPerSegment,
            secOverlap=secOverlap,
            nSkip=nSkip,
        )
        # Right trials
        psds_right = EEG.epochs_to_psd(
            RAW_SL,
            fs,
            event_dict,
            "right_",
            events_from_annot,
            tmin=tmin,
            tmax=tmax,
            fmin=fmin,
            fmax=fmax,
            resolution=resolution,
            secPerSegment=secPerSegment,
            secOverlap=secOverlap,
            nSkip=nSkip,
        )
        # Rest after Right
        psds_right_rest = EEG.epochs_to_psd(
            RAW_SL,
            fs,
            event_dict,
            "right_rest_",
            events_from_annot,
            tmin=tmin,
            tmax=tmax,
            fmin=fmin,
            fmax=fmax,
            resolution=resolution,
            secPerSegment=secPerSegment,
            secOverlap=secOverlap,
            nSkip=nSkip,
        )

        print(
            f"psds_left.shape: {psds_left_rest.shape}, psds_left_rest.shape: {psds_left.shape}, psds_right.shape:  {psds_right.shape}, psds_right_rest.shape: {psds_right_rest.shape}"
        )

        # Find all channels on the left and right hemispheres (in the subset of central, parietal and frontal channels)
        isLeft_ch = [
            x for x in eeg_dict.ch_motor if x in EEG.find_ch_left(eeg_dict.ch_location)
        ]
        isRight_ch = [
            x for x in eeg_dict.ch_motor if x in EEG.find_ch_right(eeg_dict.ch_location)
        ]

        isLeft_ch = np.unique(isLeft_ch)
        isRight_ch = np.unique(isRight_ch)

        # Convert ch_set channels into an array of True of False based on the ones to consider
        isLeft = np.array(
            [
                True if x in isLeft_ch else False
                for x in EEG.low(ch_setSLAP.get_labels())
            ]
        )
        isRight = np.array(
            [
                True if x in isRight_ch else False
                for x in EEG.low(ch_setSLAP.get_labels())
            ]
        )

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # PLOT the electrodes used in the statistical tests
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        plt.figure(figsize=(6, 4))
        # All channels in the montage
        PLOT.show_electrode(
            eeg_dict.ch_location,
            EEG.low(list(np.array(ch_setSLAP.get_labels()))),
            label=False,
            color="grey",
            alpha_back=0,
            marker="o",
        )
        # Left channels
        PLOT.show_electrode(
            eeg_dict.ch_location,
            EEG.low(list(np.array(ch_setSLAP.get_labels())[isLeft])),
            label=True,
            color="blue",
            alpha_back=0,
            marker="o",
        )
        # Right channels
        PLOT.show_electrode(
            eeg_dict.ch_location,
            EEG.low(list(np.array(ch_setSLAP.get_labels())[isRight])),
            label=True,
            color="magenta",
            alpha_back=0,
            marker="o",
        )
        # Interpolated channels
        PLOT.show_electrode(
            eeg_dict.ch_location,
            EEG.low(old_ch_bads),
            label=False,
            color="lime",
            alpha_back=0,
            marker="x",
        )
        lim = 1.8
        plt.xlim(-lim, lim)
        plt.ylim(-1.3, 1.3)
        plt.xticks([])
        plt.yticks([])
        plt.gca().spines["top"].set_visible(False)
        plt.gca().spines["right"].set_visible(False)
        plt.gca().spines["bottom"].set_visible(False)
        plt.gca().spines["left"].set_visible(False)
        plt.text(-lim * 0.95, 1 * 1.05, f"Montage", weight="bold")
        split_text = montage_type.split("_")
        plt.text(-lim * 0.65, 1 * 1.05, f"{split_text[0]} {split_text[1]} Channels")
        plt.axvline(0, lw=1, ls="--", color="grey", alpha=0.25)
        # Show the boundary of the head:
        r0 = 1.05
        theta = np.arange(0, np.pi, 0.01)
        plt.plot(r0 * np.cos(theta), r0 * np.sin(theta), color="black", lw=1)
        plt.plot(r0 * np.cos(theta), -r0 * np.sin(theta), color="black", lw=1)
        # Save the plot
        plt.savefig(f"{path_to_folder}target_electrodes.png", bbox_inches="tight")
        plt.savefig(f"{path_to_folder}target_electrodes.svg", bbox_inches="tight")
        plt.show(block=False)
        plt.pause(pause)
        plt.close()

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # STATISTICAL TESTS
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Generate frequency bins in each frequency band
        bins_ticks = np.arange(fmin, fmax + resolution, resolution)

        # Number of frequency bands to consider
        N = len(freq_band) - 1

        # Initialize useful things for the statistical test
        nSim = 2999
        perm_bool = True  # Do Permutation test
        boot_bool = True  # Do Bootstrap test

        r2_left = []
        p_left = []
        labels = []

        # LEFT HAND Statistical Test begins
        fig, axs = plt.subplots(nrows=2, ncols=N, figsize=(int(4 * N), 6))

        # Generate array of trials, rest followed by task
        x = np.vstack([psds_left_rest, psds_left])  # (trial, ch, bin)

        print(f"x = np.vstack([psds_left_rest, psds_left]): {x.shape}")

        # Generate labels for rest (True) and task (False)
        isTreatment = np.arange(x.shape[0]) < psds_left_rest.shape[0]  # (trial,)
        isContralat = isRight

        for i in range(len(freq_band) - 1):
            # Initialize the STAT class, specifying the contralaterla electrodes, the frequency bins and the frequency bands separators
            STAT = mi.Stats(
                ch_set=ch_setSLAP,
                dict_symm=eeg_dict.dict_symm,
                isContralat=isContralat,
                bins=bins_ticks,
                custom_bins=[freq_band[i], freq_band[i + 1]],
            )

            # Average the data (PSDs) within the specified frequency bins (trial, ch, bin) -> (trial, ch)
            x_ = np.mean(x[:, :, STAT.custom_ticks[0] : STAT.custom_ticks[-1]], axis=2)
            # Transform PSDs to dB
            x_ = EEG.convert_dB(x_)

            # Calculate r2
            r2_left.append(STAT.Transform(x_, isTreatment))

            # Permutation test
            if perm_bool:
                if i == 0:
                    axs[0, i].set_title(
                        f"Open/Close Left", fontsize=12, weight="bold", loc="left"
                    )
                if i == 2:
                    axs[0, i].text(
                        0.995,
                        1.1,
                        text_id,
                        ha="right",
                        va="top",
                        transform=axs[0, i].transAxes,
                        fontsize=11,
                        bbox=dict(
                            facecolor="white",
                            edgecolor="black",
                            boxstyle="square,pad=0.1",
                        ),
                    )
                axs[0, i].text(
                    0.98,
                    0.97,
                    f"[{freq_band[i]}-{freq_band[i+1]-1}] Hz",
                    va="top",
                    ha="right",
                    transform=axs[0, i].transAxes,
                    fontsize=12,
                )
                p = STAT.ApproxPermutationTest(
                    x=x_,
                    isTreatment=isTreatment,
                    stat=STAT.DifferenceOfSumsR2,
                    nSimulations=nSim,
                    plot=True,
                    ax=axs[0, i],
                )
                p_left.append(p)
                labels.append(f"{freq_band[i]}-{freq_band[i+1]-1} Hz (P)")

            # Bootstrap test
            if boot_bool:
                axs[1, i].text(
                    0.98,
                    0.97,
                    f"[{freq_band[i]}-{freq_band[i+1]-1}] Hz",
                    va="top",
                    ha="right",
                    transform=axs[1, i].transAxes,
                    fontsize=12,
                )
                p = STAT.BootstrapTest(
                    x=x_,
                    isTreatment=isTreatment,
                    stat=STAT.DifferenceOfSumsR2,
                    nSimulations=nSim,
                    nullHypothesisStatValue=0.0,
                    plot=True,
                    ax=axs[1, i],
                )
                p_left.append(p)
                labels.append(f"{freq_band[i]}-{freq_band[i+1]-1} Hz (B)")

        plt.tight_layout()
        plt.savefig(
            f"{path_to_folder}test_left_{resolution}_Hz.png", bbox_inches="tight"
        )
        plt.savefig(
            f"{path_to_folder}test_left_{resolution}_Hz.svg", bbox_inches="tight"
        )
        plt.show(block=False)
        plt.pause(pause)
        plt.close()

        p_right = []
        r2_right = []

        # RIGHT HAND Statistical Test begins
        fig, axs = plt.subplots(nrows=2, ncols=N, figsize=(int(4 * N), 6))

        # Generate array of trials, rest followed by task
        x = np.vstack([psds_right_rest, psds_right])  # (trial, ch, bin)

        print(f"x = np.vstack([psds_right_rest, psds_right]): {x.shape}")

        # Generate labels for rest (True) and task (False)
        isTreatment = np.arange(x.shape[0]) < psds_right_rest.shape[0]  # (trial,)
        isContralat = isLeft

        for i in range(len(freq_band) - 1):

            # Initialize the STAT class, specifying the contralaterla electrodes, the frequency bins and the frequency bands separators
            STAT = mi.Stats(
                ch_set=ch_setSLAP,
                dict_symm=eeg_dict.dict_symm,
                isContralat=isContralat,
                bins=bins_ticks,
                custom_bins=[freq_band[i], freq_band[i + 1]],
            )

            # Average the data (PSDs) within the specified frequency bins (trial, ch, bin) -> (trial, ch)
            x_ = np.mean(x[:, :, STAT.custom_ticks[0] : STAT.custom_ticks[-1]], axis=2)
            # Transform PSDs to dB
            x_ = EEG.convert_dB(x_)

            # Calculate r2
            r2_right.append(STAT.Transform(x_, isTreatment))

            # Permutation test
            if perm_bool:
                if i == 0:
                    axs[0, i].set_title(
                        f"Open/Close Right", fontsize=12, weight="bold", loc="left"
                    )
                if i == 2:
                    axs[0, i].text(
                        0.995,
                        1.1,
                        text_id,
                        ha="right",
                        va="top",
                        transform=axs[0, i].transAxes,
                        fontsize=11,
                        bbox=dict(
                            facecolor="white",
                            edgecolor="black",
                            boxstyle="square,pad=0.1",
                        ),
                    )
                axs[0, i].text(
                    0.98,
                    0.97,
                    f"[{freq_band[i]}-{freq_band[i+1]-1}] Hz",
                    va="top",
                    ha="right",
                    transform=axs[0, i].transAxes,
                    fontsize=12,
                )
                p = STAT.ApproxPermutationTest(
                    x=x_,
                    isTreatment=isTreatment,
                    stat=STAT.DifferenceOfSumsR2,
                    nSimulations=nSim,
                    plot=True,
                    ax=axs[0, i],
                )
                p_right.append(p)

            # Bootstrap test
            if boot_bool:
                axs[1, i].text(
                    0.98,
                    0.97,
                    f"[{freq_band[i]}-{freq_band[i+1]-1}] Hz",
                    va="top",
                    ha="right",
                    transform=axs[1, i].transAxes,
                    fontsize=12,
                )
                p = STAT.BootstrapTest(
                    x=x_,
                    isTreatment=isTreatment,
                    stat=STAT.DifferenceOfSumsR2,
                    nSimulations=nSim,
                    nullHypothesisStatValue=0.0,
                    plot=True,
                    ax=axs[1, i],
                )
                p_right.append(p)

        plt.tight_layout()
        plt.savefig(
            f"{path_to_folder}test_right_{resolution}_Hz.png", bbox_inches="tight"
        )
        plt.savefig(
            f"{path_to_folder}test_right_{resolution}_Hz.svg", bbox_inches="tight"
        )
        plt.show(block=False)
        plt.pause(pause)
        plt.close()

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # PLOT p-values extracted by Statistical tests
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        y_max = 0.5
        y_min = -0.5
        y = np.linspace(start=y_max, stop=y_min, num=len(p_left) + 2)
        y = y[1:-1]
        deltay = 0.04

        def montecarlo_error(ps, nSim):
            x_values = []
            xUp_values = []
            xDown_values = []
            for p in ps:
                p_down, p, p_up = STAT.pvalue_interval(p, nSim + 1)
                xUp_values.append(STAT.negP(p_up))
                x_values.append(STAT.negP(p))
                xDown_values.append(STAT.negP(p_down))
            return xDown_values, x_values, xUp_values

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3.5), sharey=True)

        # Left results ---------
        xDown_values, x_values, xUp_values = montecarlo_error(p_left, nSim)
        # ---------
        ax1.set_title("Open/Close Left", fontsize=12, loc="left", weight="bold")
        ax1.scatter(x_values, y, color="black", marker="o")
        # Add confidence interval on true p
        for i in range(len(y)):
            ax1.fill_betweenx(
                [y[i] - deltay, y[i] + deltay],
                xDown_values[i],
                xUp_values[i],
                color="gray",
                alpha=0.3,
            )
        ax1.hlines(y, 0, x_values, colors="black", lw=1.5, alpha=1)
        ax1.set_xlabel(r"-log(p)")
        ax1.set_xlim(ax1.get_xlim()[::-1])  # Reverse the x-axis for left plot
        ax1.set_xlim(right=0, left=6)
        ax1.set_ylim(y_min, y_max)
        ax1.axvline(
            STAT.negP(0.05),
            color="black",
            lw=1,
            ls=":",
            alpha=0.5,
            label=r"$\alpha=0.05$",
        )
        ax1.axvline(
            STAT.negP(0.01),
            color="darkviolet",
            lw=1,
            ls=":",
            alpha=0.5,
            label=r"$\alpha=0.01$",
        )
        ax1.set_yticks(y)
        ax1.set_yticklabels(labels)
        ax1.legend(loc="upper left")

        # Right results ---------
        xDown_values, x_values, xUp_values = montecarlo_error(p_right, nSim)
        # ---------
        ax2.set_title("Open/Close Right", fontsize=12, loc="right", weight="bold")
        ax2.scatter(x_values, y, color="black", marker="o")
        # Add confidence interval on true p
        for i in range(len(y)):
            ax2.fill_betweenx(
                [y[i] - deltay, y[i] + deltay],
                xDown_values[i],
                xUp_values[i],
                color="gray",
                alpha=0.3,
            )
        ax2.hlines(y, 0, x_values, colors="black", lw=1.5, alpha=1)
        ax2.set_xlabel(r"-log(p)")
        ax2.set_xlim(left=0, right=6)
        ax1.set_ylim(y_min, y_max)
        ax2.axvline(STAT.negP(0.05), color="black", lw=1, ls=":", alpha=0.5)
        ax2.axvline(STAT.negP(0.01), color="darkviolet", lw=1, ls=":", alpha=0.5)

        ax1.text(
            1.25,
            1.08,
            text_id,
            ha="right",
            va="top",
            transform=ax1.transAxes,
            fontsize=11,
            bbox=dict(facecolor="white", edgecolor="black", boxstyle="square,pad=0.1"),
        )

        plt.subplots_adjust(wspace=0)  # Adjust space between subplots
        fig.savefig(f"{path_to_folder}pVal_{resolution}_Hz.png", bbox_inches="tight")
        fig.savefig(f"{path_to_folder}pVal_{resolution}_Hz.svg", bbox_inches="tight")
        plt.show(block=False)
        plt.pause(pause)
        plt.close()

        # Plot p-values Left only (Bootstrap)
        fig, ax = plt.subplots(figsize=(3, int(2 * N)))
        y = np.linspace(start=y_max, stop=y_min, num=int(len(p_left) / 2 + 2))
        y = y[1:-1]
        deltay = 0.04

        # Left results ---------
        xDown_values, x_values, xUp_values = montecarlo_error(p_left[1::2], nSim)
        # ---------
        ax.text(
            0,
            1.05,
            "Open/Close Left",
            ha="left",
            va="top",
            transform=ax.transAxes,
            fontsize=12,
            weight="bold",
        )
        # Draw dots and lines
        # Add confidence interval on true p
        for i in range(len(y)):
            ax.fill_betweenx(
                [y[i] - deltay, y[i] + deltay],
                xDown_values[i],
                xUp_values[i],
                color="gray",
                alpha=0.3,
            )
        ax.hlines(y, 0, x_values, colors="black", lw=3, alpha=1)
        ax.scatter(x_values, y, color="black", marker="o", s=50)
        # Draw bars
        # lower_errors = [x-y for x,y in zip(x_values, xUp_values)]  # Lower error values
        # upper_errors = [x-y for x,y in zip(xDown_values, x_values)]  # Upper error values
        # asymmetric_errors = [lower_errors, upper_errors]  # Pairing the lower and upper errors
        # ax.barh(y, x_values, color='crimson', edgecolor='black', height=0.07, xerr=asymmetric_errors, capsize=0, alpha=0.9)  # Capsize adds horizontal lines at the error bars' tips
        ax.set_xlabel("-ln(p)", loc="left", fontsize=12)
        ax.set_xlim(ax.get_xlim()[::-1])  # Reverse the x-axis for left plot
        ax.set_xlim(right=0, left=6)
        dy = abs(y[-1] - y[-2])
        ax.set_ylim(y_min + dy * 2 / 3, y_max - dy * 2 / 3)
        ax.axvline(
            STAT.negP(0.05),
            0,
            0.94,
            color="black",
            lw=1,
            ls=":",
            alpha=0.5,
            label="95% C.L.",
        )
        ax.set_yticks([])
        ax.set_yticklabels([])
        ax.spines["top"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.text(
            0.69,
            0.98,
            r"$\alpha=0.05$",
            ha="right",
            va="top",
            transform=ax.transAxes,
            fontsize=11,
        )

        plt.subplots_adjust(wspace=0)  # Adjust space between subplots
        fig.savefig(
            f"{path_to_folder}pVal_left_{resolution}_Hz.png", bbox_inches="tight"
        )
        fig.savefig(
            f"{path_to_folder}pVal_left_{resolution}_Hz.svg", bbox_inches="tight"
        )
        plt.show(block=False)
        plt.pause(pause)
        plt.close()

        # Plot p-values Right only (Bootstrap)
        fig, ax = plt.subplots(figsize=(3, int(2 * N)))

        # Right results ---------
        xDown_values, x_values, xUp_values = montecarlo_error(p_right[1::2], nSim)
        # ---------
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
        # Draw dots and lines
        # Add confidence interval on true p
        for i in range(len(y)):
            ax.fill_betweenx(
                [y[i] - deltay, y[i] + deltay],
                xDown_values[i],
                xUp_values[i],
                color="gray",
                alpha=0.3,
            )
        ax.hlines(y, 0, x_values, colors="black", lw=3, alpha=1)
        ax.scatter(x_values, y, color="black", marker="o", s=50)
        # Draw bars
        # lower_errors = [x-y for x,y in zip(x_values, xUp_values)]  # Lower error values
        # upper_errors = [x-y for x,y in zip(xDown_values, x_values)]  # Upper error values
        # asymmetric_errors = [lower_errors, upper_errors]  # Pairing the lower and upper errors
        # ax.barh(y, x_values, color='darkturquoise', edgecolor='black', height=0.07, xerr=asymmetric_errors, capsize=0)  # Capsize adds horizontal lines at the error bars' tips
        ax.set_xlabel("-ln(p)", loc="right", fontsize=12)
        ax.set_xlim(ax.get_xlim()[::-1])  # Reverse the x-axis for right plot
        ax.set_xlim(left=0, right=6)
        dy = abs(y[-1] - y[-2])
        ax.set_ylim(y_min + dy * 2 / 3, y_max - dy * 2 / 3)
        ax.axvline(
            STAT.negP(0.05),
            0,
            0.94,
            color="black",
            lw=1,
            ls=":",
            alpha=0.5,
            label="95% C.L.",
        )
        ax.set_yticks([])
        ax.set_yticklabels([])
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.text(
            0.42,
            0.98,
            r"$\alpha=0.05$",
            ha="left",
            va="top",
            transform=ax.transAxes,
            fontsize=11,
        )

        plt.subplots_adjust(wspace=0)  # Adjust space between subplots
        fig.savefig(
            f"{path_to_folder}pVal_right_{resolution}_Hz.png", bbox_inches="tight"
        )
        fig.savefig(
            f"{path_to_folder}pVal_right_{resolution}_Hz.svg", bbox_inches="tight"
        )
        plt.show(block=False)
        plt.pause(pause)
        plt.close()

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # PLOT topoplots with r2
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Identify interpolated channels to show on the topomap
        mask = np.array(
            [True if x in old_ch_bads else False for x in ch_setSLAP.get_labels()]
        )
        mask_params1 = dict(marker="X", markersize=8, markerfacecolor="darkgrey")
        # Identify target channels to show on the topomap
        mask_right = np.array(
            [True if x in isLeft_ch else False for x in ch_setSLAP.get_labels()]
        )
        mask_left = np.array(
            [True if x in isRight_ch else False for x in ch_setSLAP.get_labels()]
        )
        mask_params2 = dict(
            marker="o", markersize=5, markerfacecolor="lime", alpha=0.75
        )

        # Plot
        fig = plt.figure(figsize=(6, int(2 * N)))
        gs = matplotlib.gridspec.GridSpec(N, 3, width_ratios=[2, 0.15, 2])
        # Make colormap for topoplots
        custom_cmap = PLOT.make_simple_cmap("blue", "white", "red")

        for i in range(N):
            # Create subplots in each row
            axes1 = fig.add_subplot(gs[i, 0])  # First  col
            axes2 = fig.add_subplot(gs[i, 1])  # Second col
            axes3 = fig.add_subplot(gs[i, 2])  # Third  col

            if i == 0:
                axes2.text(
                    -2.5,
                    1.35,
                    r"signed-r$^{2}$ Coefficients",
                    va="bottom",
                    ha="left",
                    transform=axes2.transAxes,
                    color="black",
                )
                axes2.text(
                    -0.5,
                    1.21,
                    "Target (O)",
                    va="bottom",
                    ha="right",
                    transform=axes2.transAxes,
                    color="limegreen",
                    fontsize=12,
                )
                axes2.text(
                    -0.5,
                    1.2,
                    "Target (O)",
                    va="bottom",
                    ha="right",
                    transform=axes2.transAxes,
                    color="black",
                    fontsize=12,
                )
                axes2.text(
                    1.5,
                    1.21,
                    "Interpolated (X)",
                    va="bottom",
                    ha="left",
                    transform=axes2.transAxes,
                    color="darkgrey",
                    fontsize=12,
                )
                axes2.text(
                    1.5,
                    1.2,
                    "Interpolated (X)",
                    va="bottom",
                    ha="left",
                    transform=axes2.transAxes,
                    color="black",
                    fontsize=12,
                )

            PLOT.plot_topomap_L_R(
                [axes1, axes2, axes3],
                RAW_SL,
                r2_left[i],
                r2_right[i],
                custom_cmap,
                (-0.3, 0.3),
                [mask, mask_left, mask_right],
                [mask_params1, mask_params2],
            )

            axes1.set_title(f"{freq_band[i]}-{freq_band[i+1]-1} Hz (Left)", fontsize=12)
            axes3.set_title(
                f"{freq_band[i]}-{freq_band[i+1]-1} Hz (Right)", fontsize=12
            )

        fig.subplots_adjust(wspace=0)
        fig.savefig(f"{path_to_folder}topoR2_{resolution}_Hz.png", bbox_inches="tight")
        fig.savefig(f"{path_to_folder}topoR2_{resolution}_Hz.svg", bbox_inches="tight")
        plt.show(block=False)
        plt.pause(pause)
        plt.close()

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # PLOT channel x-ray
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Generate array of trials, rest followed by task
        x_list = [
            np.vstack([psds_left_rest, psds_left]),  # (trial, ch, bin)]
            np.vstack([psds_right_rest, psds_right]),
        ]  # (trial, ch, bin)]
        # Generate labels for rest (True) and task (False)
        isTreatment_list = [
            np.arange(x_list[0].shape[0]) < psds_left_rest.shape[0],  # (trial,)
            np.arange(x_list[1].shape[0]) < psds_right_rest.shape[0],
        ]  # (trial,)

        # List of channels to show plots for (comes from option -xray)
        for ch in xray:
            # Find channel id
            ch_idx = ch_set.find_labels(ch)[0]

            # xlim to use in plots based on frequency range used
            xlim = [STAT.bins[0], STAT.bins[-1]]

            # Masks for target electrodes
            mask = np.array(
                [True if x == ch else False for x in ch_setSLAP.get_labels()]
            )
            mask_params = dict(marker="o", markersize=6, markerfacecolor="lime")

            # Define mid points and width of frequency bands to consider
            x_band = []
            w_band = []
            for i in range(N):
                # Frequency bands mid points
                x_band.append(freq_band[i] + (freq_band[i + 1] - freq_band[i]) / 2)
                # Frequency bands width
                w_band.append(freq_band[i + 1] - freq_band[i])

            # Generate plots for Left and Right trials
            for k, mode in enumerate(["Left", "Right"]):
                # Generate the canvases
                figTopo, axsTopo = plt.subplots(
                    1, N + 1, figsize=(int(2 * N + 1), 2)
                )  # Canvas for topoplots
                figPSD, axsPSD = plt.subplots(
                    3, 1, figsize=(6, int(2 * (3)))
                )  # Canvas for PSDs
                figCorr, axsCorr = plt.subplots(
                    N, 1, figsize=(4, int(2 * N))
                )  # Canvas for correlation coeffs

                # Define which trials to consider
                x = x_list[k]
                isTreatment = isTreatment_list[k]

                # Rest PSD trials
                xs, y_rest, ws = PLOT.plot_psd_at_channel(
                    x=x[:, [ch_idx], :][isTreatment],
                    color="navy",
                    ax=axsPSD[0],
                    freq_band=freq_band,
                    bins=STAT.bins,
                )
                # Use Rest PSD trials to define y limits in dB
                ylim = (
                    int(np.min(y_rest) - np.abs(np.min(y_rest)) - 15),
                    int(np.max(y_rest) + np.abs(np.max(y_rest)) + 15),
                )
                # Plot (Rest PSD trials)
                PLOT.plot_frequency_bands(ax=axsPSD[0], ylim=ylim, fontsize=10)
                axsPSD[0].set_xticks([])
                axsPSD[0].set_xlim(xlim)
                axsPSD[0].set_ylim(ylim)
                axsPSD[0].set_ylabel("[dB]", loc="top", fontsize=11)
                axsPSD[0].text(
                    xlim[0] + 0.25,
                    ylim[0] + (ylim[1] - ylim[0]) * 0.025,
                    f"Rest Trials ({mode})",
                )

                # Lists to store information for each frequency band
                ys = []
                r_coeffs = []
                cmap_max = 1

                # Generate plots for each frequency band
                for i in range(N):
                    idx = np.where(
                        (STAT.bins >= freq_band[i]) & (STAT.bins < freq_band[i + 1])
                    )[0]
                    start_idx = idx[0]
                    end_idx = idx[-1]
                    x_within_band = np.mean(
                        x[:, :, start_idx:end_idx], axis=2
                    )  # (trial, ch)
                    x_within_band = EEG.convert_dB(
                        x_within_band
                    )  # Transform PSDs to dB
                    r_coeff = STAT.CalculateR(x=x_within_band, isTreatment=isTreatment)
                    r_coeffs.append(r_coeff)
                    ys.append(r_coeff[[ch_idx]])

                    # Band r topoplot
                    mne.viz.plot_topomap(
                        r_coeff,
                        RAW_SL.info,
                        ch_type="eeg",
                        sensors=True,
                        cmap=PLOT.simple_cmap,
                        vlim=(-cmap_max, cmap_max),
                        mask=mask,
                        mask_params=mask_params,
                        show=False,
                        axes=axsTopo[i],
                    )

                    # Convert labels of trials into dummy variable (True/Rest = 0, False/Task = 1 by choice)
                    dummy = STAT.Dummy(isTreatment)
                    PLOT.plot_correlation_psd_groups(
                        x=x_within_band[:, [ch_idx]],
                        y=dummy,
                        isTreatment=isTreatment,
                        r=r_coeff[[ch_idx]],
                        xlim=ylim,
                        ax=axsCorr[i],
                    )

                    # Show frequency band (topoplot)
                    axsTopo[i].set_title(
                        f"[{freq_band[i]}, {freq_band[i+1]-1}] Hz", fontsize=12
                    )

                    # Show frequency band (correlation)
                    axsCorr[i].text(
                        0.98,
                        0.95,
                        f"[{freq_band[i]}, {freq_band[i+1]-1}] Hz",
                        va="top",
                        ha="right",
                        transform=axsCorr[i].transAxes,
                        fontsize=12,
                    )

                    axsCorr[i].set_xlim(ylim)
                    if i != N - 1:
                        axsCorr[i].set_xticks([])

                # Task PSD trials
                xs, y_task, ws = PLOT.plot_psd_at_channel(
                    x=x[:, [ch_idx], :][~isTreatment],
                    color="green",
                    ax=axsPSD[1],
                    freq_band=freq_band,
                    bins=STAT.bins,
                )
                # Plot (Task PSD trials)
                PLOT.plot_frequency_bands(ax=axsPSD[1], ylim=None)
                axsPSD[1].set_xticks([])
                axsPSD[1].set_xlim(xlim)
                axsPSD[1].set_ylim(ylim)
                axsPSD[1].set_ylabel("[dB]", loc="top", fontsize=11)
                axsPSD[1].text(
                    xlim[0] + 0.25,
                    ylim[0] + (ylim[1] - ylim[0]) * 0.025,
                    f"Task Trials ({mode})",
                )

                # Rest - Task PSD (mean) trials
                y_diff = [np.mean(i) - np.mean(j) for i, j in zip(y_rest, y_task)]
                ymax = int(np.max(np.abs(y_diff)) * 2)
                # Plot (Rest - Task PSD (mean) trials)
                for i, y in enumerate(y_diff):
                    if i == 0:
                        axsPSD[2].plot(
                            [x_band[i] - w_band[i] / 2, x_band[i] + w_band[i] / 2],
                            [y, y],
                            color="blue",
                            label="Rest - Task",
                        )
                    else:
                        axsPSD[2].plot(
                            [x_band[i] - w_band[i] / 2, x_band[i] + w_band[i] / 2],
                            [y, y],
                            color="blue",
                        )
                axsPSD[2].axhline(0, lw=1, ls="--", color="grey")
                axsPSD[2].set_xlim(xlim)
                axsPSD[2].set_xlabel("$f$ [Hz]", loc="right", fontsize=11)
                ylim_diff = np.max(np.abs(y_diff)) * 1.5
                axsPSD[2].set_ylim(-ylim_diff, ylim_diff)
                axsPSD[2].set_ylabel("Difference [dB]", fontsize=11, loc="top")
                PLOT.plot_frequency_bands(ax=axsPSD[2], ylim=None)
                # Add r2 on top of PSD difference
                ys = [x * abs(x) * 10 for x in ys]
                color = "magenta"
                axsPSD[2].plot(
                    x_band,
                    ys,
                    "-o",
                    color=color,
                    markersize=4,
                    label=r"signed-r$^{2}$ [$\times$10]",
                )
                axsPSD[2].scatter(x_band, ys, color="black", s=60)
                axsPSD[2].legend(loc="lower right", frameon=True)
                # Add new axis on the right
                # ax = axsPSD[2].twinx()  # instantiate a second axes that shares the same x-axis
                # ax.set_ylabel(r'[$\times$10]  r$^{2}$', color=color)  # we already handled the x-label with ax1
                # ax.tick_params(axis='y', labelcolor=color, color=color)
                # ax.set_ylim(-1,1)

                # Show channel (correlation)
                axsCorr[0].text(
                    0,
                    1.1,
                    f"Channel {ch}",
                    weight="bold",
                    va="top",
                    ha="left",
                    transform=axsCorr[0].transAxes,
                    fontsize=12,
                )
                axsCorr[0].text(
                    0.02,
                    0.95,
                    f"{mode} trials",
                    va="top",
                    ha="left",
                    transform=axsCorr[0].transAxes,
                    fontsize=12,
                )
                # axsCorr[0].set_title(r'Coeff. r & r$^{2}$', fontsize=11, loc='right')
                axsCorr[0].text(
                    0.995,
                    1.14,
                    text_id,
                    ha="right",
                    va="top",
                    transform=axsCorr[0].transAxes,
                    fontsize=11,
                    bbox=dict(
                        facecolor="white", edgecolor="black", boxstyle="square, pad=0.1"
                    ),
                )

                # Show channel (PSD)
                axsPSD[0].text(
                    0,
                    1.1,
                    f"Channel {ch}",
                    weight="bold",
                    va="top",
                    ha="left",
                    transform=axsPSD[0].transAxes,
                    fontsize=12,
                )
                axsPSD[0].text(
                    0.995,
                    1.14,
                    text_id,
                    ha="right",
                    va="top",
                    transform=axsPSD[0].transAxes,
                    fontsize=11,
                    bbox=dict(
                        facecolor="white", edgecolor="black", boxstyle="square, pad=0.1"
                    ),
                )

                # Add color bar (topoplot)
                clim = dict(kind="value", lims=[-cmap_max, 0, cmap_max])
                divider = make_axes_locatable(axsTopo[-1])
                axsTopo[-1].set_yticks([])
                axsTopo[-1].set_xticks([])
                axsTopo[-1].axis("off")
                axsTopo[-1] = divider.append_axes(position="left", size="20%", pad=0.5)
                mne.viz.plot_brain_colorbar(
                    axsTopo[-1],
                    clim=clim,
                    colormap=PLOT.simple_cmap,
                    transparent=False,
                    orientation="vertical",
                    label=None,
                )

                axsTopo[-1].text(
                    4,
                    1,
                    text_id,
                    ha="left",
                    va="top",
                    transform=axsTopo[-1].transAxes,
                    fontsize=11,
                    bbox=dict(
                        facecolor="white", edgecolor="black", boxstyle="square,pad=0.1"
                    ),
                )
                axsTopo[-1].text(4, 0.5, r"$r$ Coefficients", fontsize=12)
                axsTopo[-1].text(4, 0.3, f"Channel {ch}", weight="bold", fontsize=12)
                axsTopo[-1].text(4, 0.1, f"{mode} trials", fontsize=12)
                axsTopo[-1].text(4, -0.099, f"Target (O)", color="lime", fontsize=12)
                axsTopo[-1].text(4, -0.1, f"Target (O)", color="black", fontsize=12)

                figCorr.subplots_adjust(hspace=0)
                figPSD.subplots_adjust(hspace=0)

                figTopo.savefig(
                    f"{path_to_folder}topoR_{ch}_{mode}_{resolution}_Hz.png",
                    bbox_inches="tight",
                )
                figTopo.savefig(
                    f"{path_to_folder}topoR_{ch}_{mode}_{resolution}_Hz.svg",
                    bbox_inches="tight",
                )

                figPSD.savefig(
                    f"{path_to_folder}psd_{ch}_{mode}_{resolution}_Hz.png",
                    bbox_inches="tight",
                )
                figPSD.savefig(
                    f"{path_to_folder}psd_{ch}_{mode}_{resolution}_Hz.svg",
                    bbox_inches="tight",
                )

                figCorr.savefig(
                    f"{path_to_folder}corr_{ch}_{mode}_{resolution}_Hz.png",
                    bbox_inches="tight",
                )
                figCorr.savefig(
                    f"{path_to_folder}corr_{ch}_{mode}_{resolution}_Hz.svg",
                    bbox_inches="tight",
                )

                figTopo.show()
                figPSD.show()
                figCorr.show()

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # PLOT channel PSDs
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        min_ = np.min(EEG.convert_dB(x[:, ch_setSLAP.find_labels(xray), :]))
        max_ = np.max(EEG.convert_dB(x[:, ch_setSLAP.find_labels(xray), :]))

        min_ = int(min_ - np.abs(min_) * 0.05)
        max_ = int(max_ + np.abs(max_) * 0.02)

        def find_multiples_of_n(start, end, n):
            # Calculate the starting multiple of n greater than or equal to 'start'
            if start % n != 0:
                start_multiple = start + (n - start % n)
            else:
                start_multiple = start

            # Calculate the ending multiple of n less than or equal to 'end'
            if end % n != 0:
                end_multiple = end - (end % n)
            else:
                end_multiple = end

            # Generate the list of multiples from the start_multiple to the end_multiple
            multiples = list(range(start_multiple, end_multiple + 1, n))
            return multiples

        edges = find_multiples_of_n(min_, max_, 5)

        # xlim to use in plots based on frequency range used
        xlim = [STAT.bins[0], STAT.bins[-1]]

        for ch in xray:
            # Find channel id
            ch_idx = ch_setSLAP.find_labels(ch)[0]

            fig = plt.figure(figsize=(5, 7))
            # Define the grid layout
            gs = matplotlib.gridspec.GridSpec(2, 1, height_ratios=[3, 1])

            # Left and Right (Task)
            x_left = psds_left[:, ch_idx, :]
            x_right = psds_right[:, ch_idx, :]
            # Left and Right (Rest)
            x_left_rest = psds_left_rest[:, ch_idx, :]
            x_right_rest = psds_right_rest[:, ch_idx, :]

            # All (Rest)
            x_rest = np.vstack([x_left_rest, x_right_rest])

            def mean_and_se(x=None, axis=0):
                mean = np.mean(x, axis=axis)
                std = np.std(x, axis=axis)
                se = std / np.sqrt(x.shape[axis])

                return mean, se

            ax1 = fig.add_subplot(gs[0, 0])

            mean, se = mean_and_se(x=EEG.convert_dB(x_left), axis=0)
            ax1.fill_between(STAT.bins, mean - se, mean + se, color="blue", alpha=0.5)
            ax1.plot(STAT.bins, mean, color="blue", label="Move Left", alpha=1)

            mean, se = mean_and_se(x=EEG.convert_dB(x_left_rest), axis=0)
            ax1.fill_between(
                STAT.bins, mean - se, mean + se, color="dodgerblue", alpha=0.25
            )
            ax1.plot(STAT.bins, mean, color="dodgerblue", label="Rest Left", alpha=0.35)

            mean, se = mean_and_se(x=EEG.convert_dB(x_right), axis=0)
            ax1.fill_between(STAT.bins, mean - se, mean + se, color="red", alpha=0.5)
            ax1.plot(STAT.bins, mean, color="red", label="Move Right", alpha=1)

            mean, se = mean_and_se(x=EEG.convert_dB(x_right_rest), axis=0)
            ax1.fill_between(
                STAT.bins, mean - se, mean + se, color="magenta", alpha=0.25
            )
            ax1.plot(STAT.bins, mean, color="magenta", label="Rest Right", alpha=0.35)

            # mean, se = mean_and_se(x=x_rest, axis=0)
            # ax1.fill_between(STAT.bins, EEG.convert_dB(mean-se), EEG.convert_dB(mean+se), color='springgreen', alpha=0.5)
            # ax1.plot(STAT.bins, EEG.convert_dB(mean), color='green', label="Rest", alpha=0.75)
            # mean, se = mean_and_se(x=EEG.convert_dB(x_rest), axis=0)
            # ax1.fill_between(STAT.bins, mean-se, mean+se, color='springgreen', alpha=0.5)
            # ax1.plot(STAT.bins, mean, color='green', label="Rest", alpha=0.75)

            ax1.set_xlim(xlim[0], xlim[1])
            ax1.set_ylim(min_, max_)
            ax1.set_xscale("log")
            for i in [2, 4, 6, 8, 10, 20, 30]:
                ax1.axvline(i, lw=1, ls=":", color="grey", alpha=0.4)
            for i in edges:
                ax1.axhline(i, lw=1, ls=":", color="grey", alpha=0.4)
            legend = ax1.legend(loc="lower left")

            # Define font properties for bold text
            bold_font = FontProperties(weight="bold")
            # Create a legend and make only one label bold
            for text in legend.get_texts():
                if ch == "c3":
                    if (
                        text.get_text() == "Move Right"
                        or text.get_text() == "Rest Right"
                    ):
                        text.set_fontproperties(bold_font)
                elif ch == "c4":
                    if text.get_text() == "Move Left" or text.get_text() == "Rest Left":
                        text.set_fontproperties(bold_font)

            ax1.set_xticks([])
            ax1.set_ylabel("[dB]", loc="top")
            ax1.text(
                0,
                1.05,
                f"Channel {ch}",
                weight="bold",
                va="top",
                ha="left",
                transform=ax1.transAxes,
                fontsize=12,
            )

            PLOT.plot_frequency_bands(
                ax=ax1, ylim=(min_, max_), fontsize=12, fraction=0.07
            )

            ax2 = fig.add_subplot(gs[1, 0])
            x_ = np.vstack([x_left_rest, x_left])

            isTreatment = np.arange(x_.shape[0]) < x_left_rest.shape[0]

            ax2.plot(
                STAT.bins,
                STAT.CalculateR2(EEG.convert_dB(x_), isTreatment, signed=False),
                color="blue",
                label="Left",
            )

            x_ = np.vstack([x_right_rest, x_right])

            isTreatment = np.arange(x_.shape[0]) < x_right_rest.shape[0]
            ax2.plot(
                STAT.bins,
                STAT.CalculateR2(EEG.convert_dB(x_), isTreatment, signed=False),
                color="red",
                label="Right",
            )

            ax2.set_xlim(xlim[0], xlim[1])
            ax2.set_ylim(0, 0.8)
            ax2.set_xscale("log")
            for i in [2, 4, 6, 8, 10, 20, 30]:
                ax2.axvline(i, lw=1, ls=":", color="grey", alpha=0.4)
            for i in [0.2, 0.4, 0.6, 0.8]:
                ax2.axhline(i, lw=1, ls=":", color="grey", alpha=0.4)
            legend = ax2.legend(loc="upper left")

            # Define font properties for bold text
            bold_font = FontProperties(weight="bold")
            # Create a legend and make only one label bold
            for text in legend.get_texts():
                if ch == "c3":
                    if text.get_text() == "Right":
                        text.set_fontproperties(bold_font)
                elif ch == "c4":
                    if text.get_text() == "Left":
                        text.set_fontproperties(bold_font)

            ax2.set_xlabel("Frequency [Hz]", loc="right")
            ax2.set_ylabel(r"R$^{2}$")

            custom_ticks = [2, 4, 6, 8, 10, 20, 30, 40]
            # Set the ticks on the x-axis
            ax2.set_xticks(custom_ticks)  # Set custom ticks
            ax2.set_xticklabels(custom_ticks)  # Set labels as the same values

            custom_ticks = [0, 0.2, 0.4, 0.6]
            # Set the ticks on the x-axis
            ax2.set_yticks(custom_ticks)  # Set custom ticks
            ax2.set_yticklabels(custom_ticks)  # Set labels as the same values

            PLOT.plot_frequency_bands(ax=ax2, ylim=None, fontsize=12)
            fig.subplots_adjust(hspace=0)
            fig.savefig(
                f"{path_to_folder}{ch}_PSDs_{resolution}_Hz.png", bbox_inches="tight"
            )
            fig.savefig(
                f"{path_to_folder}{ch}_PSDs_{resolution}_Hz.svg", bbox_inches="tight"
            )
            plt.show(block=False)
            plt.pause(pause)
            plt.close()

        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # GENERATE PDF REPORT (after generating the plots)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        PdfReport.generate_pdf(
            plot_folder,
            montage_name,
            resolution,
            yob=yob,
            date_test=date_test,
            version=__version__,
        )

    else:
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # GENERATE PDF REPORT (no plots generation, use the existing ones in the folder)
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        PdfReport.generate_pdf(
            plot_folder,
            montage_name,
            resolution,
            yob=yob,
            date_test=date_test,
            version=__version__,
        )
