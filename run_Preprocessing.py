from RawImporter   import EEGRawImporter
from Preprocessing import EEGPreprocessor, EEGPreprocessorConfig

file_path = "/mnt/c/Users/scana/Desktop/motorimagery_to_run/sub-PDHC034_ses-01_task-MotorImag_run-01.dat"     # EGI64
#file_path = "/mnt/c/Users/scana/Desktop/motorimagery_to_run/sub-DOCpeds003_ses-01_task-MotorImag_run-01.dat"  # EGI128
#file_path = "/mnt/c/Users/scana/Desktop/motorimagery_to_run/sub-PDNG025_ses-01_task-HillOddBall_run-04.dat"  # GTEC32

helper_dir = "/mnt/c/Users/scana/Dropbox/WCornell/develop/eeg_tools/helper"

imp = EEGRawImporter(
    path_to_file = file_path,
    helper_dir   = helper_dir,
    keep_stim    = True,
    verbose      = False,
)

raw          = imp.raw
ch_set       = imp.ch_set
montage_type = imp.montage["montage_type"]

config = EEGPreprocessorConfig(
    # Notch
    run_notch    = True,
    notch_freqs  = 60,
    notch_kwargs = {},
    
    # Band-pass
    run_bandpass    = True,
    l_freq          = 1,
    h_freq          = 40,
    bandpass_kwargs = {},

    # PREP
    run_prep         = True,
    prep_correlation = True,
    prep_deviation   = True,
    prep_hf_noise    = True,
    prep_nan_flat    = True,
    prep_ransac      = True,

    # Manual annotation
    run_annotation        = True,
    plot                  = True,

    # Interpolation
    run_interpolation       = True,
    reset_bads_after_interp = True,

    # Re-reference
    run_rereference = True,
    reref_channels  = "tp9 tp10",

    # Spatial filter
    run_spatialfilter   = True,
    spatial_exclude     = None,

    random_state = 83092,
)

preproc = EEGPreprocessor(raw, ch_set, config=config, copy=True, montage_type=montage_type, verbose=False)
preproc.run()
raw     = preproc.raw
history = preproc.history
ch_set  = preproc.ch_set
