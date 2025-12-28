import os
from pathlib import Path
import matplotlib.pyplot as plt
from RawImporter     import EEGRawImporter
from BridgingChecker import BridgingChecker
from Preprocessing   import EEGPreprocessor, EEGPreprocessorConfig
from MotorImagery    import EEGMotorImagery
from PdfReport       import MotorImageryPdfReport

def _ensure_dir(p: Path) -> None:
    """
    Create folder and parents if they do not exist
    If parents is True, any missing parents of this path are created as needed
    If exist_ok is False, FileExistsError is raised if the target directory already exists
    """
    p.mkdir(parents=True, exist_ok=True)

helper_dir = "./helper"
root       = "/mnt/c/Users/scana/Desktop/motorimagery_to_run"
#file_path = os.path.join(root, "sub-PDHC002_ses-01_task-MotorImag_run-01.dat")     # EGI128
#file_path = os.path.join(root, "sub-PDHC002_ses-01_task-MotorImag_run-02.dat")     # EGI128
file_path = os.path.join(root, "sub-PDHC034_ses-01_task-MotorImag_run-01.dat")     # EGI64
#file_path = os.path.join(root, "sub-DOCpeds003_ses-01_task-MotorImag_run-01.dat")  # EGI128
#file_path = os.path.join(root, "sub-PDNG025_ses-01_task-HillOddBall_run-04.dat")   # GTEC32


file_name = file_path.split('/')[-1]
base_name, extension = os.path.splitext(file_name)
sub_name = base_name.split("sub-")[1].split("_ses")[0]
ses_name = base_name.split("ses-")[1].split("_")[0]


save_path = os.path.join(root, base_name)
_ensure_dir(Path(save_path))


imp = EEGRawImporter(
    path_to_file = file_path,
    helper_dir   = helper_dir,
    keep_stim    = True,
    verbose      = False,
)
raw          = imp.raw
ch_set       = imp.ch_set
montage_type = imp.montage['montage_type']
date_test    = imp.stream['date_test']


bc = BridgingChecker(
        raw                    = raw,
        verbose                = False,
        fmin                   = 1.0,
        fmax                   = 40.0,
        sigma                  = 0.05,
        window_sec             = 10.0,
        bridge_score_threshold = 0.095,
        show_extra             = False,
        figure                 = None,      
        axes                   = None, 
        save_path              = save_path,
)


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
    run_prep         = False,
    prep_correlation = True,
    prep_deviation   = True,
    prep_hf_noise    = True,
    prep_nan_flat    = True,
    prep_ransac      = True,
    # Manual annotation
    run_annotation = True,
    plot           = True,
    # Interpolation
    run_interpolation       = True,
    reset_bads_after_interp = True,
    # Re-reference 
    run_rereference = True,
    reref_channels  = "tp9 tp10",
    # Spatial filter
    run_spatialfilter = True,
    spatial_exclude   = None,
    # Random state
    random_state = 83092,
)
preproc = EEGPreprocessor(raw, ch_set, config=config, copy=True, montage_type=montage_type, verbose=False)
preproc.run()
raw     = preproc.raw
history = preproc.history
ch_set  = preproc.ch_set


resolution = 1

mi = EEGMotorImagery(
    raw,
    ch_set,
    nEpochs=6, 
    duration_task=10., 
    skip=1., 
    resolution=resolution,
    freq_bands=[4.0, 8.0, 13.0, 31.0],
    nSim=2999,
    copy=True,
    verbose=False,
    save_path=save_path,
)


pdf = MotorImageryPdfReport(
        plot_folder=save_path,
        helper_folder=helper_dir,
        date_test=date_test,
        montage_name=montage_type,
        resolution=resolution,
        age_at_test="20",
        save_folder=save_path,
)