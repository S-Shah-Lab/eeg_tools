from RawImporter import EEGRawImporter

file_path = "/mnt/c/Users/scana/Desktop/motorimagery_to_run/sub-PDHC034_ses-01_task-MotorImag_run-01.dat"     # EGI64
#file_path = "/mnt/c/Users/scana/Desktop/motorimagery_to_run/sub-DOCpeds003_ses-01_task-MotorImag_run-01.dat"  # EGI128
#file_path = "/mnt/c/Users/scana/Desktop/motorimagery_to_run/sub-PDNG025_ses-01_task-HillOddBall_run-04.dat"  # GTEC32

helper_dir = "/mnt/c/Users/scana/Dropbox/WCornell/develop/eeg_tools/helper"

imp = EEGRawImporter(
    path_to_file = file_path,
    helper_dir   = helper_dir,
    keep_stim    = True,
    verbose      = True,
)