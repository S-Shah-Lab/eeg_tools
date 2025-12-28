from RawImporter   import EEGRawImporter
#from QualityChecker import QualityChecker
from BridgingChecker import BridgingChecker

file_path = "/mnt/c/Users/scana/Desktop/motorimagery_to_run/sub-PDHC034_ses-01_task-MotorImag_run-01.dat"     # EGI64
#file_path = "/mnt/c/Users/scana/Desktop/motorimagery_to_run/sub-DOCpeds003_ses-01_task-MotorImag_run-01.dat"  # EGI128
#file_path = "/mnt/c/Users/scana/Desktop/motorimagery_to_run/sub-PDNG025_ses-01_task-HillOddBall_run-04.dat"  # GTEC32

#file_path = "/mnt/c/Users/scana/Desktop/sub-testGS_ses-13_task-HillOddBall_run-01.dat"
#file_path = "/mnt/c/Users/scana/Desktop/sub-testGS_ses-13_task-HillOddBall_run-02.dat"
file_path = "/mnt/c/Users/scana/Desktop/sub-testGS_ses-13_task-HillOddBall_run-03.dat"

#file_path = "/mnt/c/Users/scana/Desktop/motorimagery_to_run/sub-DOCpeds003_ses-01_task-MotorImag_run-01.dat"
#file_path = "/mnt/c/Users/scana/Desktop/motorimagery_to_run/sub-DOCpeds003_ses-02_task-MotorImag_run-01.dat"
#file_path = "/mnt/c/Users/scana/Desktop/motorimagery_to_run/sub-PDHC034_ses-01_task-MotorImag_run-01.dat"

#file_path = "/mnt/c/Users/scana/Desktop/motorimagery_to_run/sub-PDNG025_ses-01_task-HillOddBall_run-01.dat"
#file_path = "/mnt/c/Users/scana/Desktop/motorimagery_to_run/sub-PDNG025_ses-01_task-HillOddBall_run-02.dat"
#file_path = "/mnt/c/Users/scana/Desktop/motorimagery_to_run/sub-PDNG025_ses-01_task-HillOddBall_run-03.dat"
#file_path = "/mnt/c/Users/scana/Desktop/motorimagery_to_run/sub-PDNG025_ses-01_task-HillOddBall_run-04.dat"
#file_path = "/mnt/c/Users/scana/Desktop/motorimagery_to_run/sub-PDNG025_ses-01_task-NatLang_run-01.dat"
#file_path = "/mnt/c/Users/scana/Desktop/motorimagery_to_run/sub-PDNG025_ses-01_task-Rest_run-01.dat"


helper_dir = "/mnt/c/Users/scana/Dropbox/WCornell/develop/eeg_tools/helper"

imp = EEGRawImporter(
    path_to_file = file_path,
    helper_dir   = helper_dir,
    keep_stim    = False,
    verbose      = False,
)

raw    = imp.raw

bc = BridgingChecker(
        raw=raw,
        verbose=False,
        fmin=1.0,
        fmax=40.0,
        sigma=0.05,
        window_sec=10.0,
        bridge_score_threshold=0.095,
        show_extra=False,
        figure=None,      
        axes=None, 
        save_path=None,
)