from __future__ import annotations
import argparse
from RawImporter import EEGRawImporter
from BridgingChecker import BridgingChecker


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser"""
    parser = argparse.ArgumentParser(
        prog="run_BridgingChecker",
        description="Run bridging analysis (candidate bridged EEG channels) on a .dat file",
    )

    parser.add_argument(
        "--file-path",
        required=True,
        help="Path to the input .dat file",
    )
    parser.add_argument(
        "--helper-dir",
        default="./helper",
        help="Folder containing helper montage files",
    )
    parser.add_argument(
        "--keep-stim",
        action="store_true",
        help="Keep stimulus/state channels when importing",
    )
    parser.add_argument(
        "--import-verbose",
        action="store_true",
        help="Verbose output during import",
    )

    # BridgingChecker parameters
    parser.add_argument("--fmin",  type=float, default=1.0,  help="Band-pass low cutoff (Hz)")
    parser.add_argument("--fmax",  type=float, default=40.0, help="Band-pass high cutoff (Hz)")
    parser.add_argument("--sigma", type=float, default=0.05, help="Gaussian kernel sigma")
    parser.add_argument(
        "--window-sec",
        type=float,
        default=10.0,
        help="Sliding window length in seconds",
    )
    parser.add_argument(
        "--bridge-score-threshold",
        type=float,
        default=0.095,
        help="Threshold on correlation x affinity bridge score",
    )
    parser.add_argument(
        "--show-extra",
        action="store_true",
        help="Show extra bridging matrices (debug/inspection)",
    )
    parser.add_argument(
        "--save-path",
        default=None,
        help="Optional folder to save bridging plots into",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose output during bridging analysis",
    )

    return parser


def main(argv: list[str] | None = None) -> None:
    """Run bridging checker from the command line"""
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    imp = EEGRawImporter(
        path_to_file=args.file_path,
        helper_dir=args.helper_dir,
        keep_stim=args.keep_stim,
        verbose=args.import_verbose,
    )

    BridgingChecker(
        raw=imp.raw,
        verbose=args.verbose,
        fmin=args.fmin,
        fmax=args.fmax,
        sigma=args.sigma,
        window_sec=args.window_sec,
        bridge_score_threshold=args.bridge_score_threshold,
        show_extra=args.show_extra,
        figure=None,
        axes=None,
        save_path=args.save_path,
    )

    print("Bridging analysis complete")
    if args.save_path:
        print("Saved outputs to:", args.save_path)


if __name__ == "__main__":
    main()

    '''
    To run in VSCode Interactive Window 
    #file_path = "/mnt/c/Users/scana/Desktop/motorimagery_to_run/sub-PDHC034_ses-01_task-MotorImag_run-01.dat"     # EGI64
    #file_path = "/mnt/c/Users/scana/Desktop/motorimagery_to_run/sub-DOCpeds003_ses-01_task-MotorImag_run-01.dat"  # EGI128
    #file_path = "/mnt/c/Users/scana/Desktop/motorimagery_to_run/sub-PDNG025_ses-01_task-HillOddBall_run-04.dat"  # GTEC32
    #helper_dir = "/mnt/c/Users/scana/Dropbox/WCornell/develop/eeg_tools/helper"

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
    '''