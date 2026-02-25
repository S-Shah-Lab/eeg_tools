from __future__ import annotations
import argparse, os
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
        required=True,
        help="Folder to save bridging plots into",
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
    )

    file_name = args.file_path.split('/')[-1].split('.')[0]

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
        save_path=os.path.join(args.save_path, file_name),
    )

    print("Bridging analysis complete")
    if args.save_path:
        print("Saved outputs to:", args.save_path)


if __name__ == "__main__":
    main()