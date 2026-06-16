from __future__ import annotations

import argparse
from pathlib import Path

from BridgingChecker import BridgingChecker
from RawImporter import EEGRawImporter


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser"""
    parser = argparse.ArgumentParser(
        prog="run_BridgingChecker",
        description="Run candidate bridged-channel analysis on a BCI2000 .dat EEG file",
    )

    parser.add_argument("--file-path", required=True, help="Path to the input .dat file")
    parser.add_argument("--helper-dir", default="./helper", help="Folder containing helper montage files")
    parser.add_argument("--save-path", required=True, help="Base folder where bridging outputs are saved")

    parser.add_argument("--fmin", type=float, default=1.0, help="Band-pass low cutoff in Hz")
    parser.add_argument("--fmax", type=float, default=40.0, help="Band-pass high cutoff in Hz")
    parser.add_argument("--sigma", type=float, default=0.05, help="Gaussian affinity kernel sigma")
    parser.add_argument("--window-sec", type=float, default=10.0, help="Bridge-analysis window length in seconds")
    parser.add_argument(
        "--bridge-score-threshold",
        type=float,
        default=0.095,
        help="Threshold on correlation x affinity bridge score",
    )
    parser.add_argument("--show-extra", action="store_true", help="Show diagnostic matrix plots")
    parser.add_argument("--keep-stim", action="store_true", help="Keep BCI2000 state channels during import")
    parser.add_argument("--verbose", action="store_true", help="Print detailed progress information")

    return parser


def _print_header(title: str) -> None:
    print(f"\n=== {title} ===")


def _output_dir(base_dir: str, file_path: str) -> Path:
    return Path(base_dir).expanduser() / Path(file_path).expanduser().stem


def main(argv: list[str] | None = None) -> None:
    """Run the bridge checker from the command line"""
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    try:
        _print_header("Importing EEG")
        importer = EEGRawImporter(
            path_to_file=args.file_path,
            helper_dir=args.helper_dir,
            keep_stim=args.keep_stim,
            verbose=args.verbose,
        )

        raw = importer.raw
        if raw is None:
            raise RuntimeError("import finished without an MNE RawArray")

        save_dir = _output_dir(args.save_path, args.file_path)

        _print_header("Running bridging analysis")
        checker = BridgingChecker(
            raw=raw,
            verbose=args.verbose,
            fmin=args.fmin,
            fmax=args.fmax,
            sigma=args.sigma,
            window_sec=args.window_sec,
            bridge_score_threshold=args.bridge_score_threshold,
            show_extra=args.show_extra,
            save_path=save_dir,
        )

    except Exception as exc:
        parser.exit(status=1, message=f"[run_BridgingChecker] ERROR: {exc}\n")

    _print_header("Bridging summary")
    print(f"Input file          : {args.file_path}")
    print(f"Montage type        : {importer.montage.get('montage_type')}")
    print(f"EEG channels        : {len(checker.ch_names)}")
    print(f"Windows analyzed    : {len(checker.windows)}")
    print(f"Candidate groups    : {len(checker.groups)}")

    if checker.group_channel_names:
        for group_id, group in enumerate(checker.group_channel_names, start=1):
            print(f"Group {group_id:<12}: {', '.join(group)}")
    else:
        print("Group details       : none")

    print(f"Saved outputs       : {save_dir}")


if __name__ == "__main__":
    main()
