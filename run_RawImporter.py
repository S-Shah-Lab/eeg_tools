from __future__ import annotations
import argparse
from RawImporter import EEGRawImporter



def build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser"""
    parser = argparse.ArgumentParser(
        prog="run_RawImporter",
        description=(
            "Import a BCI2000 .dat file and construct an MNE Raw object with montage"
        ),
    )

    parser.add_argument(
        "--file-path",
        required=True,
        help="Path to the input .dat file",
    )
    parser.add_argument(
        "--helper-dir",
        default="./helper",
        help=(
            "Folder containing helper montage files (e.g., *location.txt) "
            "Defaults to the previous hard-coded path"
        ),
    )
    parser.add_argument(
        "--keep-stim",
        action="store_true",
        help="Keep stimulus/state channels when building the Raw object",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print additional information during import",
    )

    return parser


def main(argv: list[str] | None = None) -> None:
    """Run the importer from the command line."""
    parser = build_arg_parser()
    args   = parser.parse_args(argv)

    imp = EEGRawImporter(
        path_to_file=args.file_path,
        helper_dir=args.helper_dir,
        keep_stim=args.keep_stim,
        verbose=args.verbose,
    )

    # Minimal sanity output so the script doesn't feel like it did nothing.
    print("Imported file:", args.file_path)
    print("Sampling rate [Hz]:", imp.stream.get("fs"))
    print("Montage type:", imp.montage.get("montage_type"))
    print("Test date:", imp.stream.get("date_test"))
    print("EEG channels:", len(imp.raw.ch_names))


if __name__ == "__main__":
    main()