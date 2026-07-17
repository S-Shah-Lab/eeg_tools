from __future__ import annotations

import argparse
from pathlib import Path


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser"""
    parser = argparse.ArgumentParser(
        prog="run_RawImporter",
        description="Import a BCI2000 .dat file and construct an MNE Raw object with montage",
    )

    parser.add_argument("--file-path", required=True, help="Path to the input .dat file")
    parser.add_argument("--helper-dir", default="./helper", help="Folder containing helper montage files")
    parser.add_argument("--keep-stim", action="store_true", help="Keep stimulus/state channels")
    parser.add_argument("--verbose", action="store_true", help="Print detailed import information")

    parser.add_argument("--validation-plots", action="store_true", help="Save validation plots after import")
    parser.add_argument("--plot-dir", default=None, help="Folder where validation plots are saved")
    parser.add_argument("--plot-format", choices=["png", "pdf", "svg"], default="png", help="Validation plot format")
    parser.add_argument("--plot-start", type=float, default=0.0, help="Trace overview start time in seconds")
    parser.add_argument("--plot-duration", type=float, default=10.0, help="Trace overview duration in seconds")
    parser.add_argument("--max-plot-channels", type=int, default=32, help="Maximum number of channels in trace plots")
    parser.add_argument("--psd-fmax", type=float, default=60.0, help="Maximum PSD frequency in Hz")

    return parser


def _default_plot_dir(file_path: str) -> Path:
    """Build the default validation-plot output folder"""
    return Path("validation_plots") / Path(file_path).expanduser().stem


def _print_header(title: str) -> None:
    print(f"\n=== {title} ===")


def main(argv: list[str] | None = None) -> None:
    """Run the importer from the command line"""
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    try:
        from RawImporter import EEGRawImporter
    except Exception as exc:
        parser.exit(status=1, message=f"[run_RawImporter] ERROR: could not import dependencies: {exc}\n")

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

        saved_plots = []
        if args.validation_plots:
            plot_dir = Path(args.plot_dir).expanduser() if args.plot_dir else _default_plot_dir(args.file_path)
            saved_plots = importer.save_validation_plots(
                output_dir=plot_dir,
                plot_format=args.plot_format,
                start_sec=args.plot_start,
                duration_sec=args.plot_duration,
                max_channels=args.max_plot_channels,
                psd_fmax=args.psd_fmax,
            )

    except Exception as exc:
        parser.exit(status=1, message=f"[run_RawImporter] ERROR: {exc}\n")

    _print_header("Raw import summary")
    print(f"Input file          : {args.file_path}")
    print(f"Recording date      : {importer.stream.get('date_test')}")
    print(f"Sampling rate       : {importer.stream.get('fs')} Hz")
    print(f"Original shape      : {importer.stream.get('signal').shape}")
    print(f"Duration            : {importer.stream.get('duration_sec'):.2f} s")
    print(f"Montage type        : {importer.montage.get('montage_type')}")
    aux_source_channels = importer.montage.get("aux_source_channels", [])
    if aux_source_channels:
        print(f"Dropped source rows : {', '.join(aux_source_channels)}")
    print(f"Raw channels        : {len(raw.ch_names)}")
    print(f"Raw samples         : {raw.n_times}")
    print(f"Stim channels       : {'kept' if args.keep_stim else 'skipped'}")

    if args.validation_plots:
        _print_header("Validation plots")
        if saved_plots:
            for path in saved_plots:
                print(f"Saved               : {path}")
        else:
            print("Saved               : none")


if __name__ == "__main__":
    main()
