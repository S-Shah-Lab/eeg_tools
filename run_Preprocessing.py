from __future__ import annotations
import argparse
from RawImporter import EEGRawImporter
from Preprocessing import EEGPreprocessor, EEGPreprocessorConfig


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser"""
    parser = argparse.ArgumentParser(
        prog="run_Preprocessing",
        description="Import a .dat file and run the EEG preprocessing pipeline",
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

    # Step toggles (enabled by default)
    parser.add_argument("--no-notch",         action="store_true", help="Disable notch filter step"                )
    parser.add_argument("--no-bandpass",      action="store_true", help="Disable band-pass filter step"            )
    parser.add_argument("--no-prep",          action="store_true", help="Disable PREP noisy-channel detection step")
    parser.add_argument("--no-annotation",    action="store_true", help="Disable manual BAD-region annotation step")
    parser.add_argument("--no-interpolation", action="store_true", help="Disable bad-channel interpolation step"   )
    parser.add_argument("--no-rereference",   action="store_true", help="Disable re-referencing step"              )
    parser.add_argument("--no-spatialfilter", action="store_true", help="Disable spatial filtering step"           )

    # Parameters
    parser.add_argument(
        "--notch-freqs",
        type=float,
        default=60.0,
        help="Notch frequency (Hz). Default: 60",
    )
    parser.add_argument(
        "--bandpass-lfreq",
        type=float,
        default=1.0,
        help="Band-pass low cutoff (Hz). Default: 1",
    )
    parser.add_argument(
        "--bandpass-hfreq",
        type=float,
        default=40.0,
        help="Band-pass high cutoff (Hz). Default: 40",
    )
    parser.add_argument(
        "--prep-random-state",
        type=int,
        default=83092,
        help="Random seed for PREP (RANSAC). Default: 83092",
    )
    parser.add_argument(
        "--prep-no-correlation",
        action="store_true",
        help="Disable PREP correlation criterion",
    )
    parser.add_argument(
        "--prep-no-deviation",
        action="store_true",
        help="Disable PREP deviation criterion",
    )
    parser.add_argument(
        "--prep-no-hf-noise",
        action="store_true",
        help="Disable PREP HF-noise criterion",
    )
    parser.add_argument(
        "--prep-no-nan-flat",
        action="store_true",
        help="Disable PREP NaN/flat criterion",
    )
    parser.add_argument(
        "--prep-no-ransac",
        action="store_true",
        help="Disable PREP RANSAC criterion",
    )
    parser.add_argument(
        "--annotation-plot",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Show interactive plot window for manual annotations",
    )
    parser.add_argument(
        "--reref-channels",
        default="tp9 tp10",
        help="Space-separated channel names to use for re-referencing",
    )
    parser.add_argument(
        "--reset-bads-after-interp",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reset raw.info['bads'] after interpolation",
    )
    parser.add_argument(
        "--montage-type",
        default=None,
        help="Optional montage override (e.g. DSI_24, GTEC_32, EGI_64, EGI_128)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose output during preprocessing",
    )

    return parser


def _build_config(args: argparse.Namespace) -> EEGPreprocessorConfig:
    """Create an EEGPreprocessorConfig from CLI args"""
    steps = []

    if not args.no_notch:
        steps.append(
            (
                "notch",
                {
                    "freqs": args.notch_freqs,
                    "kwargs": {},
                },
            )
        )

    if not args.no_bandpass:
        steps.append(
            (
                "bandpass",
                {
                    "l_freq": args.bandpass_lfreq,
                    "h_freq": args.bandpass_hfreq,
                    "kwargs": {},
                },
            )
        )

    if not args.no_prep:
        steps.append(
            (
                "prep",
                {
                    "random_state": args.prep_random_state,
                    "correlation":  not args.prep_no_correlation,
                    "deviation":    not args.prep_no_deviation,
                    "hf_noise":     not args.prep_no_hf_noise,
                    "nan_flat":     not args.prep_no_nan_flat,
                    "ransac":       not args.prep_no_ransac,
                },
            )
        )

    if not args.no_annotation:
        steps.append(
            (
                "annotation",
                {
                    "plot": bool(args.annotation_plot),
                },
            )
        )

    if not args.no_interpolation:
        steps.append(
            (
                "interpolation",
                {
                    "reset_bads_after_interp": bool(args.reset_bads_after_interp),
                },
            )
        )

    if not args.no_rereference:
        steps.append(
            (
                "rereference",
                {
                    "channels": args.reref_channels,
                },
            )
        )

    if not args.no_spatialfilter:
        steps.append(
            (
                "spatialfilter",
                {
                    "exclude": None,
                },
            )
        )

    return EEGPreprocessorConfig(steps)


def main(argv: list[str] | None = None) -> None:
    """Run preprocessing from the command line"""
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    imp = EEGRawImporter(
        path_to_file=args.file_path,
        helper_dir=args.helper_dir,
        keep_stim=args.keep_stim,
        verbose=args.import_verbose,
    )

    raw = imp.raw
    ch_set = imp.ch_set
    montage_type = args.montage_type or imp.montage.get("montage_type")

    config = _build_config(args)

    preproc = EEGPreprocessor(
        raw,
        ch_set,
        config=config,
        copy=True,
        montage_type=montage_type,
        verbose=args.verbose,
    )
    raw_out, history = preproc.run()

    print("Preprocessing complete")
    print("Input file:", args.file_path)
    print("Montage type:", montage_type)
    print("Initial bads:", history.get("initial_bads"))
    print("Final bads:", raw_out.info.get("bads"))
    print("Highpass/Lowpass:", raw_out.info.get("highpass"), "/", raw_out.info.get("lowpass"))


if __name__ == "__main__":
    main()