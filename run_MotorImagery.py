from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import List

from RawImporter import EEGRawImporter
from BridgingChecker import BridgingChecker
from Preprocessing import EEGPreprocessor, EEGPreprocessorConfig
from MotorImagery import EEGMotorImagery
from PdfReport import MotorImageryPdfReport


def _ensure_dir(path: Path) -> None:
    """Create folder and parents if they do not exist"""
    path.mkdir(parents=True, exist_ok=True)


def _parse_freq_bands(text: str) -> List[float]:
    """Parse a comma-separated list of band edges into floats

    Example: "4,8,13,31" -> [4.0, 8.0, 13.0, 31.0]
    """
    edges = [float(x.strip()) for x in text.split(",") if x.strip()]
    if len(edges) < 2:
        raise argparse.ArgumentTypeError(
            "freq-bands must include at least two comma-separated numbers"
        )
    return edges


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser"""
    parser = argparse.ArgumentParser(
        prog="run_MotorImagery",
        description="Run motor imagery analysis and generate a PDF report",
    )

    # Input resolution
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument(
        "--file-path",
        help="Full path to the input .dat file",
    ) 
    parser.add_argument(
        "--helper-dir",
        default="./helper",
        help="Folder containing helper montage files and report assets",
    )
    parser.add_argument(
        "--save-path",
        required=True,
        help= "Output folder for plots and the PDF",
    )

    # Bridging checker
    parser.add_argument("--skip-bridging",     action="store_true",      help="Skip bridging analysis")
    parser.add_argument("--bridge-fmin",       type=float, default=1.0,  help="Bridging fmin (Hz)")
    parser.add_argument("--bridge-fmax",       type=float, default=40.0, help="Bridging fmax (Hz)")
    parser.add_argument("--bridge-sigma",      type=float, default=0.05, help="Bridging sigma")
    parser.add_argument("--bridge-window-sec", type=float, default=10.0, help="Bridging window length (s)")
    parser.add_argument(
        "--bridge-score-threshold",
        type=float,
        default=0.095,
        help="Threshold on correlation x affinity bridge score",
    )
    parser.add_argument("--bridge-verbose", action="store_true", help="Verbose bridging output")

    # Preprocessing toggles
    parser.add_argument("--skip-preprocessing", action="store_true", help="Skip preprocessing pipeline")
    parser.add_argument("--no-notch",           action="store_true", help="Disable notch filter step")
    parser.add_argument("--no-bandpass",        action="store_true", help="Disable band-pass filter step")
    parser.add_argument("--no-prep",            action="store_true", help="Disable PREP noisy-channel detection step")
    parser.add_argument("--no-annotation",      action="store_true", help="Disable manual BAD-region annotation step")
    parser.add_argument("--no-interpolation",   action="store_true", help="Disable bad-channel interpolation step")
    parser.add_argument("--no-rereference",     action="store_true", help="Disable re-referencing step")
    parser.add_argument("--no-spatialfilter",   action="store_true", help="Disable spatial filtering step")

    # Preprocessing parameters
    parser.add_argument("--notch-freqs",         type=float, default=60.0,  help="Notch frequency (Hz)")
    parser.add_argument("--bandpass-lfreq",      type=float, default=1.0,   help="Band-pass low cutoff (Hz)")
    parser.add_argument("--bandpass-hfreq",      type=float, default=40.0, help="Band-pass high cutoff (Hz)")
    parser.add_argument("--prep-random-state",   type=int, default=83092,   help="Random seed for PREP")
    parser.add_argument("--prep-no-correlation", action="store_true",       help="Disable PREP correlation criterion")
    parser.add_argument("--prep-no-deviation",   action="store_true",       help="Disable PREP deviation criterion")
    parser.add_argument("--prep-no-hf-noise",    action="store_true",       help="Disable PREP HF-noise criterion")
    parser.add_argument("--prep-no-nan-flat",    action="store_true",       help="Disable PREP NaN/flat criterion")
    parser.add_argument("--prep-no-ransac",      action="store_true",       help="Disable PREP RANSAC criterion")
    parser.add_argument(
        "--annotation-plot",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Show interactive plot window for manual annotations",
    )
    parser.add_argument(
        "--reset-bads-after-interp",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reset raw.info['bads'] after interpolation",
    )
    parser.add_argument("--reref-channels", default="tp9 tp10", help="Channels used for re-referencing")
    parser.add_argument(
        "--montage-type",
        default=None,
        help="Optional montage override (e.g. DSI_24, GTEC_32, EGI_64, EGI_128)",
    )
    parser.add_argument("--preproc-verbose", action="store_true", help="Verbose preprocessing output")

    # Motor imagery analysis
    parser.add_argument("--skip-analysis", action="store_true", help="Skip motor imagery analysis")
    parser.add_argument("--n-epochs", type=int, default=6, help="Number of epochs per trial")
    parser.add_argument("--duration-task", type=float, default=10.0, help="Task duration (s)")
    parser.add_argument("--skip", dest="skip_sec", type=float, default=1.0, help="Seconds to skip at start of each segment")
    parser.add_argument("--resolution", type=float, default=1.0, help="PSD frequency resolution (Hz/bin)")
    parser.add_argument(
        "--freq-bands",
        type=_parse_freq_bands,
        default=[4.0, 8.0, 13.0, 31.0],
        help="Comma-separated band edges, e.g. 4,8,13,31",
    )
    parser.add_argument("--n-sim",            type=int, default=2999, help="# simulations for stats tests")
    parser.add_argument("--strict",           action="store_true",    help="Use strict motor imagery channel set")
    parser.add_argument("--analysis-verbose", action="store_true",    help="Verbose motor imagery analysis output")

    # Report
    parser.add_argument("--skip-report", action="store_true", help="Skip PDF report generation")
    parser.add_argument("--age-at-test", default="N/A",        help="Age at test (string, used in the PDF header)")

    return parser

def _build_preproc_config(args: argparse.Namespace) -> EEGPreprocessorConfig:
    """Create an EEGPreprocessorConfig from CLI args"""
    steps = []

    if not args.no_notch:
        steps.append(("notch", {"freqs": args.notch_freqs, "kwargs": {}}))

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
                    "correlation": not args.prep_no_correlation,
                    "deviation": not args.prep_no_deviation,
                    "hf_noise": not args.prep_no_hf_noise,
                    "nan_flat": not args.prep_no_nan_flat,
                    "ransac": not args.prep_no_ransac,
                },
            )
        )

    if not args.no_annotation:
        steps.append(("annotation", {"plot": bool(args.annotation_plot)}))

    if not args.no_interpolation:
        steps.append(
            (
                "interpolation",
                {"reset_bads_after_interp": bool(args.reset_bads_after_interp)},
            )
        )

    if not args.no_rereference:
        steps.append(("rereference", {"channels": args.reref_channels}))

    if not args.no_spatialfilter:
        steps.append(("spatialfilter", {"exclude": None}))

    return EEGPreprocessorConfig(steps)


def main(argv: list[str] | None = None) -> None:
    """Run the full motor imagery pipeline from the command line"""
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    _ensure_dir(Path(args.save_path))

    # ------------------------------------------------------------------
    # Import
    # ------------------------------------------------------------------
    print(['--- Running EEGRawImporter ---'])
    imp = EEGRawImporter(
        path_to_file=args.file_path,
        helper_dir=args.helper_dir,
        keep_stim=True, # don't change
        verbose=True,
    )

    raw = imp.raw
    ch_set = imp.ch_set
    montage_type = args.montage_type or imp.montage.get("montage_type")
    date_test = imp.stream.get("date_test")

    file_name = args.file_path.split('/')[-1].split('.')[0]
    save_path = os.path.join(args.save_path, file_name)
    print(save_path)

    # ------------------------------------------------------------------
    # Bridging checker
    # ------------------------------------------------------------------
    if not args.skip_bridging:
        print(['--- Running BridgingChecker ---'])
        BridgingChecker(
            raw=raw,
            verbose=args.bridge_verbose,
            fmin=args.bridge_fmin,
            fmax=args.bridge_fmax,
            sigma=args.bridge_sigma,
            window_sec=args.bridge_window_sec,
            bridge_score_threshold=args.bridge_score_threshold,
            show_extra=False,
            figure=None,
            axes=None,
            save_path=save_path,
        )

    # ------------------------------------------------------------------
    # Preprocessing
    # ------------------------------------------------------------------
    if not args.skip_preprocessing:
        config = _build_preproc_config(args)
        print(['--- Running EEGPreprocessor ---'])
        preproc = EEGPreprocessor(
            raw,
            ch_set,
            config=config,
            copy=True,
            montage_type=montage_type,
            verbose=args.preproc_verbose,
        )
        raw, _history = preproc.run()
        ch_set = preproc.ch_set

    # ------------------------------------------------------------------
    # Motor imagery analysis
    # ------------------------------------------------------------------
    if not args.skip_analysis:
        print(['--- Running EEGMotorImagery ---'])
        EEGMotorImagery(
            raw,
            ch_set,
            nEpochs=args.n_epochs,
            duration_task=args.duration_task,
            skip=args.skip_sec,
            resolution=args.resolution,
            freq_bands=args.freq_bands,
            nSim=args.n_sim,
            strict=args.strict,
            copy=True,
            verbose=args.analysis_verbose,
            save_path=save_path,
        )

    # ------------------------------------------------------------------
    # PDF report
    # ------------------------------------------------------------------
    if not args.skip_report:
        print(['--- Running MotorImageryPdfReport ---'])
        MotorImageryPdfReport(
            plot_folder=save_path,
            helper_folder=args.helper_dir,
            date_test=date_test or "N/A",
            montage_name=montage_type or "N/A",
            resolution=int(args.resolution),
            age_at_test=str(args.age_at_test),
            save_folder=save_path,
        )

    print("Pipeline complete")
    print("Input:", args.file_path)
    print("Output folder:", save_path)


if __name__ == "__main__":
    main()