from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import mne

from Preprocessing import EEGPreprocessor, EEGPreprocessorConfig
from RawImporter import EEGRawImporter


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(
        prog="run_Preprocessing",
        description="Import a .dat file and run the EEG preprocessing pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--file-path", required=True, help="Path to the input .dat file")
    parser.add_argument("--helper-dir", default="./helper", help="Folder containing helper montage files")
    parser.add_argument(
        "--keep-stim",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Keep stimulus/state channels when importing",
    )
    parser.add_argument("--import-verbose", action="store_true", help="Verbose output during import")

    parser.add_argument("--no-notch", action="store_true", help="Disable notch filter step")
    parser.add_argument("--no-bandpass", action="store_true", help="Disable band-pass filter step")
    parser.add_argument("--no-prep", action="store_true", help="Disable PREP noisy-channel detection step")
    parser.add_argument("--no-annotation", action="store_true", help="Disable manual BAD-region annotation step")
    parser.add_argument("--no-interpolation", action="store_true", help="Disable bad-channel interpolation step")
    parser.add_argument("--no-rereference", action="store_true", help="Disable re-referencing step")
    parser.add_argument("--no-spatialfilter", action="store_true", help="Disable spatial filtering step")

    parser.add_argument(
        "--notch-freqs",
        type=float,
        nargs="+",
        default=[60.0],
        help="Base notch frequency or explicit list of notch frequencies in Hz",
    )
    parser.add_argument("--bandpass-lfreq", type=float, default=1.0, help="Band-pass low cutoff in Hz")
    parser.add_argument("--bandpass-hfreq", type=float, default=40.0, help="Band-pass high cutoff in Hz")
    parser.add_argument("--prep-random-state", type=int, default=83092, help="Random seed for PREP RANSAC")
    parser.add_argument("--prep-no-correlation", action="store_true", help="Disable PREP correlation criterion")
    parser.add_argument("--prep-no-deviation", action="store_true", help="Disable PREP deviation criterion")
    parser.add_argument("--prep-no-hf-noise", action="store_true", help="Disable PREP high-frequency-noise criterion")
    parser.add_argument(
        "--prep-hf-noise-action",
        choices=["review", "mark"],
        default="review",
        help="Treat PREP high-frequency-noise detections as review-only channels or mark them as bad",
    )
    parser.add_argument("--prep-no-nan-flat", action="store_true", help="Disable PREP NaN/flat criterion")
    parser.add_argument("--prep-no-ransac", action="store_true", help="Disable PREP RANSAC criterion")
    parser.add_argument(
        "--annotation-plot",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Show interactive plot window for manual annotations",
    )
    parser.add_argument(
        "--annotation-label",
        default="BAD_region",
        help="Annotation label used to summarize manually marked bad segments",
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
        "--spatial-exclude",
        nargs="+",
        default=None,
        help="Optional channel names to exclude from the spatial filter",
    )
    parser.add_argument(
        "--montage-type",
        default=None,
        help="Optional montage override, e.g. DSI_24, GTEC_32, EGI_64, EGI_128",
    )

    parser.add_argument("--verbose", action="store_true", help="Verbose MNE/preprocessing output with tracebacks")
    parser.add_argument("--quiet", action="store_true", help="Suppress progress messages from this wrapper")
    parser.add_argument(
        "--save-cleaned-raw",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save the cleaned Raw object after preprocessing",
    )
    parser.add_argument(
        "--save-path",
        default="./preprocessed",
        help="Folder used for the default cleaned Raw cache",
    )
    parser.add_argument(
        "--cleaned-raw-output",
        default=None,
        help="Optional explicit output path for the cleaned Raw cache",
    )
    parser.add_argument(
        "--overwrite-cleaned-raw",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Overwrite an existing cleaned Raw cache",
    )

    return parser


def _single_or_list(values: list[float]) -> float | list[float]:
    if len(values) == 1:
        return values[0]
    return values


def _build_config(args: argparse.Namespace) -> EEGPreprocessorConfig:
    """Create an EEGPreprocessorConfig from command-line arguments."""
    steps: list[tuple[str, dict[str, Any]]] = []

    if not args.no_notch:
        steps.append(
            (
                "notch",
                {
                    "freqs": _single_or_list(args.notch_freqs),
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
                    "correlation": not args.prep_no_correlation,
                    "deviation": not args.prep_no_deviation,
                    "hf_noise": not args.prep_no_hf_noise,
                    "hf_noise_action": args.prep_hf_noise_action,
                    "nan_flat": not args.prep_no_nan_flat,
                    "ransac": not args.prep_no_ransac,
                },
            )
        )

    if not args.no_annotation:
        steps.append(
            (
                "annotation",
                {
                    "plot": bool(args.annotation_plot),
                    "label": args.annotation_label,
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
                    "exclude": args.spatial_exclude,
                },
            )
        )

    return EEGPreprocessorConfig(steps)


def _echo(message: str, quiet: bool = False) -> None:
    if not quiet:
        print(message, flush=True)


def _format_list(values: list[str] | tuple[str, ...] | None) -> str:
    if not values:
        return "none"
    return ", ".join(str(value) for value in values)


def _print_header(title: str, quiet: bool = False) -> None:
    _echo("", quiet=quiet)
    _echo("=" * 72, quiet=quiet)
    _echo(title, quiet=quiet)
    _echo("=" * 72, quiet=quiet)


def _print_raw_summary(raw: mne.io.BaseRaw, title: str, quiet: bool = False) -> None:
    eeg_picks = mne.pick_types(raw.info, eeg=True, exclude=())
    duration = raw.n_times / float(raw.info["sfreq"])
    _echo(f"{title}", quiet=quiet)
    _echo(f"  Channels        : {raw.info['nchan']} total, {len(eeg_picks)} EEG", quiet=quiet)
    _echo(f"  Samples         : {raw.n_times}", quiet=quiet)
    _echo(f"  Sampling rate   : {raw.info['sfreq']:.3f} Hz", quiet=quiet)
    _echo(f"  Duration        : {duration:.2f} s", quiet=quiet)
    _echo(f"  Highpass/Lowpass: {raw.info.get('highpass')} / {raw.info.get('lowpass')}", quiet=quiet)
    _echo(f"  Bads            : {_format_list(raw.info.get('bads', []))}", quiet=quiet)


def _safe_montage_type(importer: EEGRawImporter, override: str | None) -> str | None:
    if override:
        return override
    montage = getattr(importer, "montage", None) or {}
    if isinstance(montage, dict):
        return montage.get("montage_type")
    return None


def _validate_paths(args: argparse.Namespace) -> tuple[Path, Path]:
    file_path = Path(args.file_path).expanduser()
    helper_dir = Path(args.helper_dir).expanduser()

    if not file_path.exists():
        raise FileNotFoundError(f"Input file not found: {file_path}")
    if not helper_dir.exists():
        _echo(f"[run_Preprocessing] Warning: helper directory not found: {helper_dir}", quiet=args.quiet)

    return file_path, helper_dir


def _metadata_path_for(raw_path: Path) -> Path:
    """Return the JSON sidecar path for a cached cleaned Raw file"""
    if raw_path.name.lower().endswith(".fif.gz"):
        return raw_path.with_name(raw_path.name[:-7] + ".json")
    return raw_path.with_suffix(".json")


def _default_cleaned_raw_output(file_path: Path, save_path: Path) -> Path:
    """Return the default cleaned Raw cache path"""
    return save_path / f"{file_path.stem}_cleaned_raw.fif"


def _save_cleaned_raw(
    raw: mne.io.BaseRaw,
    output_path: Path,
    metadata: dict[str, Any],
    overwrite: bool,
) -> Path:
    """Save cleaned Raw and a small JSON sidecar"""
    output_path = output_path.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    raw.save(str(output_path), overwrite=overwrite, verbose=False)
    _metadata_path_for(output_path).write_text(json.dumps(metadata, indent=2, default=str))
    return output_path


def run_pipeline(args: argparse.Namespace) -> tuple[mne.io.BaseRaw, dict[str, Any]]:
    file_path, helper_dir = _validate_paths(args)
    _print_header("EEG preprocessing", quiet=args.quiet)
    _echo(f"Input file : {file_path}", quiet=args.quiet)
    _echo(f"Helper dir : {helper_dir}", quiet=args.quiet)
    _echo(f"Keep stim  : {args.keep_stim}", quiet=args.quiet)

    _echo("\n[run_Preprocessing] Importing raw data", quiet=args.quiet)
    importer = EEGRawImporter(
        path_to_file=str(file_path),
        helper_dir=str(helper_dir),
        keep_stim=args.keep_stim,
        verbose=args.import_verbose,
    )

    raw = importer.raw
    ch_set = importer.ch_set
    montage_type = _safe_montage_type(importer, args.montage_type)
    config = _build_config(args)
    _print_raw_summary(raw, "\nImported raw summary", quiet=args.quiet)
    _echo(f"  Montage type    : {montage_type or 'unknown'}", quiet=args.quiet)

    preprocessor = EEGPreprocessor(
        raw,
        ch_set,
        config=config,
        copy=True,
        montage_type=montage_type,
        verbose=args.verbose,
        feedback=not args.quiet,
    )
    raw_out, history = preprocessor.run()

    _print_header("Preprocessing complete", quiet=args.quiet)
    _echo(f"Input file       : {file_path}", quiet=args.quiet)
    _echo(f"Montage type     : {montage_type or 'unknown'}", quiet=args.quiet)
    _echo(f"Initial bads     : {_format_list(history.get('initial_bads', []))}", quiet=args.quiet)
    _echo(f"Final bads       : {_format_list(raw_out.info.get('bads', []))}", quiet=args.quiet)
    _echo(
        f"Highpass/Lowpass : {raw_out.info.get('highpass')} / {raw_out.info.get('lowpass')}",
        quiet=args.quiet,
    )

    if history.get("step_log"):
        _echo("\nStep status", quiet=args.quiet)
        for item in history["step_log"]:
            status = item.get("status", "unknown")
            step = item.get("step", "")
            label = item.get("label", item.get("step", "step"))
            elapsed = float(item.get("elapsed_sec", 0.0))
            if step == "annotation" and item.get("elapsed_note"):
                _echo(f"  {status.upper():7s} {label:<35s} manual inspection time omitted", quiet=args.quiet)
            else:
                _echo(f"  {status.upper():7s} {label:<35s} {elapsed:8.2f}s", quiet=args.quiet)

    _print_raw_summary(raw_out, "\nOutput raw summary", quiet=args.quiet)

    if args.save_cleaned_raw:
        save_path = Path(args.save_path).expanduser().resolve()
        cleaned_output = (
            Path(args.cleaned_raw_output).expanduser().resolve()
            if args.cleaned_raw_output
            else _default_cleaned_raw_output(file_path, save_path)
        )
        metadata = {
            "source_file": str(file_path),
            "montage_type": montage_type,
            "date_test": getattr(importer, "stream", {}).get("date_test", "N/A"),
            "n_channels": int(raw_out.info["nchan"]),
            "sfreq": float(raw_out.info["sfreq"]),
            "preprocessing_history": history,
        }
        saved_cleaned = _save_cleaned_raw(
            raw_out,
            cleaned_output,
            metadata=metadata,
            overwrite=bool(args.overwrite_cleaned_raw),
        )
        history["cleaned_raw_path"] = str(saved_cleaned)
        _echo(f"\nSaved cleaned Raw : {saved_cleaned}", quiet=args.quiet)
        _echo(f"Saved sidecar JSON: {_metadata_path_for(saved_cleaned)}", quiet=args.quiet)

    return raw_out, history


def main(argv: list[str] | None = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    try:
        run_pipeline(args)
    except KeyboardInterrupt:
        parser.exit(130, "\n[run_Preprocessing] Interrupted by user\n")
    except Exception as exc:
        if args.verbose:
            raise
        parser.exit(
            1,
            f"\n[run_Preprocessing] ERROR: {type(exc).__name__}: {exc}\n"
            "Run again with --verbose for a full traceback\n",
        )


if __name__ == "__main__":
    main()
