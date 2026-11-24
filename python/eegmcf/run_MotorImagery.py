from __future__ import annotations

import argparse
import json
import math

import shutil
from pathlib import Path
from typing import Any, List

import mne
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from BCI2000Tools.Electrodes import ChannelSet

if __package__:
    from .RawImporter import EEGRawImporter
    from .BridgingChecker import BridgingChecker
    from .Preprocessing import EEGPreprocessor, EEGPreprocessorConfig
    from .MotorImagery import EEGMotorImagery
    from .PdfReport import MotorImageryPdfReport
else:
    from RawImporter import EEGRawImporter
    from BridgingChecker import BridgingChecker
    from Preprocessing import EEGPreprocessor, EEGPreprocessorConfig
    from MotorImagery import EEGMotorImagery
    from PdfReport import MotorImageryPdfReport


if __package__:
    from .RecordingMetadata import add_metadata_arguments, apply_overrides, metadata_from_raw, recording_stem, spatial_filter_label
else:
    from RecordingMetadata import add_metadata_arguments, apply_overrides, metadata_from_raw, recording_stem, spatial_filter_label


if __package__:
    from .CleanedChannels import CleanedEEGChannelSet, infer_cleaned_montage_type
else:
    from CleanedChannels import CleanedEEGChannelSet, infer_cleaned_montage_type


def _ensure_dir(path: Path) -> None:
    """Create folder and parents if they do not exist"""
    path.mkdir(parents=True, exist_ok=True)


def _parse_freq_bands(text: str) -> List[float]:
    """Parse a comma-separated list of band edges into floats"""
    try:
        edges = [float(x.strip()) for x in text.split(",") if x.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "freq-bands must be a comma-separated list of numbers, e.g. 4,8,13,31"
        ) from exc

    if len(edges) < 2:
        raise argparse.ArgumentTypeError(
            "freq-bands must include at least two comma-separated numbers"
        )
    if any(not math.isfinite(x) or x < 0 for x in edges) or any(b <= a for a, b in zip(edges[:-1], edges[1:])):
        raise argparse.ArgumentTypeError("freq-bands must be strictly increasing")

    return edges


def _stage(index: int, total: int, title: str) -> None:
    """Print a compact pipeline stage header"""
    print(f"\n[{index}/{total}] {title}")


def _detail(label: str, value: object) -> None:
    """Print a compact indented key/value message"""
    print(f"      {label}: {value}")


def _parse_optional_channel_list(text: str | None) -> list[str] | bool:
    """Parse a comma-separated channel list, or false/none to disable fit plots"""
    if text is None:
        return ["c3", "c4"]
    normalized = str(text).strip()
    if normalized.lower() in {"false", "off", "none", "no"}:
        return False
    channels = [item.strip().lower() for item in normalized.split(",") if item.strip()]
    return channels or False


def _copy_optional_report_assets(source_images: Path, target_images: Path) -> None:
    """Copy QC assets that are shared by the raw-power and 1/f-subtracted reports"""
    for filename in ("bridged_candidates.png", "bridged_candidates.svg", "bridged_candidate_groups.csv"):
        source = source_images / filename
        if source.is_file():
            target_images.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target_images / filename)


def _format_step_names(config: EEGPreprocessorConfig) -> str:
    """Return preprocessing step names for terminal feedback"""
    try:
        steps = [name for name, _ in config.iter_steps()]
    except Exception:
        steps = list(getattr(config, "keys", lambda: [])())
    if not steps:
        return "none"
    return " → ".join(str(step) for step in steps)


def _is_cleaned_raw_file(path: Path | None) -> bool:
    """Return True for supported MNE Raw FIF input names"""
    if path is None:
        return False
    name = path.name.lower()
    return name.endswith(".fif") or name.endswith(".fif.gz")


def _metadata_path_for(raw_path: Path) -> Path:
    """Return the JSON sidecar path for a cached cleaned Raw file"""
    if raw_path.name.lower().endswith(".fif.gz"):
        return raw_path.with_name(raw_path.name[:-7] + ".json")
    return raw_path.with_suffix(".json")


def _load_cleaned_metadata(raw_path: Path) -> dict[str, Any]:
    """Load optional metadata saved next to a cleaned Raw file"""
    meta_path = _metadata_path_for(raw_path)
    if not meta_path.is_file():
        return {}
    try:
        data = json.loads(meta_path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError("Sidecar must contain a JSON object")
        return data
    except Exception as exc:
        print(f"[run_MotorImagery warning] Could not read cleaned Raw metadata {meta_path}: {exc}")
        return {}


def _default_cleaned_raw_output(save_path: Path, input_path: Path) -> Path:
    """Return the default cleaned Raw cache path"""
    return save_path / f"{input_path.stem}_cleaned_raw.fif"


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
    meta_path = _metadata_path_for(output_path)
    meta_path.write_text(json.dumps(metadata, indent=2, default=str))
    return output_path


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser"""
    parser = argparse.ArgumentParser(
        prog="run_MotorImagery",
        description="Run motor imagery analysis and generate a PDF report",
    )

    parser.add_argument(
        "--file-path",
        required=False,
        help="Full path to the input .dat file, or a cleaned Raw .fif file",
    )
    parser.add_argument(
        "--cleaned-raw-input",
        default=None,
        help="Existing cleaned Raw .fif file to analyze directly without import, bridging, or preprocessing",
    )
    parser.add_argument(
        "--helper-dir",
        default=str(Path(__file__).resolve().parent / "helper"),
        help="Folder containing helper montage files and report assets",
    )
    parser.add_argument(
        "--save-path",
        required=True,
        help="Output folder for plots and the PDF",
    )

    parser.add_argument("--skip-bridging", action="store_true", help="Skip bridging analysis")
    parser.add_argument("--bridge-fmin", type=float, default=1.0, help="Bridging fmin (Hz)")
    parser.add_argument("--bridge-fmax", type=float, default=40.0, help="Bridging fmax (Hz)")
    parser.add_argument("--bridge-sigma", type=float, default=0.05, help="Bridging sigma")
    parser.add_argument("--bridge-window-sec", type=float, default=10.0, help="Bridging window length (s)")
    parser.add_argument(
        "--bridge-score-threshold",
        type=float,
        default=0.095,
        help="Threshold on correlation x affinity bridge score",
    )
    parser.add_argument("--bridge-verbose", action="store_true", help="Verbose bridging output")

    parser.add_argument("--skip-preprocessing", action="store_true", help="Skip preprocessing pipeline")
    parser.add_argument("--no-notch", action="store_true", help="Disable notch filter step")
    parser.add_argument("--no-bandpass", action="store_true", help="Disable band-pass filter step")
    parser.add_argument("--no-prep", action="store_true", help="Disable PREP noisy-channel detection step")
    parser.add_argument("--no-annotation", action="store_true", help="Disable manual BAD-region annotation step")
    parser.add_argument("--no-interpolation", action="store_true", help="Disable bad-channel interpolation step")
    parser.add_argument("--no-rereference", action="store_true", help="Disable re-referencing step")
    parser.add_argument("--no-spatialfilter", action="store_true", help="Disable spatial filtering step")

    parser.add_argument("--notch-freqs", type=float, default=60.0, help="Notch frequency (Hz)")
    parser.add_argument("--bandpass-lfreq", type=float, default=1.0, help="Band-pass low cutoff (Hz)")
    parser.add_argument("--bandpass-hfreq", type=float, default=40.0, help="Band-pass high cutoff (Hz)")
    parser.add_argument("--prep-random-state", type=int, default=83092, help="Random seed for PREP")
    parser.add_argument("--prep-no-correlation", action="store_true", help="Disable PREP correlation criterion")
    parser.add_argument("--prep-no-deviation", action="store_true", help="Disable PREP deviation criterion")
    parser.add_argument("--prep-no-hf-noise", action="store_true", help="Disable PREP HF-noise criterion")
    parser.add_argument("--prep-no-nan-flat", action="store_true", help="Disable PREP NaN/flat criterion")
    parser.add_argument("--prep-no-ransac", action="store_true", help="Disable PREP RANSAC criterion")
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
    parser.add_argument("--reref-channels", default="auto", help="Reference channels, or auto (TP9/TP10 when present; otherwise retain acquisition reference)")
    parser.add_argument(
        "--montage-type",
        default=None,
        help="Optional montage override (e.g. DSI_24, GTEC_32, EGI_64, EGI_128)",
    )
    parser.add_argument("--preproc-verbose", action="store_true", help="Verbose preprocessing output")
    parser.add_argument(
        "--save-cleaned-raw",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save the cleaned Raw object after preprocessing",
    )
    parser.add_argument(
        "--cleaned-raw-output",
        default=None,
        help="Optional output path for the cleaned Raw cache",
    )
    parser.add_argument(
        "--overwrite-cleaned-raw",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Overwrite an existing cleaned Raw cache",
    )

    parser.add_argument("--skip-analysis", action="store_true", help="Skip motor imagery analysis")
    parser.add_argument("--n-epochs", type=int, default=6, help="Number of epochs per trial")
    parser.add_argument("--duration-task", type=float, default=10.0, help="Task duration (s)")
    parser.add_argument("--skip", dest="skip_sec", type=float, default=1.0, help="Seconds to skip once at the beginning of each task window")
    parser.add_argument("--resolution", type=float, default=1.0, help="PSD frequency resolution (Hz/bin)")
    parser.add_argument(
        "--freq-bands",
        type=_parse_freq_bands,
        default=[4.0, 8.0, 13.0, 31.0],
        help="Comma-separated band edges, e.g. 4,8,13,31",
    )
    parser.add_argument("--n-sim", type=int, default=9999, help="# simulations for stats tests")
    parser.add_argument("--strict", action="store_true", help="Use strict motor imagery channel set")
    parser.add_argument("--analysis-verbose", action="store_true", help="Verbose motor imagery analysis output")
    parser.add_argument("--analysis-random-state", type=int, default=83092, help="Random seed for motor imagery statistics and decoding")
    parser.add_argument("--skip-decoding", action="store_true", help="Skip optional task-vs-rest decoding analysis")
    parser.add_argument(
        "--skip-one-over-f-report",
        action="store_true",
        help="Skip the second report based on selected-model 1/f background subtraction",
    )
    parser.add_argument(
        "--one-over-f-plot-channels",
        default="c3,c4",
        help="Comma-separated channels for best-fit diagnostic plots, or 'false' to disable",
    )

    parser.add_argument("--skip-report", action="store_true", help="Skip PDF report generation")
    parser.add_argument("--skip-csv", action="store_true", help="Skip CSV export (PSD per bin + signed r² per band)")
    add_metadata_arguments(parser)

    return parser


def _validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    """Validate CLI arguments before the EEG pipeline starts"""
    file_path = Path(args.file_path).expanduser() if args.file_path else None
    cleaned_input = Path(args.cleaned_raw_input).expanduser() if args.cleaned_raw_input else None
    helper_dir = Path(args.helper_dir).expanduser()

    if file_path is not None and cleaned_input is not None:
        parser.error("Provide only one of --file-path and --cleaned-raw-input")
    if file_path is None and cleaned_input is None:
        parser.error("Provide --file-path for a .dat/.fif input or --cleaned-raw-input for a cached Raw object")
    if file_path is not None and not file_path.is_file():
        parser.error(f"Input file does not exist: {file_path}")
    if cleaned_input is not None and not cleaned_input.is_file():
        parser.error(f"Cleaned Raw input does not exist: {cleaned_input}")
    if cleaned_input is not None and not _is_cleaned_raw_file(cleaned_input):
        parser.error("--cleaned-raw-input requires .fif or .fif.gz")
    if file_path is not None and not (_is_cleaned_raw_file(file_path) or file_path.suffix.lower() == ".dat"):
        parser.error("Input must be .dat, .fif or .fif.gz")
    if args.skip_analysis and not args.skip_report:
        parser.error("--skip-analysis requires --skip-report; regenerate existing plots with PdfReport directly")
    if not helper_dir.is_dir():
        parser.error(f"Helper directory does not exist: {helper_dir}")
    if args.bridge_fmin < 0 or args.bridge_fmax <= args.bridge_fmin:
        parser.error("--bridge-fmax must be greater than --bridge-fmin")
    if args.bridge_window_sec <= 0:
        parser.error("--bridge-window-sec must be positive")
    if args.bandpass_lfreq < 0 or args.bandpass_hfreq <= args.bandpass_lfreq:
        parser.error("--bandpass-hfreq must be greater than --bandpass-lfreq")
    if args.n_epochs <= 0:
        parser.error("--n-epochs must be positive")
    if args.duration_task <= 0:
        parser.error("--duration-task must be positive")
    if args.skip_sec < 0:
        parser.error("--skip must be non-negative")
    if any(not math.isfinite(x) for x in (args.duration_task, args.skip_sec, args.resolution, args.bridge_fmin, args.bridge_fmax, args.bridge_window_sec, args.bandpass_lfreq, args.bandpass_hfreq)):
        parser.error("Numeric analysis and filter settings must be finite")
    if args.duration_task <= args.skip_sec:
        parser.error("--duration-task must be greater than --skip")
    if args.resolution <= 0:
        parser.error("--resolution must be positive")
    if args.n_sim <= 0:
        parser.error("--n-sim must be positive")
    if not args.skip_analysis and (args.duration_task - args.skip_sec) / args.n_epochs < 1 / args.resolution:
        parser.error("Epoch duration must accommodate a Welch window of 1 / resolution seconds")
    try:
        apply_overrides({}, args)
    except ValueError as exc:
        parser.error(str(exc))


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
    _validate_args(parser, args)

    cleaned_arg = Path(args.cleaned_raw_input).expanduser().resolve() if args.cleaned_raw_input else None
    file_arg = Path(args.file_path).expanduser().resolve() if args.file_path else None
    use_cleaned_input = cleaned_arg is not None or _is_cleaned_raw_file(file_arg)
    input_path = cleaned_arg if cleaned_arg is not None else file_arg
    assert input_path is not None

    helper_dir = Path(args.helper_dir).expanduser().resolve()
    output_root = Path(args.save_path).expanduser().resolve()
    output_name = recording_stem(input_path)
    # Never combine stale figures from a prior analysis with a new report
    save_path = output_root / output_name
    suffix = 2
    while save_path.exists() and any(save_path.iterdir()):
        save_path = output_root / f"{output_name}_analysis-{suffix:02d}"
        suffix += 1
    images_path = save_path / "images"
    csv_path = save_path / "csv"
    one_over_f_save_path = save_path / "one_over_f_subtracted"
    one_over_f_images_path = one_over_f_save_path / "images"
    one_over_f_csv_path = one_over_f_save_path / "csv"
    run_one_over_f = (not args.skip_analysis) and (not args.skip_one_over_f_report)

    _ensure_dir(save_path)
    _ensure_dir(images_path)
    if not args.skip_csv:
        _ensure_dir(csv_path)
    if run_one_over_f:
        _ensure_dir(one_over_f_save_path)
        _ensure_dir(one_over_f_images_path)
        _ensure_dir(one_over_f_csv_path)

    stages = []
    if use_cleaned_input:
        stages.append("Loading cleaned raw EEG")
    else:
        stages.append("Importing raw EEG")
        if not args.skip_bridging:
            stages.append("Running bridging check")
        if not args.skip_preprocessing:
            stages.append("Running preprocessing")
    if not args.skip_analysis:
        stages.append("Running motor imagery analysis")
        if run_one_over_f:
            stages.append("Running 1/f-subtracted motor imagery analysis")
    if not args.skip_report:
        stages.append("Building PDF report(s)" if run_one_over_f else "Building PDF report")

    total = len(stages)
    stage_i = 1

    print("Motor Imagery Pipeline")
    _detail("Input", input_path)
    _detail("Output", save_path)
    _detail("Images", images_path)
    if not args.skip_csv:
        _detail("CSV", csv_path)
    if run_one_over_f:
        _detail("1/f output", one_over_f_save_path)
    _detail("Helper", helper_dir)
    _detail("Simulations", args.n_sim)

    metadata: dict[str, Any] = {}
    history: dict[str, Any] | None = None
    montage_type: str | None = None
    ch_set: ChannelSet | CleanedEEGChannelSet

    if use_cleaned_input:
        _stage(stage_i, total, "Loading cleaned raw EEG")
        stage_i += 1
        metadata = _load_cleaned_metadata(input_path)
        raw = mne.io.read_raw_fif(str(input_path), preload=True, verbose=False)
        montage_type = infer_cleaned_montage_type(raw, helper_dir, args.montage_type, metadata)
        ch_set = CleanedEEGChannelSet(raw)
        _detail("EEG channels", f"{len(ch_set.get_labels())} (saved FIF order)")
        _detail("EEG labels", ", ".join(ch_set.get_labels()))
        if montage_type is None:
            print("[run_MotorImagery warning] Acquisition montage type unknown; using saved FIF electrode positions. "
                  "Supply --montage-type if known for the report header")
        metadata = apply_overrides(metadata_from_raw(raw, input_path, metadata), args)
        history = metadata.get("preprocessing_history")
        _detail("Mode", "analysis only from cleaned Raw")
        _detail("Cleaned Raw", input_path)
        _detail("Montage", montage_type or "N/A")
        _detail("Date", metadata["recording_date"])
    else:
        _stage(stage_i, total, "Importing raw EEG")
        stage_i += 1
        imp = EEGRawImporter(
            path_to_file=str(input_path),
            helper_dir=str(helper_dir),
            keep_stim=True,
            verbose=True,
        )

        raw = imp.raw
        ch_set = imp.ch_set
        montage_type = args.montage_type or imp.montage.get("montage_type")
        if args.montage_type and args.montage_type != imp.montage.get("montage_type"):
            raise ValueError("--montage-type does not match the inferred DAT montage; cannot relabel the source data")
        metadata = apply_overrides(metadata_from_raw(raw, input_path, {key: value for key, value in imp.stream.items() if key not in {"signal", "states", "ch_names"}}), args)
        history = {}
        aux_source_channels = list(imp.montage.get("aux_source_channels", []))

        _detail("Montage", montage_type or "N/A")
        if aux_source_channels:
            _detail("Dropped source rows", ", ".join(aux_source_channels))
        _detail("Date", metadata["recording_date"])
        _detail("Output folder", save_path)
        _detail("Images folder", images_path)
        if not args.skip_csv:
            _detail("CSV folder", csv_path)

        if not args.skip_bridging:
            _stage(stage_i, total, "Running bridging check")
            stage_i += 1
            _detail("Frequency range", f"{args.bridge_fmin:g}-{args.bridge_fmax:g} Hz")
            _detail("Window", f"{args.bridge_window_sec:g} s")
            bridge = BridgingChecker(
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
                save_path=str(images_path),
            )
            if hasattr(bridge, "bridging_checker_fig"):
                import matplotlib.pyplot as plt
                plt.close(bridge.bridging_checker_fig)

        if not args.skip_preprocessing:
            config = _build_preproc_config(args)
            _stage(stage_i, total, "Running preprocessing")
            stage_i += 1
            _detail("Steps", _format_step_names(config))
            preproc = EEGPreprocessor(
                raw,
                ch_set,
                config=config,
                copy=True,
                montage_type=montage_type,
                verbose=args.preproc_verbose,
            )
            raw, history = preproc.run()
            ch_set = preproc.ch_set
            _detail("Bad channels", len(raw.info.get("bads", [])))

            if args.save_cleaned_raw:
                cleaned_output = (
                    Path(args.cleaned_raw_output).expanduser().resolve()
                    if args.cleaned_raw_output
                    else _default_cleaned_raw_output(save_path, input_path)
                )
                metadata.update({
                    "source_file": str(input_path),
                    "montage_type": montage_type,
                    "dropped_source_channels": aux_source_channels,
                    "n_channels": int(raw.info["nchan"]),
                    "sfreq": float(raw.info["sfreq"]),
                    "preprocessing_history": history,
                })
                saved_cleaned = _save_cleaned_raw(
                    raw,
                    cleaned_output,
                    metadata=metadata,
                    overwrite=bool(args.overwrite_cleaned_raw),
                )
                _detail("Saved cleaned Raw", saved_cleaned)
        else:
            _detail("Preprocessing", "skipped")

    metadata.update(montage_type=montage_type, sfreq=float(raw.info["sfreq"]),
                    highpass=float(raw.info["highpass"]), lowpass=float(raw.info["lowpass"]),
                    n_eeg_channels=len(mne.pick_types(raw.info, eeg=True, exclude=())),
                    eeg_channel_labels=[raw.ch_names[i] for i in mne.pick_types(raw.info, eeg=True, exclude=())],
                    duration_sec=float(raw.n_times / raw.info["sfreq"]),
                    spatial_filter=spatial_filter_label(history), preprocessing_history=history)
    (save_path / "recording_metadata.json").write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")

    if not args.skip_analysis:
        # Preprocessing can change labels and channel counts; use the resulting Raw geometry
        ch_set = CleanedEEGChannelSet(raw)
        _stage(stage_i, total, "Running motor imagery analysis")
        stage_i += 1
        _detail("Epochs per trial", args.n_epochs)
        _detail("PSD resolution", f"{args.resolution:g} Hz/bin")
        _detail("Frequency bands", ", ".join(f"{x:g}" for x in args.freq_bands))
        _detail("CSV export", "off" if args.skip_csv else "on")
        analysis = EEGMotorImagery(
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
            save_path=str(save_path),
            export_csv=not args.skip_csv,
            random_state=args.analysis_random_state,
            run_decoding=not args.skip_decoding,
        )

        if run_one_over_f:
            _stage(stage_i, total, "Running 1/f-subtracted motor imagery analysis")
            stage_i += 1
            _detail("Output", one_over_f_save_path)
            _detail("PSD scheme", "subtract selected 1/f background in dB")
            _detail("Fit diagnostics", args.one_over_f_plot_channels)
            residual_analysis = EEGMotorImagery(
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
                save_path=str(one_over_f_save_path),
                export_csv=not args.skip_csv,
                random_state=args.analysis_random_state,
                run_decoding=not args.skip_decoding,
                psd_processing="one_over_f_subtracted",
                background_fit_plot_channels=_parse_optional_channel_list(args.one_over_f_plot_channels),
            )
            _copy_optional_report_assets(images_path, one_over_f_images_path)
            (one_over_f_save_path / "recording_metadata.json").write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")

    if not args.skip_report:
        _stage(stage_i, total, "Building PDF report(s)" if run_one_over_f else "Building PDF report")
        MotorImageryPdfReport(
            plot_folder=str(images_path),
            helper_folder=str(helper_dir),
            date_test=metadata["recording_date"],
            montage_name=montage_type or "N/A",
            resolution=args.resolution,
            age_at_test=metadata.get("age_at_test", "N/A"),
            recording_metadata=metadata,
            save_folder=str(save_path),
            report_title="Command Following Report",
            report_name=output_name,
            analysis_parameters=analysis.parameter_summary,
        )
        if run_one_over_f:
            MotorImageryPdfReport(
                plot_folder=str(one_over_f_images_path),
                helper_folder=str(helper_dir),
                date_test=metadata["recording_date"],
                montage_name=montage_type or "N/A",
                resolution=args.resolution,
                age_at_test=metadata.get("age_at_test", "N/A"),
                recording_metadata=metadata,
                save_folder=str(one_over_f_save_path),
                report_title="Command Following Report (1/f-subtracted PSD)",
                report_name=output_name,
                analysis_parameters=residual_analysis.parameter_summary,
            )

    print("\nPipeline complete")
    _detail("Input", input_path)
    _detail("Output folder", save_path)
    _detail("Images folder", images_path)
    if not args.skip_csv:
        _detail("CSV folder", csv_path)
    if run_one_over_f:
        _detail("1/f output folder", one_over_f_save_path)
        _detail("1/f images folder", one_over_f_images_path)
        if not args.skip_csv:
            _detail("1/f CSV folder", one_over_f_csv_path)


if __name__ == "__main__":
    main()
