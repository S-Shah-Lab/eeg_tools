# eegmcf — EEG motor command following

Version 1.1.1. Import BCI2000 recordings, check bridged channels, preprocess EEG, analyse motor imagery and generate command-following PDF reports. Supports both new `.dat` recordings and existing cleaned `.fif` / `.fif.gz` recordings.

## 1. Installation

Use Python 3.10 or newer, preferably 3.11 or 3.12, in a dedicated environment. Open a terminal in the extracted `eeg_tools` directory (the directory containing `pyproject.toml`).

```bash
python -m venv .venv
# Linux / WSL / macOS: source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[raw,decoding]"
```

The `raw` extra installs BCI2000Tools and PyPREP for new DAT recordings; `decoding` installs scikit-learn for optional conventional-PSD decoding. To analyse only cleaned FIFs, use `python -m pip install -e ".[decoding]"` (or `python -m pip install -e .` without decoding). Neither acquisition dependency is needed for cleaned-FIF input. The dependency ranges and package version are defined in `pyproject.toml`.

If the package index cannot supply BCI2000Tools, install it from the `tools/python` directory of your existing BCI2000 distribution, using the same Python environment:

```bash
python -m pip install -e /path/to/BCI2000/tools/python
python -m pip install "pyprep>=0.5,<0.11"
python -m pip install -e ".[decoding]"
```

See the [BCI2000 Python tools](https://www.bci2000.org/mediawiki/index.php/BCI2000Tools.EventRelated) installation instructions. Check the real raw-input dependencies and commands before processing:

```bash
python -c "from BCI2000Tools.FileReader import bcistream; from BCI2000Tools.Electrodes import ChannelSet; from pyprep.find_noisy_channels import NoisyChannels; print('Raw input dependencies OK')"
eegmcf --help
eegmcf-import --help
eegmcf-bridging --help
eegmcf-preprocess --help
```

| Command | Purpose |
|---|---|
| `eegmcf` | Full DAT-to-report or cleaned-FIF analysis |
| `eegmcf-import` | DAT import and optional validation plots |
| `eegmcf-bridging` | Candidate bridged-channel QC |
| `eegmcf-preprocess` | Import and preprocessing; saves FIF and metadata |

The commands can also be invoked as Python modules, e.g. `python -m eegmcf.run_MotorImagery`. Montage files and PDF assets are bundled and located automatically. For unattended or WSL/headless runs use `--no-annotation` and, if needed, `MPLBACKEND=Agg` (PowerShell: `$env:MPLBACKEND = 'Agg'`). For manual BAD-region review, use a working interactive MNE backend and omit `--no-annotation`.

## 2. Recommended workflow

### A. Run the complete pipeline from a DAT recording

```bash
eegmcf --file-path /path/to/sub-DEMO001_ses-01_task-MotorImag_run-01.dat --save-path ./results --age-at-test 42 --no-annotation
```

This imports the DAT with BCI2000 state channels, performs bridge QC, runs preprocessing, saves a cleaned FIF and JSON sidecar, computes Welch PSDs and statistics, and writes two reports: conventional PSD and 1/f-background-subtracted PSD. Omit `--no-annotation` for interactive manual review. The default `--reref-channels auto` uses TP9/TP10 when both are present. Otherwise the acquisition reference is retained with a warning and an explicit skipped-step record. Verify that this is appropriate for the cap and protocol. Set `--reref-channels "tp9 tp10"` or another verified reference explicitly when needed; use `--no-rereference` to retain the acquisition reference on any montage. Choose the appropriate mains frequency (`--notch-freqs 50` or `60`).

### B. Reanalyse a previously cleaned FIF

```bash
eegmcf --cleaned-raw-input /path/to/sub-DEMO001_ses-01_task-MotorImag_run-01_cleaned_raw.fif --save-path ./results
```

Keep the adjacent `.json` file with the FIF. The saved metadata and `StimulusCode` (and `PresentationDisplayed`, if present) must be retained. This route bypasses import, bridging and preprocessing. `--file-path` also accepts `.fif` or `.fif.gz`. The analysis uses the EEG labels, order and electrode positions saved in the FIF, not the original helper file. Reordered, reduced or renamed EEG sets are supported when their actual electrode identities and positions are valid; no data are re-referenced, spatially filtered or relabeled. Stimulation and other non-EEG channels are excluded from PSDs. The acquisition montage type is taken from the sidecar or inferred from matching helper labels; if ambiguous, the report displays `N/A`. Use `--montage-type DSI_24`, `GTEC_32`, `EGI_64` or `EGI_128` if the hardware is known. This option only identifies the hardware in analysis-only mode; it does not override the saved positions. The FIF must have valid electrode positions, recognizable bilateral motor-imagery electrode labels and no unresolved bad EEG channels. Bridge QC plots are not automatically recovered from an earlier output folder.

### C. Import, QC or preprocess independently

```bash
eegmcf-import --file-path /path/to/recording.dat --keep-stim --validation-plots --plot-dir ./validation
eegmcf-bridging --file-path /path/to/recording.dat --save-path ./qc
eegmcf-preprocess --file-path /path/to/recording.dat --save-path ./preprocessed --no-annotation
```

The importer can export signal, channel-quality, PSD, sensor-layout and state-channel plots. The bridge checker identifies **candidate** bridged groups, not confirmed bridges, and does not alter the recording used by the rest of the pipeline. The standalone preprocessor saves `<recording>_cleaned_raw.fif` and its same-stem JSON sidecar, with state channels retained by default.

## 3. Recording and report metadata

Both reports repeat the recording ID, subject ID, session, run, recording date, report generation date (UTC), age, condition, montage, sampling rate, filter range, spatial-filter status and actual PSD frequency resolution on each page. The header also includes the page number. The unique report UUID, exact UTC generation timestamp and software version are retained in the generation metadata JSON rather than printed in the header.

| Field | Source / priority |
|---|---|
| Recording ID | Original file stem, or `--recording-id` |
| Subject/session/run | BIDS-like `sub-`, `ses-`, `run-` filename tokens; otherwise BCI2000 `SubjectName`, `SubjectSession`, `SubjectRun` when available; explicit flags override |
| Recording date | DAT `StorageTime`; saved sidecar for FIF; MNE `meas_date` (UTC) if sidecar date unavailable; `--recording-date` overrides |
| Recording timestamp | Original timestamp in sidecar when available; a timestamp without a timezone is not assumed to be UTC |
| Age / condition | Operator-provided or saved metadata; condition is not inferred as a clinical diagnosis from an ID |
| Generated date/time | Current UTC clock at PDF creation; each regenerated PDF receives a new report UUID |
| Filter and PSD settings | Actual Raw/analysis metadata, not hard-coded default values |

Unknown metadata appears as `N/A`, never as a fabricated recording date or subject. File modification time is not used as a recording date. Use pseudonymous subject IDs and keep the sidecars/reports in an appropriate restricted location.

Example supplying explicit fields (also supported by `eegmcf-preprocess`):

```bash
eegmcf --file-path /path/to/recording.dat --save-path ./results --recording-id MI-0001 --subject-id DEMO001 --session-id 02 --run-id 03 --recording-date 2026-08-19 --age-at-test 42 --condition HC --no-annotation
```

`--recording-date` accepts a date (`YYYY-MM-DD`) or ISO 8601 timestamp with an optional UTC offset. Do not enter an inferred condition or age unless verified. Operator overrides are retained in the cleaned FIF's JSON sidecar. The original source file name is preserved in metadata even when the cache is renamed.

## 4. Main configuration

All flags are listed in the relevant script's `--help`. Defaults below refer to `eegmcf` unless noted.

| Flag | Default / purpose |
|---|---|
| `--file-path` / `--cleaned-raw-input` | Provide exactly one input |
| `--save-path` | Required output root |
| `--helper-dir` | Bundled helper folder |
| `--skip-bridging` | Skip bridge candidate QC |
| `--bridge-fmin`, `--bridge-fmax` | 1, 40 Hz |
| `--bridge-sigma`, `--bridge-window-sec`, `--bridge-score-threshold` | 0.05, 10 s, 0.095 |
| `--skip-preprocessing` | Skip every preprocessing step |
| `--no-notch`, `--no-bandpass`, `--no-prep` | Disable individual cleaning steps |
| `--notch-freqs` | 60 Hz base (harmonics below Nyquist) |
| `--bandpass-lfreq`, `--bandpass-hfreq` | 1, 40 Hz |
| `--prep-random-state` | 83092 |
| `--prep-no-correlation`, `--prep-no-deviation`, `--prep-no-hf-noise`, `--prep-no-nan-flat`, `--prep-no-ransac` | Disable individual PREP criteria |
| `--no-annotation` | Skip interactive manual bad-region marking |
| `--no-annotation-plot` | Do not open annotation window |
| `--no-interpolation` | Do not interpolate bad channels |
| `--no-reset-bads-after-interp` | Retain bad flags; unresolved EEG bad flags are rejected before MI analysis to prevent PSD row misalignment |
| `--reref-channels` | `auto`: TP9/TP10 if both present; otherwise retain acquisition reference. Use explicit reference channels if appropriate |
| `--no-rereference`, `--no-spatialfilter` | Skip reference or surface Laplacian |
| `--no-save-cleaned-raw` | Do not save cleaned FIF/JSON |
| `--cleaned-raw-output` | Explicit cleaned FIF destination |
| `--no-overwrite-cleaned-raw` | Protect a pre-existing cache |
| `--n-epochs`, `--duration-task`, `--skip` | 6 segments, 10 s task window, skip 1 s once at its beginning |
| `--resolution`, `--freq-bands` | 1 Hz/bin; band edges 4,8,13,31 Hz |
| `--n-sim`, `--analysis-random-state` | 9999 simulations; seed 83092 |
| `--strict` | Use a restricted motor-imagery channel set |
| `--skip-decoding` | Skip optional task/rest decoding |
| `--skip-one-over-f-report` | Skip the complete residual-PSD analysis branch |
| `--one-over-f-plot-channels` | `c3,c4`; use `false` to disable fit diagnostic plots |
| `--skip-csv` | Skip ordinary analysis CSV exports; the 1/f fit still writes its intermediate and fit CSVs |
| `--skip-report` | Analyse and export without PDF |
| `--skip-analysis --skip-report` | Import/QC/preprocessing only |
| `--bridge-verbose`, `--preproc-verbose`, `--analysis-verbose` | Detailed progress |

Welch segments have approximately `1 / resolution` seconds of data and must fit inside `(duration_task - skip) / n_epochs`. With the defaults, each epoch is 1.5 seconds long. For 0.5 Hz resolution, for example, reduce the number of epochs to 4 or fewer. The effective frequency spacing is sampling rate divided by the integer FFT size. Ensure requested bands are covered by the recording and chosen filter. For a quick **smoke test**, use `--n-sim 99 --skip-one-over-f-report`; do not use a small simulation count for final inference.

The standalone preprocessor also accepts multiple `--notch-freqs` values, `--prep-hf-noise-action review|mark`, `--annotation-label` and `--spatial-exclude`. PREP high-frequency-noise-only detections default to review rather than automatic bad marking.

## 5. Output files

Each pipeline invocation creates its own output directory under `--save-path`; a nonempty existing directory causes a numbered `_analysis-02`, `_analysis-03`, etc. directory to be created to prevent stale plots or reports from being reused.

```text
results/<recording>/
  <recording>_cleaned_raw.fif       # if preprocessing and cache saving are enabled
  <recording>_cleaned_raw.json      # cache metadata and preprocessing history
  recording_metadata.json           # recording identity/provenance and settings
  report_metadata.json              # analysis parameters, warnings, results and export inventory
  pdf_report_generation_metadata.json
  mcf_report_<recording>.pdf
  images/                           # PNG and SVG report/diagnostic figures
  csv/                              # analysis summaries (unless disabled)
  one_over_f_subtracted/            # optional second analysis and report; own images/csv/metadata
```

The ordinary CSV outputs include `analysis_parameters.csv`, `paradigm_validation_summary.csv`, `trial_quality_summary.csv`, `psd_mean_per_channel_per_bin.csv`, `psd_long_format.csv`, `signed_r2_per_band_per_channel.csv`, `stat_summary_per_band.csv`, `bootstrap_stats_per_band.csv`, `erd_ers_timecourse.csv`, and `decoding_summary.csv` when results exist. The 1/f branch additionally exports its background fit inputs, selected-model fits and peak summaries. Conventional mean PSD uses power units (V²/Hz); the residual branch contains dB values, not physical power. Consult `report_metadata.json` for the PSD mode, warnings, and generated files.

The PDF contains three pages: timeline/topographic results/band effects; selected channel spectra; statistical distributions and bridge candidates. The original report layout and PSD-to-electrode guide lines are preserved, with aspect-ratio-preserving height limits on page 2 to prevent adjacent spectra from overlapping. Missing optional plots are skipped and recorded in the generation metadata rather than replaced by stale images. The JSON beside each PDF records the report UUID, date and time, recording metadata, analysis parameters, warnings and figure inventory.

## 6. Direct Python API and existing plots

Use the installed package. Always retain the stimulus/state channels for motor imagery analysis. The example below runs band-pass filtering only; use the CLI for the full preprocessing workflow.

```python
from eegmcf.RawImporter import EEGRawImporter
from eegmcf.Preprocessing import EEGPreprocessor, EEGPreprocessorConfig
from eegmcf.MotorImagery import EEGMotorImagery
from eegmcf.PdfReport import MotorImageryPdfReport

imp = EEGRawImporter('/path/to/recording.dat', keep_stim=True)
config = EEGPreprocessorConfig([
    ('bandpass', {'l_freq': 1.0, 'h_freq': 40.0}),
])
pre = EEGPreprocessor(imp.raw, imp.ch_set, config=config,
                      montage_type=imp.montage['montage_type'])
raw, history = pre.run()
analysis = EEGMotorImagery(raw, pre.ch_set, save_path='./analysis', nSim=9999)
```

To rebuild a PDF without repeating analysis, call the report class with the existing plots and their corresponding metadata. It reads `recording_metadata.json` and `report_metadata.json` from the parent of the images directory automatically. Do not mix plots and metadata from different recordings.

```python
from eegmcf.PdfReport import MotorImageryPdfReport
from pathlib import Path
from eegmcf.RecordingMetadata import HELPER_DIR

folder = Path('./results/sub-DEMO001_ses-01_task-MotorImag_run-01')
report = MotorImageryPdfReport(
    plot_folder=folder / 'images',
    helper_folder=HELPER_DIR,
    save_folder=folder,
    report_name=folder.name,
)
print(report.output_path)
```

Standalone background fitting:

```bash
python -m eegmcf.MotorImagery_aggregate_psds_fit_background --input-csv /path/to/csv/psd_long_format.csv --plot-channels c3,c4
```

## 7. Interpretation and validation

- The acquisition paradigm is protocol-specific. It uses BCI2000 `StimulusCode` and optionally `PresentationDisplayed`, codes 1/2 for left/right and 3/4 for corresponding rest periods, a legacy 2,000-sample initial trigger exclusion, and trial grouping keyed by left-task starts. Verify event counts and the timeline for every recording; it is not a general-purpose EEG event decoder.
- Supported source layouts are DSI 24 (20 EEG rows, optionally with recognized DSI auxiliary rows; 20/21/24 source rows), g.Nautilus 32, and EGI 64/128 (128-source-row layout, with EGI 64 inferred using flat-channel counts). Recognized electrode labels are aligned to helper order. EGI numbered source labels are normalized and ordered before the EGI 64 selection. Generic ordered GTEC Ch1..Ch32 labels use the bundled helper order with a warning; verify this acquisition order. Unknown or ambiguous labels are rejected instead of silently relabeling signals. Confirm channel names, calibration, montage, reference and recording units against your actual acquisition. BCI2000 decoded values are assumed to be in microvolts and converted to volts.
- The original lateralized signed-r² statistic is retained. Permutation p-values now use the conservative plus-one correction with ties counted; a constant statistic cannot generate a spurious significant result. Bootstrap tail scores and percentile intervals are descriptive/exploratory. Validate the exchangeability assumption and any paired/blocked design before interpreting p-values; selected best-band/ROI decoding summaries are exploratory.
- The 1/f branch fits backgrounds separately by condition/channel using the full data. Therefore its permutation results are exploratory (the fits are not repeated inside permutations), and decoding is disabled for this branch to avoid label leakage. This workflow is not established as a validated clinical diagnostic test.
- The regression suite covers EEG row alignment, raw-import orchestration using a synthetic decoded stream, MNE filtering, state preservation, FIF round-trip, recording metadata, and PDF placement. BCI2000Tools and PyPREP were unavailable in the review runtime. Raw tests use test doubles for the BCI2000 decoder and reference/spatial-filter matrices; they do not certify DAT decoding, actual SLAP/RerefMatrix behavior or PREP. Test a representative real DAT on your acquisition environment before relying on results.

Run regression tests after installing the package, from the extracted repository root:

```bash
python -m unittest discover -s tests -v
```

For a first check on a **real new DAT recording**, keep all preprocessing steps enabled and temporarily reduce the simulation count:

```bash
eegmcf --file-path /path/to/recording.dat --save-path ./test_results --no-annotation --n-sim 99 --skip-one-over-f-report
```

Inspect the bridge results, bad-channel counts, trial counts, montage and reference decisions, cleaned FIF/sidecar and three-page PDF. Repeat with your validated acquisition settings and the default simulation count for final analysis. If the run fails, do not blindly disable cleaning steps; check the message, reference, input labels and paradigm. `--no-prep` is available for explicit diagnostic testing only, not a default production fallback. See `CSV_OUTPUTS_EXPLAINED.md` for output details.
