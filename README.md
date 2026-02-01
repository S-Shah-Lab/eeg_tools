# eeg_tools

A Python toolkit for **EEG motor imagery** pipelines built around **BCI2000 `.dat`** recordings: import → (optional) bridged-channel QC → preprocessing → time–frequency (PSD) analysis → stats → **PDF report**.

---

## What's inside

### Core modules
- **`RawImporter.py`**: Loads BCI2000 `.dat` files into an `mne.io.Raw` object (plus montage/channel set metadata).
- **`BridgingChecker.py`**: Unsupervised **bridged-channel candidate detection** using a *correlation × affinity* score derived from distance-based Gaussian affinity matrices.
- **`Preprocessing.py`**: Configurable preprocessing pipeline (notch, band-pass, PREP noisy-channel detection, optional manual annotations, interpolation, re-reference, spatial filtering).
- **`MotorImagery.py`**: Motor imagery analysis using **Welch PSD** computed over within-trial segments (time-by-epoch, frequency-by-band), with permutation/bootstrapped statistics and report-ready plots.
- **`PdfReport.py`**: Assembles plots into a **PDF report**.

### CLI entry points
- `run_RawImporter.py`
- `run_BridgingChecker.py`
- `run_Preprocessing.py`
- `run_MotorImagery.py` (full pipeline + PDF report)

Helper assets (montage locations, etc.) live in `eeg_tools/helper/`.

---

## Requirements

This package assumes you already live in Python EEG-land:

- `numpy`
- `mne`
- `matplotlib`
- `scipy`
- `pyprep` (used for PREP noisy-channel detection)
- `reportlab`, `svglib`, `Pillow` (PDF report generation)
- **BCI2000 Python tools** (`BCI2000Tools`) for reading `.dat` streams and channel sets

> Note: `BCI2000Tools` must be importable (installed or on your `PYTHONPATH`). The toolkit imports modules like `BCI2000Tools.FileReader` and `BCI2000Tools.Electrodes`.

---

## Quickstart: full motor imagery pipeline (recommended)

`run_MotorImagery.py` runs:
1) import (`.dat` → `mne.Raw`)
2) (optional) bridging analysis
3) (optional) preprocessing
4) (optional) motor imagery analysis
5) (optional) PDF report

### Minimal run (provide the `.dat` file path)
```bash
python eeg_tools/run_MotorImagery.py \
  --file-path /path/to/sub-XXX_ses-01_task-MotorImag_run-01.dat \
  --helper-dir eeg_tools/helper
```

### Output
If you do not set `--save-path`, outputs go to:
```
<root>/<base_name_without_ext>/
```
and include:
- plots from bridging / preprocessing / analysis
- a generated PDF report (if not skipped)

---

## Bridged-channel QC (standalone)

Run the bridging checker directly:
```bash
python eeg_tools/run_BridgingChecker.py \
  --file-path /path/to/recording.dat \
  --helper-dir eeg_tools/helper \
  --save-path /path/to/output_folder
```

Key parameters:
- `--fmin`, `--fmax`: band-pass bounds for the QC signal
- `--sigma`: Gaussian kernel width for affinity
- `--window-sec`: sliding window length (non-overlapping)
- `--bridge-score-threshold`: threshold on *(correlation × affinity)*

---

## Preprocessing (standalone)

```bash
python eeg_tools/run_Preprocessing.py \
  --file-path /path/to/recording.dat \
  --helper-dir eeg_tools/helper
```

You can disable steps:
- `--no-notch`
- `--no-bandpass`
- `--no-prep`
- `--no-annotation`
- `--no-interpolation`
- `--no-rereference`
- `--no-spatialfilter`

...and control parameters like:
- `--notch-freqs` (default 60)
- `--bandpass-lfreq`, `--bandpass-hfreq`
- PREP toggles: `--prep-no-correlation`, `--prep-no-deviation`, etc.

---

## Motor imagery analysis details

The analysis is designed around "time-frequency" in a pragmatic sense:

- **Time:** trials are split into `--n-epochs` segments per trial after skipping `--skip` seconds.
- **Frequency:** Welch PSD is computed per segment with resolution `--resolution` (Hz/bin).
- **Bands:** defined by edges in `--freq-bands` (default `4,8,13,31`).

Stats are computed with:
- permutation testing (number of simulations `--n-sim`)
- bootstrap testing for robustness / uncertainty reporting

---

## Common CLI recipes

### Skip bridged-channel QC
```bash
python eeg_tools/run_MotorImagery.py --file-path /path/to/file.dat --skip-bridging
```

### Skip preprocessing (run analysis on raw import)
```bash
python eeg_tools/run_MotorImagery.py --file-path /path/to/file.dat --skip-preprocessing
```

### Run analysis only (no PDF report)
```bash
python eeg_tools/run_MotorImagery.py --file-path /path/to/file.dat --skip-report
```

### Change frequency bands and PSD resolution
```bash
python eeg_tools/run_MotorImagery.py \
  --file-path /path/to/file.dat \
  --freq-bands 4,8,13,30,45 \
  --resolution 0.5
```

---

## Python API usage (direct)

### Import a recording
```python
from eeg_tools.RawImporter import EEGRawImporter

imp = EEGRawImporter(
    path_to_file="/path/to/recording.dat",
    helper_dir="eeg_tools/helper",
    keep_stim=False,
    verbose=False,
)

raw = imp.raw          # mne.io.Raw
ch_set = imp.ch_set    # BCI2000Tools ChannelSet
```

### Bridging checker
```python
from eeg_tools.BridgingChecker import BridgingChecker

bc = BridgingChecker(
    raw=raw,
    fmin=1.0,
    fmax=40.0,
    sigma=0.05,
    window_sec=10.0,
    bridge_score_threshold=0.095,
    save_path="./outputs",
    verbose=False,
)
```

### Preprocessing
```python
from eeg_tools.Preprocessing import EEGPreprocessor, EEGPreprocessorConfig

config = EEGPreprocessorConfig([
    ("notch", {"freqs": 60.0, "kwargs": {}}),
    ("bandpass", {"l_freq": 1.0, "h_freq": 100.0, "kwargs": {}}),
    ("prep", {"random_state": 83092, "correlation": True, "deviation": True,
              "hf_noise": True, "nan_flat": True, "ransac": True}),
    ("interpolation", {"reset_bads_after_interp": True}),
    ("rereference", {"channels": "tp9 tp10"}),
])

preproc = EEGPreprocessor(raw, ch_set, config=config, copy=True, verbose=False)
raw_clean, history = preproc.run()
```

### Motor imagery analysis + plots
```python
from eeg_tools.MotorImagery import EEGMotorImagery

mi = EEGMotorImagery(
    raw_clean,
    ch_set,
    nEpochs=6,
    duration_task=10.0,
    skip=1.0,
    resolution=1.0,
    freq_bands=[4.0, 8.0, 13.0, 31.0],
    nSim=2999,
    strict=False,
    copy=True,
    verbose=False,
    save_path="./outputs",
)
```

### PDF report
```python
from eeg_tools.PdfReport import MotorImageryPdfReport

MotorImageryPdfReport(
    plot_folder="./outputs",
    helper_folder="eeg_tools/helper",
    date_test="N/A",
    montage_name="N/A",
    resolution=1,
    age_at_test="20",
    save_folder="./outputs",
)
```

---

## Data assumptions

- Input is a **BCI2000 `.dat`** file readable by `BCI2000Tools.FileReader.bcistream`.
- Montages are inferred via helper files, unless overridden in the CLI using `--montage-type`.
- Some workflows may include stimulus/state channels; use `--keep-stim` if you need them.

---

## Project layout

```
eeg_tools/
  BridgingChecker.py
  MotorImagery.py
  PdfReport.py
  Preprocessing.py
  RawImporter.py
  run_BridgingChecker.py
  run_MotorImagery.py
  run_Preprocessing.py
  run_RawImporter.py
  helper/
    *_location.txt
    eeg_dict.py
    ...
```

---

## Troubleshooting

- **`ModuleNotFoundError: BCI2000Tools...`**
  You need the BCI2000 Python tools installed and importable.

- **PDF report generation fails**
  Ensure `reportlab`, `svglib`, and `Pillow` are installed. The report generator also expects certain plot files to exist in the output folder.

- **Montage/channel name mismatches**
  Check `eeg_tools/helper/` location files and `helper/eeg_dict.py`, or override montage selection with `--montage-type`.

---

## License

TODO