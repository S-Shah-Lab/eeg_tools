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
- `run_BridgingChecker.py`
- `run_MotorImagery.py`

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

## 1. Importing (module)
  ```bash
  RawImporter.py
  ```

  Key parameters:
  - `--keep-stim`: allows to keep stim channels attached to the mne.io.Raw object

  Note: 
  - `RawImporter.py` is a module that is imported by other scripts e.g. `run_BridgingChecker.py`
  - Look at `run_BridgingChecker.py` in the next section for a proper implementation or API usage below
  
---

## 2. Bridged-channel QC (standalone)
  Run the bridging checker directly:
  ```bash
  python eeg_tools/run_BridgingChecker.py \
    --file-path /path/to/recording.dat \
    --helper-dir ./helper  \
    --show-extra \
    --save-path /path/to/output_folder \
    --verbose
  ```

  Key parameters:
  - `--fmin`, `--fmax`: Band-pass filter; default = 1.0, 40.0
  - `--sigma`: Gaussian kernel width for affinity; default = 0.05 (don't change)
  - `--window-sec`: Sliding window length (non-overlapping); default = 10.0 
  - `--bridge-score-threshold`: Threshold on *(correlation × affinity)*; default = 0.095 (don't change)
  - `--show-extra`: Optional, will show plots but still generate output image of bridged candidates

  Note: 
  - `run_BridgingChecker.py` has optimized parameters, the bridged channel groups are only candidates, but if the results are weird, feel free to change the default parameters
  
---

## 3. Preprocessing (module)
  ```bash
  Preprocessing.py
  ```
  
  These are all the steps that can be performed, they are all optional:
    - `Notch filter`                  (Raw.notch_filter)       [optional]
    - `Band-pass filter`              (Raw.filter)             [optional]
    - `PREP bad-channel detection`    (NoisyChannels)          [optional]
    - `Interpolate bad channels`      (Raw.interpolate_bads)   [optional]
    - `Re-reference`                  (ChannelSet.RerefMatrix) [optional]
    - `Spatial filter`                (ChannelSet.SLAP)        [optional]
    - `Manual BAD segment annotation` (Raw.plot)               [optional]

  Note: 
  - `Preprocessing.py` is a module that is imported by other scripts e.g. `run_BridgingChecker.py`
  - A `config` dictionary needs to be passed to the class to initialize the steps properly
  - Look at `run_MotorImagery.py` in the next section for a proper implementation or API usage below

---

## 4. Motor Imagery Analysis

The analysis is designed as follows:

- Each trial is split into `--n-epochs` segments after skipping `--skip` seconds after instruction
-  Welch PSD is computed per segment with resolution `--resolution` (Hz/bin)
- A statistical analysis is performed for each frequency band, defined by edges in `--freq-bands` (default `4,8,13,31`)

Stats are computed with:
- Permutation testing
- Bootstrap testing for robustness / uncertainty reporting

Quickstart: full pipeline (recommended)

`run_MotorImagery.py` runs:
1) import (`.dat` → `mne.Raw`)
2) (optional) bridging analysis
3) (optional) preprocessing
4) (optional) motor imagery analysis
5) (optional) PDF report

### Minimal run (provide the `.dat` file path)
```bash
python eeg_tools/run_MotorImagery.py \
  --file-path /path/to/file.dat \
  --helper-dir ./helper \
  --save-path /path/to/output_folder \
  --age-at-test AGE_OF_SUBJECT
```

  Key parameters:
    - `--file-path`:  Path to motor imagery .dat file
    - `--helper-dir`: Path to helper directory
    - `--save-path`:  Path to directory for output save
    
    - `--skip-bridging`:                Skip bridging analysis
    - `--bridge-fmin`, `--bridge-fmax`: Band-pass filter; default = 1.0, 40.0
    - `--bridge-sigma`:                 Gaussian kernel width for affinity; default = 0.05 (don't change)
    - `--bridge-window-sec`:            Sliding window length (non-overlapping); default = 10.0 
    - `--bridge-score-threshold`:       Threshold on *(correlation × affinity)*; default = 0.095 (don't change)
    - `--bridge-verbose`:               Verbose output during bridging analysis
    
    - `--skip-preprocessing`: Skip preprocessing pipeline
    - `--no-notch`:           Disable notch filter step
        - `--notch-freqs`: Notch filter frequencies and multiples; default = 60

    - `--no-bandpass`: Disable band-pass filter step
        - `--bandpass-lfreq`, `--bandpass-hfreq`: Band pass filter; default = 1.0, 40.0

    - `--no-prep`:          Disable PREP noisy-channel detection step
        - `--prep-no-correlation`: Disable PREP correlation criterion
        - `--prep-no-deviation`:   Disable PREP deviation criterion
        - `--prep-no-hf-noise`:    Disable PREP HF-noise criterion
        - `--prep-no-nan-flat`:    Disable PREP NaN/flat criterion
        - `--prep-no-ransac`:      Disable PREP RANSAC criterion
        - `--prep-random-state:    Seed for RANSAC
        - `--preproc-verbose`: Verbose preprocessing output

    - `--no-annotation`:    Disable manual BAD-region annotation step

    - `--no-interpolation`: Disable bad-channel interpolation step
        - `--reset-bads-after-interp`: Reset raw.info['bads'] after interpolation

    - `--no-rereference`:   Disable re-referencing step
        - `--reref-channels`: Channels used for re-referencing, e.g. "tp9 tp10"

    - `--no-spatialfilter`: Disable spatial filtering step, otherwise SLAP is applied
    
    - `--skip-analysis`:    Skip Motor Imagery analysis
    - `--n-epochs`:         Number of epochs per trial
    - `--duration-task`:    Task duration (s)
    - `--skip`:             Seconds to skip at start of each segment
    - `--resolution`:       PSD frequency resolution (Hz/bin)
    - `--freq-bands`:       Comma-separated band edges, e.g. 4,8,13,31
    - `--n-sim`:            Number of simulations for stats tests
    - `--strict`:           Use strict motor imagery channel set (else more channels are used)
    - `--analysis-verbose`: Verbose motor imagery analysis output
    
    - `--skip-report`: Skip PDF report generation
    - `--age-at-test`: Age at test (string, used in the PDF header), otherwise N/A will show

### Output
- plots from bridging / preprocessing / analysis
- a generated PDF report (if not skipped)

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
  --resolution 2.0
```

---

## Python API usage (direct)

### 1. Import a recording
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

### 2. Bridging checker
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

### 3. Preprocessing
```python
from eeg_tools.Preprocessing import EEGPreprocessor, EEGPreprocessorConfig

config = EEGPreprocessorConfig([
    ("notch", {"freqs": 60.0, "kwargs": {}}),
    ("bandpass", {"l_freq": 1.0, "h_freq": 40.0, "kwargs": {}}),
    ("prep", {"random_state": 83092, "correlation": True, "deviation": True,
              "hf_noise": True, "nan_flat": True, "ransac": True}),
    ("interpolation", {"reset_bads_after_interp": True}),
    ("rereference", {"channels": "tp9 tp10"}),
])

preproc = EEGPreprocessor(raw, ch_set, config=config, copy=True, verbose=False)
raw_clean, history = preproc.run()
```

### 4. Motor Imagery Analysis
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

### 5. PDF report
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
- Input is a **BCI2000 `.dat`** file readable by `BCI2000Tools.FileReader.bcistream`
- Montages are inferred via helper files, unless overridden in the CLI using `--montage-type`
- Some workflows may include stimulus/state channels; use `--keep-stim` if you need them

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
  You need the BCI2000 Python tools installed and importable
    `pip install BCI2000Tools`

- **PDF report generation fails**
  Ensure `reportlab`, `svglib`, and `Pillow` are installed. The report generator also expects certain plot files to exist in the output folder

- **Montage/channel name mismatches**
  Check `eeg_tools/helper/` location files and `helper/eeg_dict.py`, or override montage selection with `--montage-type`

---

## License

TODO
