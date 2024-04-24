# Motor Imagery Analysis

This folder contains a collection of scripts, libraries, and files used in the motor imagery analysis pipeline and report generation. Instructions are provided below.

## Table of Contents

- [Requirements](#requirements)
  - [Libraries](#libraries)
  - [Supporting Files](#supporting-files)
- [Motor Imagery](#motor-imagery)
  - [Paradigm](#paradigm)
- [Usage](#usage)
  - [Pre-processing](#pre-processing)
  - [Statistical Tests](#statistical-tests)
  - [Plots](#plots)
- [PDF Report](#pdf-report)
- [Licence](#license)

## Requirements
### Libraries
The following is a list of libraries needed and how to install them and/or where to find info regarding the installation:

- `numpy` (arrays)              -- pip install numpy
- `matplotlib` (plotting)       -- pip install matplotlib
- `reportlab` (pdf generation)  -- pip install reportlab
- `svglib` (.svg images)        -- pip install svglib
- `PIL` (.png images)           -- pip install Pillow

- `pyprep` (PREP)		        -- https://pypi.org/project/pyprep/
- `BCI2000Tools` (e.g. SLAP)    -- https://www.bci2000.org/mediawiki/index.php/Programming_Howto:Quickstart_Guide
- `mne` (EEG data objects)      -- https://mne.tools/stable/install/

### Supporting Files
The following files are needed to run the scripts correctly. They should be placed in the same folder with the other scripts.

These files contain channel information for the different montages that we are currently using. 

- DSI24_location.txt
- EGI128_location.txt
- GTEC32_location.txt

This Python file contains important dictionaries such as channel names conversion, symmetric channels, channels in each brain region, etc.

- eeg_dict.py

These three libraries contain tools for EEG data analysis, plotting, and statistical tests, as well as the report generation.

- PlotStyle.py
- PdfReport.py
- MotorImageryTools.py

This is the main script which runs the analysis and the report generation.

- motorimagery.py

## Motor Imagery

"Motor imagery is a cognitive process in which a subject imagines that they perform a movement without actually performing the movement and without even tensing the muscles. It is a dynamic state during which the representation of a specific motor action is internally activated without any motor output. In other words motor imagery requires the conscious activation of brain regions that are also involved in movement preparation and execution, accompanied by a voluntary inhibition of the actual movement." [https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2797860/]


Further reading: 
- https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2797860/
- 

### Paradigm
A file is 418 s long and it contains 8 blocks in which 4 trials are repeated. There are 2 s of silence at the beginning of the file. 
Each block contains:
- 4 trials in the following sequence: Left, Rest-after-Left, Right, Rest-after-Right

Each trial contains: 
- 1 cue of 3 s in which a voice instructs the participant to "Keep opening and closing your left/right hand", or "Stop and relax"
- 1 silence of 10 s in which the participant performs the given instruction

2 + ((3 + 10) x 4) x 8 = 418 seconds

Notes:
Sometimes the files is not 418 seconds long. Several reasons for this e.g. the paradigm was stopped at some point before it naturally finished, the block size in BCI2000 was set to a value that resulted in a fractional number of blocks perfectly fitting into the sampling frequency. The latter has being fixed in March 2024.

## Usage
### Pre-processing
### Statistical Tests
### Plots


## PDF Report


## Licence