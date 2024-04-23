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

pyprep (PREP)		            -- https://pypi.org/project/pyprep/
BCI2000Tools (ChannelSet, SLAP) -- https://www.bci2000.org/mediawiki/index.php/Programming_Howto:Quickstart_Guide
mne (EEG data objects)          -- https://mne.tools/stable/install/

numpy (arrays)                  -- pip install numpy
matplotlib (plotting)           -- pip install matplotlib
reportlab (pdf generation)      -- pip install reportlab
svglib (.svg images)            -- pip install svglib
PIL (.png images)               -- pip install Pillow

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
### Paradigm


## Usage
### Pre-processing
### Statistical Tests
### Plots


## PDF Report


## Licence