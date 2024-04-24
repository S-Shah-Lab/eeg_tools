# Motor Imagery Analysis

This folder contains a collection of scripts, libraries, and files used in the motor imagery analysis pipeline and report generation. Instructions are provided below.

## Table of Contents

- [Requirements](#requirements)
  - [Libraries](#libraries)
  - [Supporting Files](#supporting-files)
- [Motor Imagery](#motor-imagery)
  - [Paradigm](#paradigm)
- [Usage](#usage)
  - [How to Run the Script](#how-to-run)
- [Details](#details)
  - [Pre-processing](#pre-processing)
  - [Epochs & PSDs](#epochs-psds)
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

These files contain channel information for the different montages that we are currently using. Currenlty only three montages are accepted by the scripts: DSI 24 channels, g.tec 32 channels, EGI 128 channels.

- DSI24_location.txt
- GTEC32_location.txt
- EGI128_location.txt

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


***Further reading:***
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

***Notes:***
Sometimes the files is not 418 seconds long. Several reasons for this e.g. the paradigm was stopped at some point before it naturally finished, the block size in BCI2000 was set to a value that resulted in a fractional number of blocks perfectly fitting into the sampling frequency. The latter has being fixed in March 2024.



## Usage
### How to Run the Script


## Details
### Pre-processing
- ***Re-Referencing:***
Only the EGI 128 channels montage is re-referenced to the average mastorids (TP9 and TP10). 
<img src="https://github.com/S-Shah-Lab/motor_imagery/blob/main/assets/dsi24.png" width="750" alt="DSI 24 montage">
<img src="https://github.com/S-Shah-Lab/motor_imagery/blob/main/assets/gtec32.png" width="750" alt="GTEC 32 montage">
<img src="https://github.com/S-Shah-Lab/motor_imagery/blob/main/assets/egi128.png" width="750" alt="EGI 128 montage">

- ***Filtering:***
Bandpass filter [1, 40] Hz

- ***PREP*** (https://pyprep.readthedocs.io/en/latest/index.html):  
These are the thresholds set for PREP, and the analysis flow used for BAD channel identification and interpolation.
<img src="https://github.com/S-Shah-Lab/motor_imagery/blob/main/assets/prep.png" width="750" alt="PREP thresholds">
<img src="https://github.com/S-Shah-Lab/motor_imagery/blob/main/assets/prep_flow.png" width="750" alt="PREP flow in this analysis">

- ***BAD Region Identification:***
BAD regions are identified and marked using MNE directly on the EEG signal by the user. There is no automatic algorithm currenlty in place for this. The first part of the file (in theory 2 s at the very beginning but modified based on block size) is automatically flagged as BAD. 

- ***Spatial Filtering:***
The BCI2000 tool called "SLAP" is used to handle spatial filtering. The idea behind spatial filtering is that it allows to isolate differences among neighboring electrodes by subtracting to each one the average of the electrodes that surround it. Note that not all electrodes are used. The following two slides give more explainations about how this is done:
<img src="https://github.com/S-Shah-Lab/motor_imagery/blob/main/assets/slap.png" width="750" alt="SLAP algorithm">
<img src="https://github.com/S-Shah-Lab/motor_imagery/blob/main/assets/slap_idea.png" width="750" alt="SLAP concept with tool example">

### Epochs and PSDs
The following slides illustrate how Epochs and PSDs are extracted for each trial: 
<img src="https://github.com/S-Shah-Lab/motor_imagery/blob/main/assets/epochs.png" width="750" alt="Information regarding Epochs">
<img src="https://github.com/S-Shah-Lab/motor_imagery/blob/main/assets/psds.png" width="750" alt="Information regarding PSDs">

The following variables can be modified in `motorimagery.py` if changes are needed:
- `nSplit`: defines the number of Epochs in each trial (default = 6). It also defines the length in seconds of each Epoch as it appears in the equation that defines `tmax`.
- `rejectSec`: defines the time to be rejected after each cue (default = 1). This might be slighlty convoluted but the idea is that the integer in the equation is the window time to be used for analysis for each trial. 

### Statistical Tests
Both questions of whether the subject is able to perform motor imagery with the left AND right hand are asked. A p-value is extracted for each hand movement. This is done via bootstrap and permutation tests. The bootstrap test is the main test performed, while the permutation test is manly done as validation of the bootstrap.
While the PSDs are the starting point for both tests, for each frequency band of interest (e.g. alpha band) they are first aggregated over the bins in that frequency, converted to dB units and transformed into another variable called `signed-r²` which is used in the tests. 
More information here: 
<img src="https://github.com/S-Shah-Lab/motor_imagery/blob/main/assets/eta2.png" width="750" alt="PSDs to signed-r2 conversion">
<img src="https://github.com/S-Shah-Lab/motor_imagery/blob/main/assets/stat_test.png" width="750" alt="Statistical Tests">
<img src="https://github.com/S-Shah-Lab/motor_imagery/blob/main/assets/perm.png" width="750" alt="Permutation Test">
<img src="https://github.com/S-Shah-Lab/motor_imagery/blob/main/assets/perm_2.png" width="750" alt="Permutation Test">
<img src="https://github.com/S-Shah-Lab/motor_imagery/blob/main/assets/perm_3.png" width="750" alt="Permutation Test">
<img src="https://github.com/S-Shah-Lab/motor_imagery/blob/main/assets/boot.png" width="750" alt="Bootstrap Test">

### Plots


## PDF Report


## Licence