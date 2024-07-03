import numpy as np
from BCI2000Tools.FileReader import bcistream


def import_file_dat(path=None, file_name=None, verbose=True):
    """
    Imports and processes EEG data from a .dat file.

    Args:
        path (str, optional): The file path leading to the .dat file
        file_name (str): The name of the .dat file to be imported.
        verbose (bool, optional): If True, prints detailed information

    Returns:
        tuple: A tuple containing the processed signal data, state information, sampling rate, channel names, block size, and montage type
    """
    # Load the .dat file
    b = bcistream(path + file_name)
    signal, states = b.decode()
    signal = np.array(signal)

    # Retrieve additional parameters from the file
    fs = b.samplingrate()  # Sampling rate
    ch_names = b.params["ChannelNames"]  # Channel names
    blockSize = b.params.SampleBlockSize  # Block size used in data acquisition

    print(b.params["StorageTime"])

    return signal, states, fs, ch_names, blockSize


# path = "/mnt/c/Users/scana/Desktop/RI149/motor_imagery/"
# file_name = "20201212_ri149-MotorImagery-S001R01.dat"

path = "/mnt/c/Users/scana/Desktop/RI149/motor_imagery/"
file_name = "sub-BIpeds149_ses-14_task-MotorImagery_run-01.dat"

signal, states, fs, ch_names, blockSize = import_file_dat(
    path=path, file_name=file_name
)

print(fs, ch_names)
