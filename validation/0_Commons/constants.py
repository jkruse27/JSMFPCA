import os
import numpy as np

FS = 2
RESAMPLE = 100
DMA_RESAMPLE = 100
SEED = 42
order = 4
N_SPLITS = 5
N_REPEATS = 5
USE_RESAMPLED = True
INTERP = True
DPI = 600
WINDOWS = [1, 2, 4, 6]

cond_24h = {
    'time': (0, 24*60*60),
    'duration': (5*60*60, 25*60*60)
}
cond_day = {
    'time': (10*60*60, 16*60*60),
    'duration': (0, 3*60*60)
}
cond_night = {
    'time': (0, 5*60*60),
    'duration': (0, 3*60*60)
}

SCALE_RANGES = {
    'short': (np.log10(10.0), np.log10(30.0)),
    'long':  (np.log10(100.0), np.log10(1000.0)),
}

FUNC = np.max

PARENT_FOLDER = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PATH_BASE = {
    '986': os.path.join(PARENT_FOLDER, 'data/CHF_901'),
    '108': os.path.join(PARENT_FOLDER, 'data/CHF_108'),
    'Healthy': os.path.join(PARENT_FOLDER, 'data/Healthy'),
}

PATH_HRV = {
    '986': [
        os.path.join(PARENT_FOLDER, 'data/CHF_901/SV_24h'),
        os.path.join(PARENT_FOLDER, 'data/CHF_901/NS_24h')
    ],
    '108': [
        os.path.join(PARENT_FOLDER, 'data/CHF_108/SV_24h'),
        os.path.join(PARENT_FOLDER, 'data/CHF_108/NS_24h')
    ],
    'Healthy': [os.path.join(PARENT_FOLDER, 'data/Healthy/HRV')]
}

PROCESSED_DATA = os.path.join(PARENT_FOLDER, 'data/')
CLEAN_DATA = os.path.join(PARENT_FOLDER, 'data/clean_hrv')
CLEAN_DATA_RESAMPLED = os.path.join(PARENT_FOLDER, 'data/clean_hrv_resampled')
RESULTS = os.path.join(PARENT_FOLDER, 'data/results')
