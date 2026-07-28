import os
import numpy as np
from matplotlib import rc
import scipy.signal as signal
from hrv_utils.dma import create_scales, dma
import sys

sys.path.append(
    os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        '0_Commons'
    )
)

rc('font', **{'family': 'serif', 'serif': ['Arial']})
rc('text', usetex=True)


def get_scale_mask(scales, ranges):
    return (scales > np.log10(ranges[0])) & (scales < np.log10(ranges[1]))


def get_time(df, conditions):
    new_df = df.copy()
    for feature in conditions:
        a, b = conditions[feature]
        new_df = new_df[(new_df[feature] >= a) & (new_df[feature] <= b)]
    return new_df.drop(columns=list(conditions.keys()))


def get_mask(rri, ts, A, B):
    if A < B:
        mask = (ts >= A) & (ts < B)
    else:
        mask = (ts >= A) | (ts < B)

    padded = np.concatenate([[False], mask, [False]])
    diff = np.diff(padded.astype(int))
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0] - 1
    lengths = ends - starts + 1

    if len(lengths) == 0:
        return np.array([])

    longest_idx = np.argmax(lengths)
    s, e = starts[longest_idx], ends[longest_idx]

    return rri[s:e+1]


def compute_alphas(rri, order=0, resampled=False, top_limit=None, scales=None):
    if (len(rri) <= 10):
        return np.nan, np.nan, np.nan

    if (resampled):
        limit1 = np.log10(11)
        limit2 = np.log10(1024)
    else:
        limit1 = np.log10(40)
        limit2 = np.log10(4000)

    if (top_limit is not None):
        limit2 = top_limit

    if (scales is None):
        scales = create_scales(3, 2 * top_limit).astype(np.int64)
    coefs = np.log10(dma(rri, scales, order=order))
    scales = np.log10(scales)

    fit1 = np.polyfit(
        scales[(scales <= limit1)],
        coefs[(scales <= limit1)],
        1
    )
    alpha1 = fit1[0]
    fit2 = np.polyfit(
        scales[(scales >= limit1) & (scales <= limit2)],
        coefs[(scales >= limit1) & (scales <= limit2)],
        1
    )
    alpha2 = fit2[0]

    return alpha1, alpha2, coefs


def spectral_indices(rri, fs=4, nfft=4096):
    if len(rri) == 0:
        return np.nan, np.nan, np.nan, np.nan, np.nan
    try:
        freq, psd = signal.welch(
                        x=rri,
                        fs=fs,
                        window='hann',
                        nperseg=min(256, len(rri)),
                        nfft=nfft
                    )

        vlf_indexes = np.logical_and(freq >= 0.003, freq < 0.04)
        lf_indexes = np.logical_and(freq >= 0.04, freq < 0.15)
        hf_indexes = np.logical_and(freq >= 0.15, freq < 0.40)

        lf = np.trapz(y=psd[lf_indexes], x=freq[lf_indexes])
        hf = np.trapz(y=psd[hf_indexes], x=freq[hf_indexes])

        vlf = np.trapz(y=psd[vlf_indexes], x=freq[vlf_indexes])
        total_power = vlf + lf + hf
        if (hf != 0):
            lfhf = lf/hf
        else:
            lfhf = None

        return vlf, lf, hf, lfhf, total_power
    except Exception as e:
        print(e)
        return np.nan, np.nan, np.nan, np.nan, np.nan
