import numpy as np
import pandas as pd
from physfunc.data import JSMFPCAData, SubjectCurves
import os

PARENT_FOLDER = os.path.dirname(os.path.abspath(__file__))

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

PROCESSED_DATA = os.path.join(PARENT_FOLDER, 'data')
RESULTS = os.path.join(PARENT_FOLDER, 'data/results')
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

SCALING_LAMBDAS = {
    'coher_range': (np.log10(10), np.log10(30)),
    'curv_range': (np.log10(10), np.log10(30)),
    'monoton_range': (np.log10(5), np.log10(100))
}
SCALING_ALPHAS = {
    'coher_range': (np.log10(5), np.log10(2000)),
    'curv_range': (np.log10(5), np.log10(2000)),
    'monoton_range': (np.log10(5), np.log10(2000))
}
MORPH_FEATURES = {
    'coherence': (np.log10(10), np.log10(30)),
    'monotonicity': (np.log10(5), np.log10(100)),
    'dynamic_range': (np.log10(5), np.log10(100)),
    'auc': (np.log10(5), np.log10(100)),
    'centroid': (np.log10(5), np.log10(100)),
    'entropy': (np.log10(5), np.log10(100)),
}
HRV_COLS = [
    'meanNN', 'SDNN', 'RMSSD', 'VLF', 'min', 'max',
    'LF', 'HF', 'LFHF_ratio', 'alpha1',
    'alpha2', 'lambda_25s', 'lambda_625s'
]
LMD25s = ['Non-Gaussian index 25s']
LMDS = ['Non-Gaussian index 25s', 'Non-Gaussian index 625s']
CLINICAL_COLS = ['age', 'NYHA', 'sex']
MEDICATION_COLS = ['ACE', 'beta', 'isch', 'Sp', 'LD']

pub_names = {
    'meanNN': 'meanNN',
    'SDNN': 'SDNN',
    'RMSSD': 'RMSSD',
    'VLF': 'VLF',
    'min': r'$NN_{min}$',
    'max': r'$NN_{max}$',
    'LF': 'LF',
    'HF': 'HF',
    'LFHF_ratio': 'LF/HF ratio',
    'alpha1': r'$\alpha_1$',
    'alpha2': r'$\alpha_2$',
    'lambda_25s': r'$\lambda^2_{25s}$',
    'lambda_625s': r'$\lambda^2_{625s}$',
    'coherence': r'$E^{\nabla}_{\lambda}$',
    'monotonicity': r'$Mw_{\lambda}$',
    'curvature': r'$\kappa_{\lambda}$',
    'coherence_alpha': r'$E^{\nabla}_{\alpha}$',
    'monotonicity_alpha': r'$Mw_{\alpha}$',
    'curvature_alpha': r'$\kappa_{\alpha}$',
    'age': 'Age',
    'NYHA': 'NYHA',
    'sex': 'Sex',
    'ACE': 'ACE',
    'beta': 'beta',
    'isch': 'isch',
    'Sp': 'Sp',
    'LD': 'LD',
}


def coherence(signal, mask):
    val = signal.loc[:, mask]
    num = val.diff(axis=1).pow(2).sum(axis=1)
    den = val.sub(val.mean(axis=1), axis=0).pow(2).sum(axis=1)
    return -np.log10(num / den)


def monotonicity_index(signal, mask, epsilon=0.01):
    val = signal.loc[:, mask]
    delta = val.diff(axis=1)
    total_variation = delta.abs().sum(axis=1)
    upward_variation = delta.clip(lower=0).sum(axis=1)

    Mw = (upward_variation / total_variation).fillna(0.0)

    return 1 - Mw


def compute_mean_abs_curvature(curve, mask):
    return curve.loc[:, mask].diff(axis=1).diff(axis=1).abs().mean(axis=1)


def compute_morphological_features(df_curves, scales):
    scales = np.asarray(scales)

    def mask(name):
        low, high = MORPH_FEATURES[name]
        return (scales >= low) & (scales <= high)

    feats = pd.DataFrame(index=df_curves.index)

    feats['coherence'] = coherence(df_curves, mask('coherence'))
    feats['monotonicity'] = monotonicity_index(df_curves, mask('monotonicity'))
    m = mask('dynamic_range')
    vals = df_curves.loc[:, m]
    feats['dynamic_range'] = vals.max(axis=1) - vals.min(axis=1)

    m = mask('auc')
    vals = df_curves.loc[:, m]
    feats['auc'] = np.trapz(vals.values, scales[m], axis=1)

    m = mask('centroid')
    vals = df_curves.loc[:, m]
    x = scales[m]

    w = vals.sub(vals.min(axis=1), axis=0)
    den = w.sum(axis=1).replace(0, np.nan)

    feats['centroid'] = w.mul(x, axis=1).sum(axis=1) / den

    m = mask('entropy')
    vals = df_curves.loc[:, m]
    w = vals.sub(vals.min(axis=1), axis=0)

    p = w.div(w.sum(axis=1).replace(0, np.nan), axis=0)
    feats['entropy'] = -(p * np.log(p + 1e-12)).sum(axis=1)

    return feats


def get_curves(window, hour, feat, dataset='986'):
    df_v = pd.read_csv(
        f"{PROCESSED_DATA}/{dataset}/covariates_{window}h.csv",
        index_col=0,
    )
    df_v = df_v[df_v.age >= 18]

    df_c = pd.read_csv(
            (f"{PROCESSED_DATA}/{dataset}/{feat}"
             f"_{window}h_{hour}.csv"),
            index_col=0,
        ).dropna(axis=1)

    idx = df_c.index.intersection(df_v.index)
    df_c.columns = df_c.columns.map(float)
    return df_c.loc[idx]


def get_features(window, dataset='986'):
    df_v = pd.read_csv(
        f"{PROCESSED_DATA}/{dataset}/covariates_{window}h.csv",
        index_col=0,
    )
    df_v = df_v[df_v.age >= 18]

    all_feats = pd.DataFrame()

    for hour in range(24):
        df_c = pd.read_csv(
            (f"{PROCESSED_DATA}/{dataset}/lmds"
             f"_{window}h_{hour}.csv"),
            index_col=0,
        ).dropna(axis=1)
        scales = np.array(df_c.columns.astype(float))
        idx = df_c.index.intersection(df_v.index)
        df_c = df_c.loc[idx]
        df_v = df_v.loc[idx]
        feats = compute_morphological_features(df_c, scales)
        feats['hour'] = hour
        all_feats = pd.concat([all_feats, feats])
    return pd.merge(
        df_v.reset_index(), all_feats.reset_index(), on=['Code', 'hour']
        ).set_index('Code')


def load_jsmfpca_dataset(
    window_size, feat, dataset_name, valid_hours=range(24)
):
    print(
        f"Loading data for {dataset_name} - {feat} (Window: {window_size}h)..."
    )

    subject_data = {}
    scales = None

    for hour in valid_hours:
        try:
            df_curves = get_curves(window_size, hour, feat, dataset_name)
        except Exception as e:
            print(f"  Warning: Could not load hour {hour} ({e})")
            continue

        if df_curves is None or df_curves.empty:
            continue

        if scales is None:
            scales = df_curves.columns.astype(float).values

        for subj_id, row in df_curves.iterrows():
            curve = row[df_curves.columns].values.astype(float)

            if np.isnan(curve).all():
                continue

            if subj_id not in subject_data:
                subject_data[subj_id] = {
                    'hours': [],
                    'curves': []
                }

            subject_data[subj_id]['hours'].append(hour)
            subject_data[subj_id]['curves'].append(curve)

    subjects = []

    for subj_id, data in subject_data.items():
        if len(data['hours']) < 4:
            continue

        subjects.append(
            SubjectCurves(
                subject_id=str(subj_id),
                hours=np.array(data['hours'], dtype=int),
                curves=np.vstack(data['curves'])
            )
        )

    if not subjects:
        raise ValueError("No valid subjects found after aggregation.")

    print(f"  -> Successfully loaded {len(subjects)} subjects.")

    dataset = JSMFPCAData(subjects=subjects, scales=scales)
    return dataset
