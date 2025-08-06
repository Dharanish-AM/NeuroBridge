import numpy as np
import pandas as pd
from scipy.signal import welch
from scipy.integrate import trapezoid


df = pd.read_csv("adhdata.csv", usecols=['Fp1', 'Fp2', 'C3', 'C4'])

fs = 256  
window_size = 256  


bands = {
    "Delta": (0.5, 4),
    "Theta": (4, 8),
    "Alpha": (8, 13),
    "Beta": (13, 30),
    "Gamma": (30, 100)
}

def extract_bandpower(signal, fs, band):
    fmin, fmax = band
    freqs, psd = welch(signal, fs=fs, nperseg=min(len(signal), fs))
    idx_band = (freqs >= fmin) & (freqs <= fmax)
    return trapezoid(psd[idx_band], freqs[idx_band])

rows = []


for start in range(0, len(df) - window_size + 1, window_size):
    window = df.iloc[start:start + window_size]

    
    inputs = window.mean().to_dict()

    
    bandpowers = {}
    for band_name, freq_range in bands.items():
        powers = []
        for ch in ['Fp1', 'Fp2', 'C3', 'C4']:
            signal = window[ch].values
            bp = extract_bandpower(signal, fs, freq_range)
            powers.append(bp)
        bandpowers[band_name] = np.mean(powers)

    
    combined = {**inputs, **bandpowers}
    rows.append(combined)


processed_df = pd.DataFrame(rows)


processed_df.to_csv("eeg_brainwave_dataset.csv", index=False)
print("✅ New dataset with brainwave values saved as 'eeg_brainwave_dataset.csv'")
