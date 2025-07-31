import numpy as np
import pandas as pd
from scipy.signal import welch
from scipy.integrate import trapezoid

# Load only EEG channels from your dataset
df = pd.read_csv("adhdata.csv", usecols=['Fp1', 'Fp2', 'C3', 'C4'])

fs = 256  # Sampling frequency (Hz)
window_size = 256  # 1-second window (number of samples)

# Brainwave frequency bands
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

# Slide through the data in windows (non-overlapping)
for start in range(0, len(df) - window_size + 1, window_size):
    window = df.iloc[start:start + window_size]

    # Calculate average raw EEG signals per channel in this window
    inputs = window.mean().to_dict()

    # Calculate average band powers across all channels for each brainwave band
    bandpowers = {}
    for band_name, freq_range in bands.items():
        powers = []
        for ch in ['Fp1', 'Fp2', 'C3', 'C4']:
            signal = window[ch].values
            bp = extract_bandpower(signal, fs, freq_range)
            powers.append(bp)
        bandpowers[band_name] = np.mean(powers)

    # Combine inputs and band powers for this window
    combined = {**inputs, **bandpowers}
    rows.append(combined)

# Create DataFrame from processed data
processed_df = pd.DataFrame(rows)

# Save to CSV
processed_df.to_csv("eeg_brainwave_dataset.csv", index=False)
print("✅ New dataset with brainwave values saved as 'eeg_brainwave_dataset.csv'")
