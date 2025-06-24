import numpy as np
import tensorflow as tf
import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler
from scipy.signal import butter, filtfilt

# -------- Helper Functions --------
def bandpass_filter(data, lowcut, highcut, fs=256, order=4):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data)

# -------- Load Model and Encoders --------
model = tf.keras.models.load_model('/Users/dharanisham/Developer/Github-Repositories/NeuroBridge/model/neurobridge_model.h5')
brainwave_encoder = joblib.load('/Users/dharanisham/Developer/Github-Repositories/NeuroBridge/model/brainwave_encoder.pkl')
state_encoder = joblib.load('/Users/dharanisham/Developer/Github-Repositories/NeuroBridge/model/state_encoder.pkl')

# -------- Load Test EEG Data --------
df = pd.read_csv('/Users/dharanisham/Developer/Github-Repositories/NeuroBridge/datasets/adhdata.csv')

# Extract channels
test_data = df[['Fp1', 'Fp2', 'C3', 'C4']].values

# Ensure 256 samples (1 second worth of EEG assuming 256 Hz)
time_points = 256
if test_data.shape[0] > time_points:
    test_data = test_data[:time_points]
elif test_data.shape[0] < time_points:
    test_data = np.pad(test_data, ((0, time_points - test_data.shape[0]), (0, 0)), mode='constant')

# Reshape for model
test_data = np.expand_dims(test_data, axis=0)  # (1, 256, 4)
test_data = np.transpose(test_data, (0, 2, 1))  # (1, 4, 256)

# -------- Preprocessing: Filtering and Normalization --------
# Bandpass filter per channel
brainwave_bands = {
    'Delta': (0.5, 4),
    'Theta': (4, 8),
    'Alpha': (8, 12),
    'Beta': (12, 30),
    'Gamma': (30, 100)
}

for i in range(test_data.shape[1]):  # channels
    for band, (lowcut, highcut) in brainwave_bands.items():
        test_data[0, i] = bandpass_filter(test_data[0, i], lowcut, highcut)

# Normalize
scaler = StandardScaler()
test_data = scaler.fit_transform(test_data.reshape(-1, test_data.shape[-1])).reshape(test_data.shape)

# -------- Prediction --------
brainwave_pred, state_pred = model.predict(test_data)

# Get predictions
brainwave_class = np.argmax(brainwave_pred, axis=1)[0]
state_class = np.argmax(state_pred, axis=1)[0]

# Decode
predicted_brainwave = brainwave_encoder.inverse_transform([brainwave_class])[0]
predicted_state = state_encoder.inverse_transform([state_class])[0]

# Print Results
print(f"🧠 Predicted Brainwave: {predicted_brainwave}")
print(f"🧘 Predicted Mental State: {predicted_state}")