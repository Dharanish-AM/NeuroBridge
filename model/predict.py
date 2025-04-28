import numpy as np
import tensorflow as tf
import joblib
import pandas as pd

# Load the trained model
model = tf.keras.models.load_model('/Users/dharanisham/Developer/Github-Repositories/NeuroBridge/model/neurobridge_model.h5')

# Load the label encoders
brainwave_encoder = joblib.load('/Users/dharanisham/Developer/Github-Repositories/NeuroBridge/model/brainwave_encoder.pkl')
state_encoder = joblib.load('/Users/dharanisham/Developer/Github-Repositories/NeuroBridge/model/state_encoder.pkl')

# -------- Test Data Example --------
# 4 channels (Fp1, Fp2, C3, C4) × 256 time points
# Create random data for testing (replace with real EEG later)
test_eeg = np.random.randn(4, 256)

# Expand dims because model expects batch dimension
test_eeg = np.expand_dims(test_eeg, axis=0)  # Shape becomes (1, 4, 256)

# -------- Make Prediction --------
brainwave_pred, state_pred = model.predict(test_eeg)

# Get the class with the highest probability
brainwave_class = np.argmax(brainwave_pred, axis=1)[0]
state_class = np.argmax(state_pred, axis=1)[0]

# Decode the labels
predicted_brainwave = brainwave_encoder.inverse_transform([brainwave_class])[0]
predicted_state = state_encoder.inverse_transform([state_class])[0]

# -------- Display Brainwave Percentages --------
brainwave_probabilities = brainwave_pred[0]  # Probabilities for each brainwave type

# Map brainwave types
brainwave_types = ['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma']

# Print the brainwave percentages
print(f"🧠 Predicted Brainwave: {predicted_brainwave}")
print(f"🧘 Predicted Mental State: {predicted_state}")