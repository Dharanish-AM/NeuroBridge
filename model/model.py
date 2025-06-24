import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib
from scipy.signal import butter, filtfilt
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.callbacks import EarlyStopping

# Bandpass filter function to isolate the desired frequency band
def bandpass_filter(data, lowcut, highcut, fs, order=4):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data)

# Load the dataset (make sure the path is correct)
df = pd.read_csv('/Users/dharanisham/Developer/Github-Repositories/NeuroBridge/datasets/adhdata.csv')
df = df.head(1000)  # Limit to first 1000 rows for faster processing
print("Dataset loaded successfully. Shape:", df.shape)

# Extract only the columns Fp1, Fp2, C3, C4
X = np.expand_dims(np.array(df[['Fp1', 'Fp2', 'C3', 'C4']]), axis=-1)

# Reshaping data to create 256 time points (repeat data to simulate time-series data)
X = np.repeat(X, 256, axis=-1)

# -------- Preprocessing: Apply Bandpass Filter --------
fs = 256  # Sampling frequency
brainwave_bands = {
    'Delta': (0.5, 4),    # Delta: 0.5-4 Hz
    'Theta': (4, 8),      # Theta: 4-8 Hz
    'Alpha': (8, 12),     # Alpha: 8-12 Hz
    'Beta': (12, 30),     # Beta: 12-30 Hz
    'Gamma': (30, 100)    # Gamma: 30-100 Hz
}

# Apply bandpass filter to each channel
filtered_eeg = np.zeros_like(X)
for i in range(X.shape[0]):  # Iterate over samples
    for j in range(X.shape[1]):  # Iterate over channels
        for band, (lowcut, highcut) in brainwave_bands.items():
            filtered_eeg[i][j] = bandpass_filter(X[i][j], lowcut, highcut, fs)
print("Bandpass filtering completed. Shape:", filtered_eeg.shape)

# -------- Normalize the EEG Data --------
scaler = StandardScaler()
filtered_eeg = scaler.fit_transform(filtered_eeg.reshape(-1, filtered_eeg.shape[-1])).reshape(filtered_eeg.shape)
print("Normalization completed. Sample:", filtered_eeg[0, 0, :5])

# -------- Simulate the brainwave and state labels --------
brainwave_labels = ['Alpha', 'Beta', 'Gamma', 'Theta', 'Delta']
state_labels = ['Relaxed', 'Focused', 'Stressed', 'Drowsy']

df['Brainwave_Label'] = np.random.choice(brainwave_labels, size=len(df))
df['State_Label'] = np.random.choice(state_labels, size=len(df))

# Encode the labels using LabelEncoder
brainwave_encoder = LabelEncoder()
state_encoder = LabelEncoder()

y_brainwave = brainwave_encoder.fit_transform(df['Brainwave_Label'])
y_state = state_encoder.fit_transform(df['State_Label'])
print("Labels encoded. Brainwave classes:", brainwave_encoder.classes_, ", State classes:", state_encoder.classes_)

# Train-Test Split
X_train, X_test, y_brainwave_train, y_brainwave_test, y_state_train, y_state_test = train_test_split(
    filtered_eeg, y_brainwave, y_state, test_size=0.2, random_state=42
)

# Build the Model
input_layer = layers.Input(shape=(4, 256))  # 4 channels and 256 time points

x = layers.Conv1D(32, kernel_size=5, activation='relu', padding='same')(input_layer)
x = layers.MaxPooling1D(pool_size=2)(x)
x = layers.Conv1D(64, kernel_size=5, activation='relu', padding='same')(x)
x = layers.MaxPooling1D(pool_size=2)(x)
x = layers.Flatten()(x)
x = layers.Dense(128, activation='relu')(x)
x = layers.Dropout(0.3)(x)  # Added Dropout for regularization

# Outputs for both brainwave and state
brainwave_output = layers.Dense(5, activation='softmax', name="brainwave_output")(x)  # 5 classes (Alpha, Beta, etc.)
state_output = layers.Dense(4, activation='softmax', name="state_output")(x)  # 4 classes (Relaxed, Focused, etc.)

# Define the model
model = models.Model(inputs=input_layer, outputs=[brainwave_output, state_output])

# Compile the model
model.compile(optimizer='adam',
              loss={'brainwave_output': 'sparse_categorical_crossentropy', 'state_output': 'sparse_categorical_crossentropy'},
              metrics={'brainwave_output': 'accuracy', 'state_output': 'accuracy'})

# Show the model summary
model.summary()

# -------- Callbacks for Model Training --------
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

print("Starting model training...")

# Train the model
history = model.fit(
    X_train, {"brainwave_output": y_brainwave_train, "state_output": y_state_train},
    validation_data=(X_test, {"brainwave_output": y_brainwave_test, "state_output": y_state_test}),
    epochs=30,  # Number of epochs to train
    batch_size=32,  # Batch size
    callbacks=[early_stopping]  # Adding EarlyStopping to avoid overfitting
)

print("Model training completed.")

# Save the trained model after training completes
model.save('/Users/dharanisham/Developer/Github-Repositories/NeuroBridge/model/neurobridge_model.h5')

# Evaluate the model
results = model.evaluate(X_test, {"brainwave_output": y_brainwave_test, "state_output": y_state_test})
print("Test Loss and Accuracy:", results)

# Save the trained model and label encoders
joblib.dump(brainwave_encoder, '/Users/dharanisham/Developer/Github-Repositories/NeuroBridge/model/brainwave_encoder.pkl')
joblib.dump(state_encoder, '/Users/dharanisham/Developer/Github-Repositories/NeuroBridge/model/state_encoder.pkl')

print("Model and encoders saved successfully!")