import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib

# Load dataset
df = pd.read_csv('eeg_brainwave_data.csv')

# Input (EEG signals) and Output (Brainwave frequencies)
X = df[['Fp1', 'Fp2', 'C3', 'C4']]
y = df[['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma']]

# Scale input features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split for model evaluation
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Model training
base_model = RandomForestRegressor(n_estimators=100, random_state=42)
model = MultiOutputRegressor(base_model)
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
brainwave_names = ['Delta', 'Theta', 'Alpha', 'Beta', 'Gamma']

print("\n=== Evaluation Metrics per Brainwave ===")
for i, wave in enumerate(brainwave_names):
    r2 = r2_score(y_test.iloc[:, i], y_pred[:, i])
    mae = mean_absolute_error(y_test.iloc[:, i], y_pred[:, i])
    rmse = np.sqrt(mean_squared_error(y_test.iloc[:, i], y_pred[:, i]))
    print(f"\nBrainwave: {wave}")
    print(f"R² Score: {r2:.3f}")
    print(f"MAE     : {mae:.3f}")
    print(f"RMSE    : {rmse:.3f}")

joblib.dump(model, 'brainwave_model.pkl')
joblib.dump(scaler, 'scaler.pkl')

# ----------- Real-time-like testing -----------

def predict_brainwaves(raw_eeg_list):
    """Predict and plot brainwaves from new raw EEG inputs."""
    test_scaled = scaler.transform(raw_eeg_list)
    prediction = model.predict(test_scaled)

    for i, sample in enumerate(raw_eeg_list):
        print(f"\n🧠 EEG Input {i+1}: {sample}")
        for name, val in zip(brainwave_names, prediction[i]):
            print(f"  {name}: {val:.3f} Hz")
        
        # Plotting
        plt.figure(figsize=(8, 4))
        plt.bar(brainwave_names, prediction[i], color='skyblue')
        plt.title(f"Predicted Brainwave Frequencies (Sample {i+1})")
        plt.ylabel("Frequency (Hz)")
        plt.ylim(0, max(prediction[i]) + 5)
        plt.grid(True, axis='y', linestyle='--', alpha=0.7)
        plt.show()


# Example real-time test data (you can replace this with continuous input later)
test_eeg_data = [
    [0.6, -0.2, 0.3, 0.1],
    [-0.1, 0.4, -0.3, 0.2],
    [0.3, 0.1, -0.2, -0.1]
]

# Predict and display
predict_brainwaves(test_eeg_data)
