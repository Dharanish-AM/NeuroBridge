from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import numpy as np
from typing import Dict


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


model = joblib.load("brainwave_model.pkl")
scaler = joblib.load("scaler.pkl")

brainwave_names = ["Delta", "Theta", "Alpha", "Beta", "Gamma"]



class EEGInput(BaseModel):
    Fp1: float
    Fp2: float
    C3: float
    C4: float


@app.post("/predict", response_model=Dict[str, float])
async def predict_brainwaves(eeg: EEGInput):
    try:
        
        input_array = np.array([[eeg.Fp1, eeg.Fp2, eeg.C3, eeg.C4]])
        input_scaled = scaler.transform(input_array)
        prediction = model.predict(input_scaled)[0]

        
        result = {name: round(val, 3) for name, val in zip(brainwave_names, prediction)}
        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
