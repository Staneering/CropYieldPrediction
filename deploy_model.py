from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI()

# Load preprocessor and model
preprocessor = joblib.load('preprocessor.pkl')
model = joblib.load('yield_prediction_model.pkl')

class CropYieldInput(BaseModel):
    Year: int
    average_rain_fall_mm_per_year: float
    pesticides_tonnes: float
    avg_temp: float
    Area: str
    Item: str

@app.get("/")
def read_root():
    return {"message": "Welcome to the Crop Yield Prediction API. Use /predict to get predictions."}

@app.post("/predict")
def predict_yield(data: CropYieldInput):
    try:
        # Arrange input in the correct order
        input_arr = np.array([[data.Year, data.average_rain_fall_mm_per_year, data.pesticides_tonnes,
                               data.avg_temp, data.Area, data.Item]])
        # Preprocess input
        input_processed = preprocessor.transform(input_arr)
        # Predict
        prediction = model.predict(input_processed)
        return {"predicted_yield": float(prediction[0])}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))