# Crop Yield Prediction API

This project provides a machine learning API for predicting crop yield based on environmental and agricultural features. It uses a trained regression model and preprocessing pipeline, deployed with FastAPI.

---

## **Project Structure**

```
fastapiProject/
│
├── crop-yield-prediction.ipynb      # Jupyter notebook for data prep, training, and model export
├── deploy_model.py                  # FastAPI app for serving predictions
├── preprocessor.pkl                 # Saved preprocessing pipeline (generated after training)
├── yield_prediction_model.pkl             # Saved regression model (generated after training)
├── requirements.txt                 # (Optional) List of dependencies
└── README.md                        # Project documentation
```

---

## **Features Used for Prediction**

- `Year` (int)
- `average_rain_fall_mm_per_year` (float)
- `pesticides_tonnes` (float)
- `avg_temp` (float)
- `Area` (str, categorical)
- `Item` (str, categorical)

---

## **Setup Instructions**

### 1. **Clone the Repository**

```sh
git clone <repository-url>
cd fastapiProject
```

### 2. **Create and Activate a Virtual Environment (Recommended)**

```sh
python -m venv .venv
.\.venv\Scripts\activate
```

### 3. **Install Dependencies**

```sh
pip install fastapi uvicorn joblib scikit-learn numpy
```

Or, if you have a `requirements.txt`:

```sh
pip install -r requirements.txt
```

### 4. **Train and Export the Model**

- Open `crop-yield-prediction.ipynb` in Jupyter Notebook or VS Code.
- Run all cells to train the model and export `preprocessor.pkl` and `crop_yield_model.pkl`.

### 5. **Start the API Server**

```sh
uvicorn deploy_model:app --reload --reload-dir .
```

### 6. **Test the API**

- Open your browser at [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs) for the interactive Swagger UI.
- Use the `/predict` endpoint to make predictions.

---

## **Example API Request**

**POST** `/predict`

```json
{
  "Year": 2020,
  "average_rain_fall_mm_per_year": 1200.5,
  "pesticides_tonnes": 50.2,
  "avg_temp": 27.3,
  "Area": "Asia",
  "Item": "Wheat"
}
```

**Response:**
```json
{
  "predicted_yield": 3456.78
}
```

---

## **Notes**

- Ensure `preprocessor.pkl` and `crop_yield_model.pkl` are present in the project directory before starting the API.
- If you retrain the model, re-export these files.

---

## **License**

MIT License (or specify your license)
