from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np

# Initialize FastAPI app
app = FastAPI(title="Material Model Prediction API")

# Load the model from the .sav file
try:
    # Use joblib as it is generally more efficient for sklearn models
    model = joblib.load('ml_models.sav')
except Exception as e:
    raise RuntimeError(f"Could not load model: {e}")

# Define the input schema based on model feature names
class MaterialInput(BaseModel):
    Tensile_Strength: float
    Weight_Capacity: float
    Biodegradability_Score: float
    Recyclability_Percent: float

@app.get("/")
def home():
    return {"message": "Material Model API is running. Use /predict for inferences."}

@app.post("/predict")
def predict(data: MaterialInput):
    try:
        # Convert input data to the format expected by the model (2D array)
        features = np.array([[
            data.Tensile_Strength,
            data.Weight_Capacity,
            data.Biodegradability_Score,
            data.Recyclability_Percent
        ]])
        
        # The model is a pipeline, so it handles scaling automatically
        prediction = model.predict(features)
        
        # Convert prediction to standard Python type for JSON response
        return {
            "prediction": int(prediction[0]),
            "status": "success"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

import os

if __name__ == "__main__":
    import uvicorn
    # This ensures it uses the port assigned by the cloud provider
    port = int(os.environ.get("PORT", 8000)) 
    uvicorn.run(app, host="0.0.0.0", port=port)
