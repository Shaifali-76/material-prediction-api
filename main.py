from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
import os
from contextlib import asynccontextmanager

# 1. Global dictionary to hold the loaded resources
ml_resources = {}

# 2. Lifespan manager: Loads the model once when the server starts
@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        # Load the dictionary from the .sav file
        bundle = joblib.load("ml_models.sav")
        
        # CRITICAL FIX: Use 'material_model' (The Pipeline) 
        # This ensures the input data is SCALED before prediction
        ml_resources["pipeline"] = bundle["material_model"]
        print("Successfully loaded ML Pipeline (Scaler + Model).")
    except Exception as e:
        print(f"Error loading model file: {e}")
        # If the model fails to load, the API shouldn't start
        raise RuntimeError(f"Startup failed: {e}")
    yield
    # Clean up resources on shutdown
    ml_resources.clear()

app = FastAPI(title="Material Model Prediction API", lifespan=lifespan)

# 3. Data Validation Schema
class MaterialInput(BaseModel):
    Tensile_Strength: float
    Weight_Capacity: float
    Biodegradability_Score: float
    Recyclability_Percent: float

@app.get("/")
def home():
    return {
        "status": "online",
        "message": "Material Prediction API is active. Use /predict for inferences."
    }

@app.post("/predict")
def predict(data: MaterialInput):
    # Ensure the pipeline is ready
    if "pipeline" not in ml_resources:
        raise HTTPException(status_code=503, detail="Model is not initialized.")

    try:
        # 4. Convert input to 2D NumPy array
        # Order must match: Tensile, Weight, Biodegradability, Recyclability
        features = np.array([[ 
            data.Tensile_Strength,
            data.Weight_Capacity,
            data.Biodegradability_Score,
            data.Recyclability_Percent
        ]])

        # 5. Inference via Pipeline
        # The pipeline automatically applies the StandardScaler then the Model
        prediction = ml_resources["pipeline"].predict(features)

        return {
            "prediction": int(prediction[0]),
            "status": "success"
        }

    except Exception as e:
        # Generic error handling for unexpected data issues
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    # Use environment variables for port (required for cloud hosting)
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
