from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
import os
from contextlib import asynccontextmanager

# Global variable to hold the pipeline
model_bundle = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the FULL pipeline (includes the scaler)
    try:
        bundle = joblib.load("ml_models.sav")
        # Use the pipeline object 'material_model' to ensure data is scaled correctly
        model_bundle["pipeline"] = bundle["material_model"]
        print("Model pipeline loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        raise RuntimeError(f"Could not load model: {e}")
    yield
    model_bundle.clear()

app = FastAPI(title="Material Model Prediction API", lifespan=lifespan)

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
    if "pipeline" not in model_bundle:
        raise HTTPException(status_code=503, detail="Model not loaded")
        
    try:
        # Convert Pydantic model to the format expected by the pipeline
        features = np.array([[ 
            data.Tensile_Strength,
            data.Weight_Capacity,
            data.Biodegradability_Score,
            data.Recyclability_Percent
        ]])

        # The pipeline handles scaling AND prediction automatically
        prediction = model_bundle["pipeline"].predict(features)

        return {
            "prediction": int(prediction[0]),
            "status": "success"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
