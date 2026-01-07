from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
import os
from contextlib import asynccontextmanager

# Use a dictionary or global to store the loaded pipeline
ml_resources = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the model bundle once at startup
    try:
        bundle = joblib.load("ml_models.sav")
        # CRITICAL: Load the full pipeline to include scaling logic
        ml_resources["pipeline"] = bundle["material_model"]
        print("Model pipeline loaded successfully.")
    except Exception as e:
        print(f"Startup Error: {e}")
        raise RuntimeError(f"Could not load model file: {e}")
    yield
    # Clean up resources on shutdown
    ml_resources.clear()

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
    if "pipeline" not in ml_resources:
        raise HTTPException(status_code=503, detail="Model pipeline is not initialized.")

    try:
        # Convert input data to the shape (1, 4)
        features = np.array([[ 
            data.Tensile_Strength,
            data.Weight_Capacity,
            data.Biodegradability_Score,
            data.Recyclability_Percent
        ]])

        # Calling predict on the PIPELINE automatically applies the StandardScaler
        prediction = ml_resources["pipeline"].predict(features)

        return {
            "prediction": int(prediction[0]),
            "status": "success"
        }

    except Exception as e:
        # Log the error here in a real production environment
        raise HTTPException(status_code=500, detail="Prediction failed. Check input formats.")

if __name__ == "__main__":
    import uvicorn
    # Use environment variable for port to support cloud platforms like Heroku/Render
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
