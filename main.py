from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
import os

app = FastAPI(title="Material Model Prediction API")

# Load the entire Pipeline at startup
try:
    # The file contains a dictionary where 'material_model' is the full Pipeline
    bundle = joblib.load("ml_models.sav")
    material_pipeline = bundle["material_model"] 
except Exception as e:
    raise RuntimeError(f"Could not load model: {e}")

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
        # Prepare features in the exact order the model expects
        features = np.array([[ 
            data.Tensile_Strength,
            data.Weight_Capacity,
            data.Biodegradability_Score,
            data.Recyclability_Percent
        ]])

        # Calling predict on the pipeline automatically applies the StandardScaler
        prediction = material_pipeline.predict(features)

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
