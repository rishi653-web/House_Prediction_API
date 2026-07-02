import joblib
import pandas as pd
import io
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

app = FastAPI()

# Relative paths ensure no unicode escape errors
model = joblib.load("house_model.joblib")
features = joblib.load("house_feature.joblib")

# Input schema
class HouseFeatures(BaseModel):
    MedInc: float = Field(default=3.5, description="Median income in block")
    HouseAge: float = Field(default=28.0, description="Median house age in block")
    AveRooms: float = Field(default=5.2, description="Average number of rooms")
    AveBedrms: float = Field(default=1.0, description="Average number of bedrooms")
    Population: float = Field(default=1425.0, description="Block population")
    AveOccup: float = Field(default=3.0, description="Average house occupancy")
    Latitude: float = Field(default=35.6, ge=32, le=42, description="Latitude")
    Longitude: float = Field(default=-119.5, ge=-124, le=-114, description="Longitude")

# Home route
@app.get("/")
def home():
    return {
        "message": "California house prediction API",
        "status": "running",
        "endpoint": "Send a POST request to /predict"
    }

# Health check route
@app.get("/health")
def health():
    return {
        "status": "running",
        "model": "RandomForestRegressor",
        "features": features,
        "avg_error": "$39,000"
    }

# Prediction route
@app.post("/predict")
def predict(house: HouseFeatures):
    try:
        input_data = pd.DataFrame([{
            "MedInc": house.MedInc,
            "HouseAge": house.HouseAge,
            "AveRooms": house.AveRooms,
            "AveBedrms": house.AveBedrms, 
            "Population": house.Population,
            "AveOccup": house.AveOccup,
            "Latitude": house.Latitude,
            "Longitude": house.Longitude
        }])

        predicted = model.predict(input_data)[0]
        price_usd = predicted * 100000

        return {
            "predicted_price": f"${price_usd:,.0f}",
            "predicted_price_short": f"${predicted:.2f} hundred thousands",
            "confidence_range": f"${price_usd - 39000:,.0f} to ${price_usd + 39000:,.0f}"
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )

# CSV File Prediction route
@app.post("/predict-file")
async def predict_file(file: UploadFile = File(...)): # FIX: Capitalized UploadFile

    if not file.filename.endswith(".csv"):
        raise HTTPException(
            status_code=400,
            detail="Please upload a CSV file only"
        )

    contents = await file.read()
    
    try:
        # FIX: Used pd.read_csv instead of pd.DataFrame
        df = pd.read_csv(io.BytesIO(contents))
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Failed to parse CSV file: {str(e)}"
        )

    # FIX: Fixed typo 'MenInc' to 'MedInc'
    required_columns = [
        'MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup', 'Latitude', 'Longitude'
    ]

    missing_columns = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing_columns:
        raise HTTPException(
            status_code=400,
            detail=f'These columns are missing from your file: {missing_columns}'
        )

    if len(df) == 0:
        raise HTTPException(
            status_code=400,
            detail='The uploaded file has no data rows'
        )

    try:
        # Generate model arrays
        predictions = model.predict(df[required_columns])

        # FIX: Converted raw target scale to actual USD value safely
        df["predicted_raw"] = predictions
        df["predicted_price_usd"] = df["predicted_raw"] * 100000
        df["predicted_price_formatted"] = df["predicted_price_usd"].apply(lambda x: f"${x:,.0f}")

        # Convert back to string CSV format
        output = df.to_csv(index=False)

        # FIX: Fixed io.StringID typo to io.StringIO
        return StreamingResponse(
            io.StringIO(output),
            media_type="text/csv",
            headers={
                "Content-Disposition": "attachment; filename=predictions.csv"
            }
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction processing failed: {str(e)}"
        )