import os
import json
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional

from nlp_extractor import SymptomExtractor
from routing_engine import RoutingEngine, UnknownLocationError
from triage_engine import triage

app = FastAPI(
    title="Sanjeevani AI - ML & NLP Microservice",
    description="Disease prediction microservice using Decision Trees and NLP extraction",
    version="1.0.0"
)

# Global variables for model, its training feature order, the extractor, and the map
model = None
feature_names = []
extractor = None
router = None

# Request Schema
class SymptomRequest(BaseModel):
    user_text: str
    location: Optional[str] = None  # A location id from GET /locations
    duration: Optional[str] = None  # The form's answer, e.g. "More than a week"

@app.on_event("startup")
def load_artifacts():
    global model, feature_names, extractor, router

    # Load the Delhi NCR map for hospital routing
    graph_path = os.path.join("data", "delhi_ncr_graph.json")
    if os.path.exists(graph_path):
        router = RoutingEngine.from_file(graph_path)
        print(f"Loaded map with {len(router.locations)} locations.")
    else:
        print("Warning: delhi_ncr_graph.json not found! Hospital routing is unavailable.")

    # Load the exact feature order the model was trained with
    metadata_path = os.path.join("models", "sanjeevani_model_metadata.json")
    if os.path.exists(metadata_path):
        with open(metadata_path, "r", encoding="utf-8") as f:
            feature_names = json.load(f)["feature_names"]
        extractor = SymptomExtractor(feature_names)
        print(f"Loaded metadata with {len(feature_names)} symptoms.")
    else:
        print("Warning: sanjeevani_model_metadata.json not found! Run train_model.py first.")

    # Load trained Decision Tree model
    model_path = os.path.join("models", "sanjeevani_model.pkl")
    if os.path.exists(model_path):
        model = joblib.load(model_path)
        print("Successfully loaded trained Decision Tree model.")
    else:
        print("Warning: sanjeevani_model.pkl not found! Run train_model.py first.")

@app.get("/")
def health_check():
    return {
        "status": "running",
        "service": "Sanjeevani AI ML & NLP Microservice",
        "triage_ready": extractor is not None,
        "model_loaded": model is not None,
        "map_loaded": router is not None
    }

@app.get("/locations")
def list_locations():
    if router is None:
        raise HTTPException(status_code=500, detail="Hospital routing map not initialized.")
    return [{"id": loc["id"], "name": loc["name"]} for loc in router.locations.values()]

@app.get("/map")
def get_map():
    if router is None:
        raise HTTPException(status_code=500, detail="Hospital routing map not initialized.")
    return router.map_data()

def find_route(location, care):
    """Returns (route, error). Never raises: a routing problem must not hide a triage result."""
    if not location:
        return None, None
    if router is None:
        return None, "Hospital routing is unavailable."
    try:
        return router.route(location, care), None
    except UnknownLocationError:
        return None, f"Unknown location: {location}"

@app.post("/predict")
def predict_disease(request: SymptomRequest):
    if model is None or extractor is None:
        raise HTTPException(status_code=500, detail="Model or metadata not initialized. Run train_model.py first.")

    # 1. NLP Symptom Extraction
    extraction = extractor.extract(request.user_text)
    extracted = extraction["symptoms"]

    response = {
        "status": "success",
        "user_text": request.user_text,
        "extracted_symptoms": extracted,
        "negated_symptoms": extraction["negated_symptoms"],
        "clinical_flags": extraction["clinical_flags"],
        "negated_flags": extraction["negated_flags"],
        "predictions": []
    }

    # 2. Forward Chaining Triage: a detected emergency bypasses the ML model
    triage_result = triage(extracted, extraction["clinical_flags"], request.duration)
    response["triage"] = triage_result

    # 3. A* Search for the fastest hospital offering the required care
    response["route"], response["route_error"] = find_route(request.location, triage_result["primary_care"])

    if triage_result["emergency"]:
        response["status"] = "emergency"
        return response

    # An all-zero input would still yield a confident-looking disease, so don't predict
    if not extracted:
        response["status"] = "no_symptoms_recognized"
        return response

    # 4. Build Binary Feature Vector (named columns, in training order)
    feature_vector = [1 if symptom in extracted else 0 for symptom in feature_names]
    X_input = pd.DataFrame([feature_vector], columns=feature_names, dtype=np.uint8)

    # 5. Decision Tree Prediction & Probabilities
    probabilities = model.predict_proba(X_input)[0]
    classes = model.classes_

    # Top 3 predictions
    top_indices = np.argsort(probabilities)[::-1][:3]
    response["predictions"] = [
        {
            "disease": str(classes[idx]),
            "confidence": round(float(probabilities[idx]), 4)
        }
        for idx in top_indices if probabilities[idx] > 0
    ]

    return response
