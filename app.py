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
from tree_explainer import choose_follow_up, explain_tree_prediction
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
disease_specialties = {}

# Not the working directory, so the service loads its files wherever it's started from
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Measured on test rows with 1-4 symptoms: below 25% the top answer was right at most 20%
# of the time, and above 75% it was right 96% of the time, so asking more adds little.
LOW_CONFIDENCE_BELOW = 0.25
FOLLOW_UP_BELOW = 0.75

# Request Schema
class SymptomRequest(BaseModel):
    user_text: str
    location: Optional[str] = None  # A location id from GET /locations
    duration: Optional[str] = None  # The form's answer, e.g. "More than a week"

@app.on_event("startup")
def load_artifacts():
    global model, feature_names, extractor, router, disease_specialties

    # Load the Delhi NCR map for hospital routing
    graph_path = os.path.join(BASE_DIR, "data", "delhi_ncr_graph.json")
    if os.path.exists(graph_path):
        router = RoutingEngine.from_file(graph_path)
        print(f"Loaded map with {len(router.locations)} locations.")
    else:
        print("Warning: delhi_ncr_graph.json not found! Hospital routing is unavailable.")

    # Load the care each disease needs, used to route non-emergency cases
    specialties_path = os.path.join(BASE_DIR, "data", "disease_specialties.json")
    if os.path.exists(specialties_path):
        with open(specialties_path, "r", encoding="utf-8") as f:
            disease_specialties = json.load(f)["specialties"]
    else:
        print("Warning: disease_specialties.json not found! Predictions will route to general care.")

    # Load the exact feature order the model was trained with
    metadata_path = os.path.join(BASE_DIR, "models", "sanjeevani_model_metadata.json")
    if os.path.exists(metadata_path):
        with open(metadata_path, "r", encoding="utf-8") as f:
            feature_names = json.load(f)["feature_names"]
        extractor = SymptomExtractor(feature_names)
        print(f"Loaded metadata with {len(feature_names)} symptoms.")
    else:
        print("Warning: sanjeevani_model_metadata.json not found! Run train_model.py first.")

    # Load trained Decision Tree model
    model_path = os.path.join(BASE_DIR, "models", "sanjeevani_model.pkl")
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

def add_route(response, location, care, source, disease=None):
    response["route"], response["route_error"] = find_route(location, care)
    response["route_basis"] = {"source": source, "disease": disease}

@app.post("/predict")
def predict_disease(request: SymptomRequest):
    # The model is checked later: emergency rules must work without it
    if extractor is None:
        raise HTTPException(status_code=500, detail="Model metadata not initialized. Run train_model.py first.")

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
        "predictions": [],
        "low_confidence": False,
        "follow_up": None,
        "decision_path": None
    }

    # 2. Forward Chaining Triage: a detected emergency bypasses the ML model
    triage_result = triage(extracted, extraction["clinical_flags"], request.duration)
    response["triage"] = triage_result

    triage_care = triage_result["primary_care"]

    # 3. A* Search for an emergency, so routing never waits on a prediction
    if triage_result["emergency"]:
        response["status"] = "emergency"
        add_route(response, request.location, triage_care, "triage")
        return response

    # An all-zero input would still yield a confident-looking disease, so don't predict
    if not extracted:
        response["status"] = "no_symptoms_recognized"
        add_route(response, request.location, triage_care, "triage" if triage_care else "default")
        return response

    if model is None:
        response["status"] = "model_unavailable"
        add_route(response, request.location, triage_care, "triage" if triage_care else "default")
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
    response["decision_path"] = explain_tree_prediction(model, feature_names, feature_vector)

    top_score = float(probabilities[top_indices[0]])
    response["low_confidence"] = top_score < LOW_CONFIDENCE_BELOW
    if response["decision_path"] and top_score < FOLLOW_UP_BELOW:
        response["follow_up"] = choose_follow_up(response["decision_path"], extraction["negated_symptoms"])

    # 6. A* Search for non-emergencies. A specific rule outranks a prediction, so chest pain
    # stays with cardiac care, and a weak guess must not choose the hospital.
    top_disease = str(classes[top_indices[0]])
    if triage_care and triage_care != "general":
        add_route(response, request.location, triage_care, "triage")
    elif response["low_confidence"]:
        add_route(response, request.location, triage_care, "triage" if triage_care else "default")
    else:
        add_route(response, request.location, disease_specialties.get(top_disease, "general"), "prediction", top_disease)

    return response
