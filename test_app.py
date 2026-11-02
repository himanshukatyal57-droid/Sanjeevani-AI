"""Tests the /predict contract with a stand-in model, so no training is needed."""
import os
import unittest

import numpy as np
from fastapi.testclient import TestClient

import app as service
from nlp_extractor import SymptomExtractor
from routing_engine import RoutingEngine

GRAPH_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "delhi_ncr_graph.json")

# Deliberately not alphabetical, like the real training columns
FEATURES = ["headache", "cough", "fever"]


class StubModel:
    classes_ = np.array(["flu", "migraine", "other"])

    def __init__(self):
        self.last_input = None

    def predict_proba(self, X):
        self.last_input = X
        return np.array([[0.7, 0.3, 0.0]])


class TestPredict(unittest.TestCase):
    def setUp(self):
        self.model = StubModel()
        service.model = self.model
        service.feature_names = FEATURES
        service.extractor = SymptomExtractor(FEATURES)
        self.client = TestClient(service.app)

    def post(self, text):
        return self.client.post("/predict", json={"user_text": text})

    def test_input_uses_training_feature_order(self):
        self.post("I have a fever")
        X = self.model.last_input
        self.assertEqual(list(X.columns), FEATURES)
        self.assertEqual(X.iloc[0].tolist(), [0, 0, 1])

    def test_returns_predictions_above_zero(self):
        body = self.post("I have a fever").json()
        self.assertEqual(body["status"], "success")
        self.assertEqual(body["extracted_symptoms"], ["fever"])
        self.assertEqual([p["disease"] for p in body["predictions"]], ["flu", "migraine"])

    def test_no_symptoms_means_no_prediction(self):
        body = self.post("hello there").json()
        self.assertEqual(body["status"], "no_symptoms_recognized")
        self.assertEqual(body["predictions"], [])
        self.assertIsNone(self.model.last_input)

    def test_only_negated_symptoms_means_no_prediction(self):
        body = self.post("no fever").json()
        self.assertEqual(body["status"], "no_symptoms_recognized")
        self.assertEqual(body["negated_symptoms"], ["fever"])
        self.assertIsNone(self.model.last_input)

    def test_emergency_bypasses_the_model(self):
        body = self.post("I am bleeding heavily and have a fever").json()
        self.assertEqual(body["status"], "emergency")
        self.assertEqual(body["clinical_flags"], ["severe_bleeding"])
        self.assertEqual(body["triage"]["level"], "EMERGENCY")
        self.assertEqual(body["predictions"], [])
        self.assertIsNone(self.model.last_input)

    def test_non_emergency_includes_triage_and_predictions(self):
        body = self.post("I have a fever").json()
        self.assertEqual(body["triage"]["level"], "ROUTINE")
        self.assertTrue(body["predictions"])

    def test_nothing_recognized_is_not_called_routine(self):
        body = self.post("hello there").json()
        self.assertEqual(body["triage"]["level"], "UNKNOWN")

    def test_emergency_includes_a_route_to_suitable_care(self):
        service.router = RoutingEngine.from_file(GRAPH_PATH)
        body = self.client.post("/predict", json={"user_text": "I am bleeding heavily",
                                                  "location": "lajpat_nagar"}).json()
        self.assertEqual(body["status"], "emergency")
        self.assertIn("trauma", body["route"]["hospital"]["specialties"])
        self.assertEqual(body["route"]["path"][0]["id"], "lajpat_nagar")

    def test_route_is_optional(self):
        self.assertIsNone(self.post("I have a fever").json()["route"])

    def test_unknown_location_never_hides_an_emergency(self):
        service.router = RoutingEngine.from_file(GRAPH_PATH)
        response = self.client.post("/predict", json={"user_text": "I am bleeding heavily", "location": "atlantis"})
        body = response.json()
        self.assertEqual(response.status_code, 200)
        self.assertEqual(body["status"], "emergency")
        self.assertIsNone(body["route"])
        self.assertIn("atlantis", body["route_error"])

    def test_missing_map_never_hides_an_emergency(self):
        service.router = None
        body = self.client.post("/predict", json={"user_text": "I am bleeding heavily",
                                                  "location": "saket"}).json()
        self.assertEqual(body["status"], "emergency")
        self.assertEqual(body["route_error"], "Hospital routing is unavailable.")

    def test_locations_endpoint(self):
        service.router = RoutingEngine.from_file(GRAPH_PATH)
        names = [loc["name"] for loc in self.client.get("/locations").json()]
        self.assertIn("Saket", names)

    def test_missing_model_is_an_error(self):
        service.model = None
        self.assertEqual(self.post("fever").status_code, 500)


if __name__ == "__main__":
    unittest.main()
