"""Tests the /predict contract with a stand-in model, so no training is needed."""
import os
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from fastapi.testclient import TestClient
from sklearn.tree import DecisionTreeClassifier

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

    @staticmethod
    def cough_tree():
        X = pd.DataFrame([[0, 1, 0]] * 8 + [[0, 1, 1]] * 2, columns=FEATURES, dtype=np.uint8)
        return DecisionTreeClassifier(random_state=0).fit(X, [str(i) for i in range(8)] + ["x", "x"])

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

    def test_duration_reaches_triage(self):
        features = FEATURES + ["slurring_words"]
        service.feature_names = features
        service.extractor = SymptomExtractor(features)
        recent = self.client.post("/predict", json={"user_text": "slurred speech", "duration": "1–3 days"}).json()
        self.assertEqual(recent["status"], "emergency")
        self.assertIsNone(self.model.last_input)
        chronic = self.client.post("/predict", json={"user_text": "slurred speech",
                                                     "duration": "More than a week"}).json()
        self.assertEqual(chronic["triage"]["level"], "URGENT")
        self.assertTrue(chronic["predictions"])

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

    def test_real_tree_returns_its_path(self):
        X = np.array([[0, 0, 0], [0, 0, 1], [1, 0, 0], [0, 1, 1]])
        service.model = DecisionTreeClassifier(random_state=0).fit(
            pd.DataFrame(X, columns=FEATURES, dtype=np.uint8), ["a", "b", "c", "d"])
        body = self.post("I have a fever").json()
        self.assertTrue(body["decision_path"]["steps"])
        fever_steps = [s for s in body["decision_path"]["steps"] if s["symptom"] == "fever"]
        self.assertTrue(all(s["present"] for s in fever_steps))

    def routed_care(self, text, tree):
        service.router = RoutingEngine.from_file(GRAPH_PATH)
        service.model = tree
        return self.client.post("/predict", json={"user_text": text, "location": "saket"}).json()

    def asthma_tree(self):
        X = pd.DataFrame([[0, 1, 0]] * 5 + [[0, 0, 1]] * 5, columns=FEATURES, dtype=np.uint8)
        return DecisionTreeClassifier(random_state=0).fit(X, ["asthma"] * 5 + ["flu"] * 5)

    def test_confident_prediction_chooses_the_specialty(self):
        service.disease_specialties = {"asthma": "respiratory", "flu": "general"}
        self.addCleanup(setattr, service, "disease_specialties", {})
        body = self.routed_care("I have a cough", self.asthma_tree())
        self.assertEqual(body["route"]["care"], "respiratory")
        self.assertIn("respiratory", body["route"]["hospital"]["specialties"])
        self.assertEqual(body["route_basis"], {"source": "prediction", "disease": "asthma"})

    def test_weak_prediction_never_chooses_the_hospital(self):
        service.disease_specialties = {str(i): "cardiac" for i in range(8)}
        self.addCleanup(setattr, service, "disease_specialties", {})
        body = self.routed_care("I have a cough", self.cough_tree())
        self.assertTrue(body["low_confidence"])
        self.assertEqual(body["route"]["care"], "general")
        self.assertEqual(body["route_basis"]["source"], "default")

    def test_specific_triage_care_outranks_the_prediction(self):
        features = FEATURES + ["sharp_chest_pain"]
        service.feature_names = features
        service.extractor = SymptomExtractor(features)
        service.disease_specialties = {"a": "respiratory"}
        self.addCleanup(setattr, service, "disease_specialties", {})
        X = pd.DataFrame([[0, 0, 0, 1]] * 5 + [[0, 1, 0, 0]] * 5, columns=features, dtype=np.uint8)
        tree = DecisionTreeClassifier(random_state=0).fit(X, ["a"] * 5 + ["b"] * 5)
        body = self.routed_care("I have chest pain", tree)
        self.assertEqual(body["triage"]["level"], "URGENT")
        self.assertEqual(body["route"]["care"], "cardiac")
        self.assertEqual(body["route_basis"]["source"], "triage")

    def test_emergency_route_never_uses_a_prediction(self):
        service.disease_specialties = {"asthma": "respiratory"}
        self.addCleanup(setattr, service, "disease_specialties", {})
        body = self.routed_care("I am bleeding heavily and have a cough", self.asthma_tree())
        self.assertEqual(body["route"]["care"], "trauma")
        self.assertEqual(body["route_basis"], {"source": "triage", "disease": None})
        self.assertEqual(body["predictions"], [])

    def test_disease_missing_from_the_mapping_uses_general_care(self):
        service.disease_specialties = {}
        body = self.routed_care("I have a cough", self.asthma_tree())
        self.assertEqual(body["route"]["care"], "general")

    def test_route_works_when_nothing_is_recognized_or_the_model_is_missing(self):
        service.router = RoutingEngine.from_file(GRAPH_PATH)
        body = self.client.post("/predict", json={"user_text": "hello there", "location": "saket"}).json()
        self.assertEqual(body["route_basis"]["source"], "default")
        self.assertEqual(body["route"]["care"], "general")
        service.model = None
        body = self.client.post("/predict", json={"user_text": "I have a cough", "location": "saket"}).json()
        self.assertEqual(body["status"], "model_unavailable")
        self.assertEqual(body["route"]["care"], "general")

    def test_low_scores_are_flagged_and_ask_a_question(self):
        # Cough alone is shared by 8 diseases, and only "fever" separates them from a ninth
        service.model = self.cough_tree()
        body = self.post("I have a cough").json()
        self.assertTrue(body["low_confidence"])
        self.assertEqual(body["follow_up"], "fever")

    def test_confident_result_asks_nothing(self):
        X = pd.DataFrame([[0, 1, 0]] * 5 + [[0, 0, 1]] * 5, columns=FEATURES, dtype=np.uint8)
        service.model = DecisionTreeClassifier(random_state=0).fit(X, ["a"] * 5 + ["b"] * 5)
        body = self.post("I have a cough").json()
        self.assertFalse(body["low_confidence"])
        self.assertIsNone(body["follow_up"])

    def test_denied_symptoms_are_not_asked_again(self):
        service.model = self.cough_tree()
        self.assertEqual(self.post("I have a cough").json()["follow_up"], "fever")
        self.assertIsNone(self.post("I have a cough, no fever").json()["follow_up"])

    def test_emergency_asks_nothing(self):
        body = self.post("I am bleeding heavily").json()
        self.assertIsNone(body["follow_up"])
        self.assertFalse(body["low_confidence"])

    def test_no_path_when_the_model_is_skipped(self):
        self.assertIsNone(self.post("I am bleeding heavily").json()["decision_path"])
        self.assertIsNone(self.post("hello there").json()["decision_path"])

    def test_map_endpoint(self):
        service.router = RoutingEngine.from_file(GRAPH_PATH)
        body = self.client.get("/map").json()
        self.assertEqual(len(body["locations"]), 30)
        self.assertEqual(set(body["roads"][0]), {"from", "to", "congestion"})
        self.assertTrue(body["hospitals"])

    def test_map_endpoint_without_a_map(self):
        service.router = None
        self.assertEqual(self.client.get("/map").status_code, 500)

    def test_health_reports_what_is_loaded(self):
        service.router = None
        body = self.client.get("/").json()
        self.assertTrue(body["triage_ready"])
        self.assertTrue(body["model_loaded"])
        self.assertFalse(body["map_loaded"])

    def test_emergency_rules_work_without_the_model(self):
        service.model = None
        body = self.post("I am bleeding heavily").json()
        self.assertEqual(body["status"], "emergency")
        self.assertEqual(body["triage"]["level"], "EMERGENCY")

    def test_missing_model_only_stops_the_prediction(self):
        service.model = None
        response = self.post("I have a fever")
        body = response.json()
        self.assertEqual(response.status_code, 200)
        self.assertEqual(body["status"], "model_unavailable")
        self.assertEqual(body["triage"]["level"], "ROUTINE")
        self.assertEqual(body["predictions"], [])

    def test_missing_metadata_is_an_error(self):
        service.extractor = None
        self.assertEqual(self.post("fever").status_code, 500)

    def test_files_load_from_any_working_directory(self):
        service.router = service.extractor = None
        previous = os.getcwd()
        with tempfile.TemporaryDirectory() as elsewhere:
            os.chdir(elsewhere)
            try:
                # The real model takes 1 GB, and this only checks the paths
                with patch.object(service.joblib, "load", return_value=self.model) as load:
                    service.load_artifacts()
            finally:
                os.chdir(previous)
        self.assertIsNotNone(service.router)
        self.assertIsNotNone(service.extractor)
        load.assert_called_once()


if __name__ == "__main__":
    unittest.main()
