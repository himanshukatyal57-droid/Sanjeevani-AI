import heapq
import itertools
import json
import unittest
from pathlib import Path

from routing_engine import RoutingEngine, UnknownLocationError, haversine_km
from triage_engine import CARE_PRIORITY

GRAPH_PATH = Path(__file__).parent / "data" / "delhi_ncr_graph.json"


def load_graph():
    with open(GRAPH_PATH, encoding="utf-8") as f:
        return json.load(f)


class TestGraphData(unittest.TestCase):
    """A* is only guaranteed optimal if the map data keeps the heuristic valid."""

    @classmethod
    def setUpClass(cls):
        cls.graph = load_graph()
        cls.engine = RoutingEngine(cls.graph)

    def test_roads_and_hospitals_reference_real_locations(self):
        ids = set(self.engine.locations)
        for road in self.graph["roads"]:
            self.assertIn(road["from"], ids)
            self.assertIn(road["to"], ids)
        for hospital in self.graph["hospitals"]:
            self.assertIn(hospital["location"], ids)

    def test_no_road_is_shorter_than_the_straight_line(self):
        for road in self.graph["roads"]:
            straight = self.engine.straight_line_km(road["from"], road["to"])
            self.assertGreaterEqual(road["km"], straight, f"{road['from']} -> {road['to']}")

    def test_congestion_never_speeds_a_road_up(self):
        for road in self.graph["roads"]:
            self.assertGreaterEqual(road["congestion"], 1.0)

    def test_heuristic_is_consistent(self):
        # h(a) <= cost(a, b) + h(b) for every road and every possible goal set
        for care in CARE_PRIORITY:
            goals = {h["location"] for h in self.engine.hospitals_for(care)}
            for node, edges in self.engine.adjacency.items():
                for edge in edges:
                    self.assertLessEqual(
                        self.engine.heuristic_minutes(node, goals),
                        edge["minutes"] + self.engine.heuristic_minutes(edge["to"], goals) + 1e-9,
                    )

    def test_every_location_can_reach_every_other(self):
        seen, stack = set(), [next(iter(self.engine.locations))]
        while stack:
            node = stack.pop()
            if node not in seen:
                seen.add(node)
                stack.extend(edge["to"] for edge in self.engine.adjacency[node])
        self.assertEqual(seen, set(self.engine.locations))

    def test_every_care_type_has_a_choice_of_hospitals(self):
        for care in CARE_PRIORITY:
            self.assertGreaterEqual(len(self.engine.hospitals_for(care)), 2, care)

    def test_location_coordinates_are_in_delhi_ncr(self):
        for loc in self.graph["locations"]:
            self.assertTrue(28.2 < loc["lat"] < 28.9 and 76.8 < loc["lon"] < 77.6, loc["id"])

    def test_map_says_it_is_demo_data(self):
        self.assertIn("fictional", self.graph["note"])


class TestAStar(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine = RoutingEngine(load_graph())

    def brute_force_minutes(self, start, goals):
        """Independent check: textbook Dijkstra that shares no code with the search method."""
        distances = {start: 0.0}
        queue = [(0.0, start)]
        while queue:
            d, node = heapq.heappop(queue)
            if d > distances.get(node, float("inf")):
                continue
            for edge in self.engine.adjacency[node]:
                candidate = d + edge["minutes"]
                if candidate < distances.get(edge["to"], float("inf")):
                    distances[edge["to"]] = candidate
                    heapq.heappush(queue, (candidate, edge["to"]))
        return min(distances[g] for g in goals)

    def test_astar_is_optimal_for_every_start_and_care_type(self):
        for start, care in itertools.product(self.engine.locations, CARE_PRIORITY):
            goals = {h["location"] for h in self.engine.hospitals_for(care)}
            result = self.engine.search(start, goals)
            self.assertAlmostEqual(result["minutes"], self.brute_force_minutes(start, goals), places=6,
                                   msg=f"{start} / {care}")

    def test_astar_expands_no_more_than_dijkstra_overall(self):
        astar, dijkstra = 0, 0
        for start, care in itertools.product(self.engine.locations, CARE_PRIORITY):
            goals = {h["location"] for h in self.engine.hospitals_for(care)}
            astar += len(self.engine.search(start, goals)["expansion_steps"])
            dijkstra += len(self.engine.search(start, goals, use_heuristic=False)["expansion_steps"])
        self.assertLess(astar, dijkstra)

    def test_path_is_connected_and_ends_at_goal(self):
        result = self.engine.search("rohini", {"saket"})
        self.assertEqual(result["path"][0], "rohini")
        self.assertEqual(result["path"][-1], "saket")
        for a, b in zip(result["path"], result["path"][1:]):
            self.assertIn(b, [edge["to"] for edge in self.engine.adjacency[a]])

    def test_path_cost_matches_reported_minutes(self):
        result = self.engine.search("dwarka", {"noida_city_centre"})
        total = 0.0
        for a, b in zip(result["path"], result["path"][1:]):
            total += min(e["minutes"] for e in self.engine.adjacency[a] if e["to"] == b)
        self.assertAlmostEqual(total, result["minutes"], places=6)

    def test_start_at_goal_costs_nothing(self):
        result = self.engine.search("saket", {"saket"})
        self.assertEqual(result["path"], ["saket"])
        self.assertEqual(result["minutes"], 0.0)

    def test_unreachable_goal_returns_none(self):
        graph = load_graph()
        graph["locations"].append({"id": "island", "name": "Island", "lat": 28.6, "lon": 77.2})
        self.assertIsNone(RoutingEngine(graph).search("saket", {"island"}))

    def test_haversine_known_distance(self):
        # One degree of latitude is about 111 km
        self.assertAlmostEqual(haversine_km(28.0, 77.0, 29.0, 77.0), 111.2, delta=0.5)


class TestRoute(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine = RoutingEngine(load_graph())

    def test_chosen_hospital_offers_the_care(self):
        for start, care in itertools.product(self.engine.locations, CARE_PRIORITY):
            route = self.engine.route(start, care)
            self.assertIn(care, route["hospital"]["specialties"], f"{start} / {care}")

    def test_never_slower_than_the_closest_suitable_hospital(self):
        for start, care in itertools.product(self.engine.locations, CARE_PRIORITY):
            self.assertGreaterEqual(self.engine.route(start, care)["comparison"]["minutes_saved"], 0)

    def test_skips_a_nearby_clinic_that_cannot_treat(self):
        route = self.engine.route("lajpat_nagar", "cardiac")
        self.assertFalse(route["comparison"]["closest_hospital"]["can_treat"])
        self.assertNotIn("Clinic", route["hospital"]["name"])

    def test_no_care_type_routes_to_general(self):
        route = self.engine.route("lajpat_nagar", None)
        self.assertEqual(route["care"], "general")
        self.assertEqual(route["minutes"], 0.0)

    def test_unknown_location_is_rejected(self):
        with self.assertRaises(UnknownLocationError):
            self.engine.route("atlantis", "cardiac")

    def test_unknown_care_type_returns_none(self):
        self.assertIsNone(self.engine.route("saket", "dentistry"))


class TestSharedData(unittest.TestCase):
    """Files that must stay in step with the trained model."""

    @classmethod
    def setUpClass(cls):
        root = GRAPH_PATH.parent.parent
        cls.metadata = json.loads((root / "models" / "sanjeevani_model_metadata.json").read_text(encoding="utf-8"))
        cls.schema = json.loads((root / "models" / "symptom_schema.json").read_text(encoding="utf-8"))
        cls.specialties = json.loads((root / "data" / "disease_specialties.json").read_text(encoding="utf-8"))["specialties"]

    def test_every_disease_has_a_specialty_the_map_can_serve(self):
        self.assertEqual(set(self.specialties), set(self.metadata["class_labels"]))
        engine = RoutingEngine(load_graph())
        for disease, care in self.specialties.items():
            self.assertTrue(engine.hospitals_for(care), f"{disease}: {care}")

    def test_specialties_make_sense_for_well_known_diseases(self):
        expected = {
            "heart attack": "cardiac", "atrial fibrillation": "cardiac", "stroke": "neuro", "migraine": "neuro",
            "epilepsy": "neuro", "asthma": "respiratory", "pneumonia": "respiratory", "tuberculosis": "respiratory",
            "fracture of the arm": "trauma", "open wound of the head": "trauma", "dislocation of the knee": "trauma",
            "poisoning due to opioids": "toxicology", "insulin overdose": "toxicology", "depression": "mental_health",
            "schizophrenia": "mental_health", "common cold": "general", "kidney stone": "general", "eczema": "general",
        }
        for disease, care in expected.items():
            self.assertEqual(self.specialties[disease], care, disease)

    def test_schema_matches_the_model_exactly(self):
        self.assertEqual(self.schema, self.metadata["feature_names"])
        self.assertEqual(len(self.schema), 377)
        self.assertNotIn("diseases", self.schema)
        self.assertIn("neck_weakness", self.schema)


if __name__ == "__main__":
    unittest.main()
