"""
A* search for the fastest route to a hospital that can provide the required care.

Implemented from scratch on a custom Delhi NCR graph, with no map API. Edge cost is
travel time in minutes: distance / road speed * congestion.

Heuristic: straight-line distance to the closest suitable hospital, at the fastest road
speed with no congestion. No real trip can beat that, so the heuristic never
overestimates and A* is guaranteed to find the fastest route.
"""
import heapq
import json
import math
from typing import Dict, List, Optional, Set

EARTH_RADIUS_KM = 6371.0


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    d_phi = phi2 - phi1
    d_lambda = math.radians(lon2 - lon1)
    a = math.sin(d_phi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2) ** 2
    return 2 * EARTH_RADIUS_KM * math.asin(math.sqrt(a))


class UnknownLocationError(ValueError):
    pass


class RoutingEngine:
    def __init__(self, graph: dict):
        self.note = graph["note"]
        self.locations = {loc["id"]: loc for loc in graph["locations"]}
        self.hospitals = graph["hospitals"]
        speeds = graph["road_speeds_kmph"]
        self.max_speed_kmph = max(speeds.values())

        self.adjacency: Dict[str, List[dict]] = {loc_id: [] for loc_id in self.locations}
        for road in graph["roads"]:
            minutes = road["km"] / speeds[road["type"]] * 60 * road["congestion"]
            # Roads are two-way
            self.adjacency[road["from"]].append({"to": road["to"], "km": road["km"], "minutes": minutes})
            self.adjacency[road["to"]].append({"to": road["from"], "km": road["km"], "minutes": minutes})

    @classmethod
    def from_file(cls, path: str) -> "RoutingEngine":
        with open(path, "r", encoding="utf-8") as f:
            return cls(json.load(f))

    def straight_line_km(self, a: str, b: str) -> float:
        first, second = self.locations[a], self.locations[b]
        return haversine_km(first["lat"], first["lon"], second["lat"], second["lon"])

    def heuristic_minutes(self, node: str, goals: Set[str]) -> float:
        closest_km = min(self.straight_line_km(node, goal) for goal in goals)
        return closest_km / self.max_speed_kmph * 60

    def search(self, start: str, goals: Set[str], use_heuristic: bool = True) -> Optional[dict]:
        """A* from start to the best of several goals. Without the heuristic this is Dijkstra."""
        def h(node):
            return self.heuristic_minutes(node, goals) if use_heuristic else 0.0

        # Entries are (f, tie-breaker, g, node). The counter avoids comparing further fields.
        counter = 0
        frontier = [(h(start), counter, 0.0, start)]
        best_g = {start: 0.0}
        came_from: Dict[str, Optional[str]] = {start: None}
        expanded: Set[str] = set()
        expansion_steps = []

        while frontier:
            f, _, g, node = heapq.heappop(frontier)
            if node in expanded:
                continue  # A faster way here was already processed
            expanded.add(node)
            expansion_steps.append({"node": node, "g_minutes": round(g, 2), "h_minutes": round(f - g, 2),
                                    "f_minutes": round(f, 2), "frontier_size": len(frontier)})

            if node in goals:
                path = [node]
                while came_from[path[-1]] is not None:
                    path.append(came_from[path[-1]])
                path.reverse()
                return {"goal": node, "path": path, "minutes": g, "expansion_steps": expansion_steps}

            for edge in self.adjacency[node]:
                neighbour = edge["to"]
                new_g = g + edge["minutes"]
                if neighbour not in best_g or new_g < best_g[neighbour]:
                    best_g[neighbour] = new_g
                    came_from[neighbour] = node
                    counter += 1
                    heapq.heappush(frontier, (new_g + h(neighbour), counter, new_g, neighbour))
        return None

    def _path_km(self, path: List[str]) -> float:
        total = 0.0
        for a, b in zip(path, path[1:]):
            total += min(edge["km"] for edge in self.adjacency[a] if edge["to"] == b)
        return total

    def hospitals_for(self, care: str) -> List[dict]:
        return [h for h in self.hospitals if care in h["specialties"]]

    def _closest_hospital(self, start: str, hospitals: List[dict]) -> dict:
        return min(hospitals, key=lambda h: self.straight_line_km(start, h["location"]))

    def route(self, start: str, care: Optional[str]) -> Optional[dict]:
        """Fastest route to a suitable hospital, compared with simpler choices."""
        if start not in self.locations:
            raise UnknownLocationError(start)
        care = care or "general"
        suitable = self.hospitals_for(care)
        if not suitable:
            return None
        goals = {h["location"] for h in suitable}

        result = self.search(start, goals)
        if result is None:
            return None
        dijkstra = self.search(start, goals, use_heuristic=False)

        hospital = next(h for h in suitable if h["location"] == result["goal"])
        closest_any = self._closest_hospital(start, self.hospitals)
        closest_suitable = self._closest_hospital(start, suitable)
        closest_suitable_route = self.search(start, {closest_suitable["location"]})

        return {
            "care": care,
            "hospital": {"id": hospital["id"], "name": hospital["name"], "specialties": hospital["specialties"],
                         "location": self.locations[hospital["location"]]["name"]},
            "path": [{"id": node, "name": self.locations[node]["name"], "lat": self.locations[node]["lat"],
                      "lon": self.locations[node]["lon"]} for node in result["path"]],
            "minutes": round(result["minutes"], 1),
            "km": round(self._path_km(result["path"]), 1),
            "comparison": {
                # The nearest hospital on the map may not offer this care at all
                "closest_hospital": {"name": closest_any["name"], "can_treat": care in closest_any["specialties"]},
                # The nearest suitable one by distance may still be slower because of traffic
                "closest_suitable_hospital": {"name": closest_suitable["name"],
                                              "minutes": round(closest_suitable_route["minutes"], 1)},
                "minutes_saved": round(closest_suitable_route["minutes"] - result["minutes"], 1),
            },
            "search": {
                "nodes_expanded": len(result["expansion_steps"]),
                "dijkstra_nodes_expanded": len(dijkstra["expansion_steps"]),
                "expansion_steps": result["expansion_steps"],
            },
            "data_note": self.note,
        }
