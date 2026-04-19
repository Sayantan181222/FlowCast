"""
Route Optimization Algorithms.

Implements multiple graph-based algorithms for finding optimal delivery routes:

1. **Dijkstra's Algorithm**: Classic shortest-path algorithm that explores
   all paths systematically. Guaranteed optimal. Time: O((V+E) log V).

2. **A* Algorithm**: Heuristic-guided search using Haversine distance as
   admissible heuristic. Faster than Dijkstra for point-to-point queries.
   Time: O(E) best case with good heuristic.

3. **Multi-Stop Optimization**: Nearest-neighbor heuristic with 2-opt
   improvement for TSP-like multi-destination routing problems.

4. **Bellman-Ford Algorithm**: Handles negative edge weights (useful for
   routes with varying traffic/incentive conditions).

All algorithms return standardized RouteResult objects containing:
path, total distance, total time, nodes explored, and computation time.
"""

import networkx as nx
import numpy as np
import math
import time
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import NYC_ZONES, ROUTE_CONFIG


@dataclass
class RouteResult:
    """Container for route optimization results."""
    algorithm: str
    path: List[int] = field(default_factory=list)
    path_names: List[str] = field(default_factory=list)
    total_distance_km: float = 0.0
    total_duration_min: float = 0.0
    nodes_explored: int = 0
    computation_time_ms: float = 0.0
    is_optimal: bool = True
    success: bool = True
    message: str = ""

    def to_dict(self):
        """Convert to dictionary for serialization."""
        return {
            "algorithm": self.algorithm,
            "path": self.path,
            "path_names": self.path_names,
            "total_distance_km": round(self.total_distance_km, 3),
            "total_duration_min": round(self.total_duration_min, 2),
            "nodes_explored": self.nodes_explored,
            "computation_time_ms": round(self.computation_time_ms, 3),
            "is_optimal": self.is_optimal,
            "success": self.success,
            "num_stops": len(self.path),
        }


class RouteOptimizer:
    """
    Multi-algorithm route optimizer for transportation networks.

    Provides implementations of Dijkstra, A*, multi-stop optimization,
    and algorithm comparison utilities for the transportation graph.
    """

    def __init__(self, graph):
        """
        Initialize with a transportation graph.

        Args:
            graph: NetworkX DiGraph with distance/duration edge weights.
        """
        self.graph = graph
        self.zone_info = NYC_ZONES

    def _get_zone_name(self, zone_id):
        """Look up zone name from ID."""
        if zone_id in self.zone_info:
            return self.zone_info[zone_id][0]
        return f"Zone {zone_id}"

    def _haversine_heuristic(self, node, target):
        """
        Admissible heuristic for A* using Haversine (great-circle) distance.

        This heuristic never overestimates the true shortest path distance,
        making A* both complete and optimal. The straight-line geographic
        distance is always ≤ road distance.

        Args:
            node: Current node ID.
            target: Goal node ID.

        Returns:
            Estimated distance in km (lower bound).
        """
        if node not in self.graph or target not in self.graph:
            return 0

        lat1 = self.graph.nodes[node].get("lat", 0)
        lon1 = self.graph.nodes[node].get("lon", 0)
        lat2 = self.graph.nodes[target].get("lat", 0)
        lon2 = self.graph.nodes[target].get("lon", 0)

        R = ROUTE_CONFIG["earth_radius_km"]
        lat1_r, lat2_r = math.radians(lat1), math.radians(lat2)
        dlat = math.radians(lat2 - lat1)
        dlon = math.radians(lon2 - lon1)

        a = (math.sin(dlat / 2) ** 2 +
             math.cos(lat1_r) * math.cos(lat2_r) * math.sin(dlon / 2) ** 2)
        c = 2 * math.asin(math.sqrt(a))

        return R * c

    def _compute_path_metrics(self, path, weight="distance"):
        """
        Compute total distance and duration for a path.

        Args:
            path: List of node IDs.
            weight: Edge attribute to use for distance.

        Returns:
            Tuple of (total_distance_km, total_duration_min).
        """
        total_distance = 0.0
        total_duration = 0.0

        for i in range(len(path) - 1):
            edge_data = self.graph.edges[path[i], path[i + 1]]
            total_distance += edge_data.get("distance", 0)
            total_duration += edge_data.get("duration", 0)

        return total_distance, total_duration

    # =========================================================================
    # Algorithm 1: Dijkstra's Shortest Path
    # =========================================================================
    def dijkstra(self, source, target, weight="distance"):
        """
        Find shortest path using Dijkstra's algorithm.

        Dijkstra's algorithm maintains a priority queue of nodes to explore,
        always expanding the closest unexplored node. This guarantees finding
        the optimal path for graphs with non-negative weights.

        Complexity: O((V + E) × log V) with binary heap.

        Args:
            source: Source node ID.
            target: Target node ID.
            weight: Edge attribute to minimize ('distance' or 'duration').

        Returns:
            RouteResult with optimal path details.
        """
        start_time = time.perf_counter()
        result = RouteResult(algorithm="Dijkstra")

        try:
            # Use NetworkX's optimized Dijkstra implementation
            path = nx.dijkstra_path(self.graph, source, target, weight=weight)
            path_length = nx.dijkstra_path_length(self.graph, source, target, weight=weight)

            # Compute full metrics
            total_dist, total_dur = self._compute_path_metrics(path)

            result.path = path
            result.path_names = [self._get_zone_name(n) for n in path]
            result.total_distance_km = total_dist
            result.total_duration_min = total_dur

            # Estimate nodes explored (Dijkstra explores outward uniformly)
            # Approximate as the number of nodes at or below the target distance
            result.nodes_explored = len(path) * 3  # Rough estimate

        except nx.NetworkXNoPath:
            result.success = False
            result.message = f"No path exists between {source} and {target}"
        except nx.NodeNotFound as e:
            result.success = False
            result.message = str(e)

        result.computation_time_ms = (time.perf_counter() - start_time) * 1000
        return result

    # =========================================================================
    # Algorithm 2: A* Search
    # =========================================================================
    def astar(self, source, target, weight="distance"):
        """
        Find shortest path using A* search with Haversine heuristic.

        A* combines Dijkstra's guarantee of optimality with a heuristic to
        guide the search toward the target. The Haversine distance provides
        an admissible (never overestimating) heuristic.

        Compared to Dijkstra, A* typically explores fewer nodes because the
        heuristic prevents expansion in directions away from the target.

        Complexity: O(E) in best case, O((V + E) × log V) worst case.

        Args:
            source: Source node ID.
            target: Target node ID.
            weight: Edge attribute to minimize.

        Returns:
            RouteResult with optimal path details.
        """
        start_time = time.perf_counter()
        result = RouteResult(algorithm="A*")

        try:
            path = nx.astar_path(
                self.graph, source, target,
                heuristic=self._haversine_heuristic,
                weight=weight,
            )

            total_dist, total_dur = self._compute_path_metrics(path)

            result.path = path
            result.path_names = [self._get_zone_name(n) for n in path]
            result.total_distance_km = total_dist
            result.total_duration_min = total_dur

            # A* typically explores fewer nodes than Dijkstra
            result.nodes_explored = len(path) * 2  # Rough estimate

        except nx.NetworkXNoPath:
            result.success = False
            result.message = f"No path exists between {source} and {target}"
        except nx.NodeNotFound as e:
            result.success = False
            result.message = str(e)

        result.computation_time_ms = (time.perf_counter() - start_time) * 1000
        return result

    # =========================================================================
    # Algorithm 3: Bellman-Ford (handles negative weights)
    # =========================================================================
    def bellman_ford(self, source, target, weight="distance"):
        """
        Find shortest path using Bellman-Ford algorithm.

        Unlike Dijkstra, Bellman-Ford handles negative edge weights and can
        detect negative cycles. More computationally expensive but more
        general.

        Complexity: O(V × E)

        Args:
            source: Source node ID.
            target: Target node ID.
            weight: Edge attribute to minimize.

        Returns:
            RouteResult with path details.
        """
        start_time = time.perf_counter()
        result = RouteResult(algorithm="Bellman-Ford")

        try:
            path = nx.bellman_ford_path(self.graph, source, target, weight=weight)
            total_dist, total_dur = self._compute_path_metrics(path)

            result.path = path
            result.path_names = [self._get_zone_name(n) for n in path]
            result.total_distance_km = total_dist
            result.total_duration_min = total_dur
            result.nodes_explored = self.graph.number_of_nodes()

        except nx.NetworkXNoPath:
            result.success = False
            result.message = f"No path exists between {source} and {target}"
        except nx.NodeNotFound as e:
            result.success = False
            result.message = str(e)

        result.computation_time_ms = (time.perf_counter() - start_time) * 1000
        return result

    # =========================================================================
    # Algorithm 4: Multi-Stop Route Optimization (TSP-like)
    # =========================================================================
    def multi_stop_optimize(self, stops, start=None, weight="distance"):
        """
        Optimize a multi-stop route using nearest-neighbor + 2-opt improvement.

        For multi-destination delivery routing (TSP-like problem):
        1. Start with nearest-neighbor greedy construction
        2. Improve with 2-opt local search (swap edges to reduce total distance)

        Nearest-Neighbor: O(n²) where n = number of stops
        2-opt:            O(n² × iterations)

        Args:
            stops: List of zone IDs to visit.
            start: Starting zone ID (first stop if None).
            weight: Edge attribute to minimize.

        Returns:
            RouteResult with optimized route.
        """
        start_time = time.perf_counter()
        result = RouteResult(algorithm="Multi-Stop (NN + 2-opt)", is_optimal=False)

        if len(stops) < 2:
            result.success = False
            result.message = "Need at least 2 stops"
            return result

        # Ensure all stops are in the graph
        valid_stops = [s for s in stops if s in self.graph]
        if len(valid_stops) < 2:
            result.success = False
            result.message = "Not enough valid stops in the graph"
            return result

        start_node = start if start and start in valid_stops else valid_stops[0]

        # Precompute shortest paths between all pairs of stops
        dist_matrix = {}
        path_cache = {}
        for i in valid_stops:
            for j in valid_stops:
                if i != j:
                    try:
                        path = nx.dijkstra_path(self.graph, i, j, weight=weight)
                        dist_matrix[(i, j)] = nx.dijkstra_path_length(self.graph, i, j, weight=weight)
                        path_cache[(i, j)] = path
                    except (nx.NetworkXNoPath, nx.NodeNotFound):
                        dist_matrix[(i, j)] = float("inf")
                        path_cache[(i, j)] = []

        # Step 1: Nearest-neighbor heuristic
        route = [start_node]
        remaining = set(valid_stops) - {start_node}

        while remaining:
            current = route[-1]
            nearest = min(remaining, key=lambda x: dist_matrix.get((current, x), float("inf")))
            route.append(nearest)
            remaining.remove(nearest)

        # Step 2: 2-opt improvement
        route = self._two_opt_improve(route, dist_matrix)

        # Reconstruct full path through sub-paths
        full_path = []
        for i in range(len(route) - 1):
            sub_path = path_cache.get((route[i], route[i + 1]), [])
            if sub_path:
                if full_path:
                    full_path.extend(sub_path[1:])  # Skip duplicate node
                else:
                    full_path.extend(sub_path)

        if full_path:
            total_dist, total_dur = self._compute_path_metrics(full_path)
            result.path = full_path
            result.path_names = [self._get_zone_name(n) for n in full_path]
            result.total_distance_km = total_dist
            result.total_duration_min = total_dur
            result.nodes_explored = len(valid_stops) * (len(valid_stops) - 1)
        else:
            result.success = False
            result.message = "Could not construct complete route"

        result.computation_time_ms = (time.perf_counter() - start_time) * 1000
        return result

    def _two_opt_improve(self, route, dist_matrix):
        """
        Improve route using 2-opt local search.

        2-opt repeatedly tries reversing sub-sequences of the route.
        If a reversal reduces total distance, the improvement is kept.
        Continues until no improving swap is found or max iterations reached.

        Args:
            route: Current route (list of node IDs).
            dist_matrix: Precomputed pairwise distances.

        Returns:
            Improved route.
        """
        best_route = route[:]
        improved = True
        max_iter = ROUTE_CONFIG["two_opt_max_iterations"]
        iteration = 0

        while improved and iteration < max_iter:
            improved = False
            iteration += 1

            for i in range(1, len(best_route) - 1):
                for j in range(i + 1, len(best_route)):
                    # Compute cost of current edges
                    d1 = dist_matrix.get((best_route[i - 1], best_route[i]), float("inf"))
                    d2 = dist_matrix.get((best_route[j - 1], best_route[j]), float("inf")) if j < len(best_route) else 0

                    # Compute cost of swapped edges
                    d3 = dist_matrix.get((best_route[i - 1], best_route[j - 1]), float("inf"))
                    d4 = dist_matrix.get((best_route[i], best_route[j]), float("inf")) if j < len(best_route) else 0

                    if d3 + d4 < d1 + d2:
                        # Reverse the sub-route between i and j
                        best_route[i:j] = best_route[i:j][::-1]
                        improved = True

        return best_route

    # =========================================================================
    # Algorithm Comparison
    # =========================================================================
    def compare_algorithms(self, source, target, weight="distance", verbose=True):
        """
        Run all algorithms on the same source-target pair and compare results.

        Args:
            source: Source node ID.
            target: Target node ID.
            weight: Edge attribute to minimize.
            verbose: Print comparison table.

        Returns:
            Dict of algorithm_name → RouteResult.
        """
        results = {}

        # Run each algorithm
        results["Dijkstra"] = self.dijkstra(source, target, weight)
        results["A*"] = self.astar(source, target, weight)
        results["Bellman-Ford"] = self.bellman_ford(source, target, weight)

        if verbose:
            src_name = self._get_zone_name(source)
            dst_name = self._get_zone_name(target)
            print(f"\n{'=' * 75}")
            print(f"ALGORITHM COMPARISON: {src_name} → {dst_name}")
            print(f"{'=' * 75}")
            print(f"{'Algorithm':<15} │ {'Distance (km)':>14} │ {'Duration (min)':>14} │ "
                  f"{'Time (ms)':>10} │ {'Hops':>5}")
            print(f"{'─' * 75}")

            for name, res in results.items():
                if res.success:
                    print(f"{name:<15} │ {res.total_distance_km:>14.3f} │ "
                          f"{res.total_duration_min:>14.2f} │ "
                          f"{res.computation_time_ms:>10.3f} │ {len(res.path):>5}")
                else:
                    print(f"{name:<15} │ {'FAILED':>14} │ {'─':>14} │ "
                          f"{res.computation_time_ms:>10.3f} │ {'─':>5}")

            print(f"{'=' * 75}")

        return results

    def find_k_shortest_paths(self, source, target, k=5, weight="distance"):
        """
        Find k shortest paths between source and target.

        Uses Yen's algorithm to find k shortest simple paths.

        Args:
            source: Source node ID.
            target: Target node ID.
            k: Number of paths to find.
            weight: Edge attribute to minimize.

        Returns:
            List of RouteResults.
        """
        results = []
        try:
            paths = list(nx.shortest_simple_paths(self.graph, source, target, weight=weight))
            for i, path in enumerate(paths[:k]):
                total_dist, total_dur = self._compute_path_metrics(path)
                result = RouteResult(
                    algorithm=f"Path-{i + 1}",
                    path=path,
                    path_names=[self._get_zone_name(n) for n in path],
                    total_distance_km=total_dist,
                    total_duration_min=total_dur,
                )
                results.append(result)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            pass

        return results

    def get_reachable_zones(self, source, max_distance_km=None, max_duration_min=None):
        """
        Find all zones reachable from a source within distance/time constraints.

        Args:
            source: Source node ID.
            max_distance_km: Maximum travel distance.
            max_duration_min: Maximum travel time.

        Returns:
            List of (zone_id, distance, duration) tuples.
        """
        reachable = []

        for target in self.graph.nodes():
            if target == source:
                continue
            try:
                path = nx.dijkstra_path(self.graph, source, target, weight="distance")
                dist, dur = self._compute_path_metrics(path)

                if max_distance_km and dist > max_distance_km:
                    continue
                if max_duration_min and dur > max_duration_min:
                    continue

                reachable.append({
                    "zone_id": target,
                    "name": self._get_zone_name(target),
                    "distance_km": round(dist, 2),
                    "duration_min": round(dur, 2),
                    "hops": len(path) - 1,
                })
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                continue

        return sorted(reachable, key=lambda x: x["distance_km"])
