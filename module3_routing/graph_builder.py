"""
Transportation Network Graph Builder.

Constructs a weighted graph G = (V, E) representing the transportation network:
- V (Vertices/Nodes): NYC taxi zones with geographic coordinates
- E (Edges): Connections between zones derived from actual trip data
- Edge weights: average travel time, average distance, trip frequency

The graph enables route optimization algorithms (Dijkstra, A*) to find
optimal paths between any two connected locations in the network.
"""

import numpy as np
import pandas as pd
import networkx as nx
import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import NYC_ZONES, ROUTE_CONFIG


class TransportationGraphBuilder:
    """
    Builds and manages the transportation network graph.

    Constructs a directed weighted graph from taxi trip data where:
    - Each taxi zone becomes a node with (lat, lon) attributes
    - Each observed route between zones becomes an edge
    - Edge weights include distance, travel time, and trip count
    """

    def __init__(self):
        """Initialize the graph builder."""
        self.graph = None
        self.zone_info = NYC_ZONES
        self.stats = {}

    @staticmethod
    def haversine_distance(lat1, lon1, lat2, lon2):
        """
        Compute great-circle distance between two geographic points.

        Uses the Haversine formula which accounts for Earth's curvature,
        providing accurate distance estimates for nearby points.

        Args:
            lat1, lon1: Coordinates of point 1 (degrees).
            lat2, lon2: Coordinates of point 2 (degrees).

        Returns:
            Distance in kilometers.
        """
        R = ROUTE_CONFIG["earth_radius_km"]

        lat1_r, lat2_r = math.radians(lat1), math.radians(lat2)
        dlat = math.radians(lat2 - lat1)
        dlon = math.radians(lon2 - lon1)

        a = (math.sin(dlat / 2) ** 2 +
             math.cos(lat1_r) * math.cos(lat2_r) * math.sin(dlon / 2) ** 2)
        c = 2 * math.asin(math.sqrt(a))

        return R * c

    def build_from_trip_data(self, trip_df, min_trips=3, verbose=True):
        """
        Build the transportation graph from trip data.

        Aggregates trips between zone pairs to create weighted edges.
        Filters out rarely-traveled routes (fewer than min_trips).

        Args:
            trip_df: DataFrame with PULocationID, DOLocationID,
                     trip_distance, trip_duration_min columns.
            min_trips: Minimum trips required to create an edge.
            verbose: Print construction progress.

        Returns:
            NetworkX DiGraph.
        """
        if verbose:
            print(f"\n{'=' * 60}")
            print("BUILDING TRANSPORTATION NETWORK GRAPH")
            print(f"{'=' * 60}")

        # Create directed graph
        self.graph = nx.DiGraph()

        # Add nodes with geographic attributes
        for zone_id, (name, lat, lon) in self.zone_info.items():
            self.graph.add_node(
                zone_id,
                name=name,
                lat=lat,
                lon=lon,
                pos=(lon, lat),  # For visualization (x, y)
            )

        if verbose:
            print(f"  Added {self.graph.number_of_nodes()} zone nodes")

        # Aggregate edge data from trips
        edge_data = (
            trip_df.groupby(["PULocationID", "DOLocationID"])
            .agg(
                avg_distance=("trip_distance", "mean"),
                avg_duration=("trip_duration_min", "mean"),
                trip_count=("trip_distance", "count"),
                min_distance=("trip_distance", "min"),
                max_distance=("trip_distance", "max"),
            )
            .reset_index()
        )

        # Filter by minimum trip count
        edge_data = edge_data[edge_data["trip_count"] >= min_trips]

        # Add edges with weights
        for _, row in edge_data.iterrows():
            src = int(row["PULocationID"])
            dst = int(row["DOLocationID"])

            # Only add edges between known zones
            if src in self.zone_info and dst in self.zone_info:
                # Compute Haversine distance as reference
                lat1, lon1 = self.zone_info[src][1], self.zone_info[src][2]
                lat2, lon2 = self.zone_info[dst][1], self.zone_info[dst][2]
                haversine_km = self.haversine_distance(lat1, lon1, lat2, lon2)

                self.graph.add_edge(
                    src, dst,
                    distance=round(row["avg_distance"] * 1.60934, 2),  # miles → km
                    duration=round(row["avg_duration"], 2),  # minutes
                    trip_count=int(row["trip_count"]),
                    haversine_km=round(haversine_km, 2),
                    # Normalized weight for optimization (distance in km)
                    weight=round(row["avg_distance"] * 1.60934, 2),
                )

        # Compute graph statistics
        self._compute_statistics()

        if verbose:
            self._print_statistics()

        return self.graph

    def build_complete_graph(self, verbose=True):
        """
        Build a complete graph connecting all zones by geographic proximity.

        Useful when trip data doesn't cover all possible connections.
        Connects zones within 10km of each other.

        Args:
            verbose: Print construction progress.

        Returns:
            NetworkX DiGraph.
        """
        if verbose:
            print(f"\nBuilding complete geographic graph...")

        self.graph = nx.DiGraph()

        # Add nodes
        for zone_id, (name, lat, lon) in self.zone_info.items():
            self.graph.add_node(
                zone_id, name=name, lat=lat, lon=lon, pos=(lon, lat)
            )

        # Connect all pairs within 10km
        zone_ids = list(self.zone_info.keys())
        for i in range(len(zone_ids)):
            for j in range(len(zone_ids)):
                if i == j:
                    continue
                src, dst = zone_ids[i], zone_ids[j]
                lat1, lon1 = self.zone_info[src][1], self.zone_info[src][2]
                lat2, lon2 = self.zone_info[dst][1], self.zone_info[dst][2]

                dist_km = self.haversine_distance(lat1, lon1, lat2, lon2)

                if dist_km <= 10:
                    # Estimate duration based on default speed
                    duration = (dist_km / ROUTE_CONFIG["default_speed_kmh"]) * 60

                    self.graph.add_edge(
                        src, dst,
                        distance=round(dist_km, 2),
                        duration=round(duration, 2),
                        weight=round(dist_km, 2),
                        haversine_km=round(dist_km, 2),
                        trip_count=0,
                    )

        self._compute_statistics()
        if verbose:
            self._print_statistics()

        return self.graph

    def add_demand_to_nodes(self, demand_df):
        """
        Add demand information as node attributes.

        Args:
            demand_df: DataFrame with zone_id and demand columns.
        """
        if self.graph is None:
            raise RuntimeError("Graph not built yet. Call build_from_trip_data first.")

        zone_demand = demand_df.groupby("zone_id")["demand"].sum().to_dict()
        max_demand = max(zone_demand.values()) if zone_demand else 1

        for node in self.graph.nodes():
            demand = zone_demand.get(node, 0)
            self.graph.nodes[node]["demand"] = demand
            self.graph.nodes[node]["demand_normalized"] = demand / max_demand

    def _compute_statistics(self):
        """Compute and cache graph statistics."""
        if self.graph is None:
            return

        self.stats = {
            "num_nodes": self.graph.number_of_nodes(),
            "num_edges": self.graph.number_of_edges(),
            "density": nx.density(self.graph),
            "avg_degree": sum(dict(self.graph.degree()).values()) / max(self.graph.number_of_nodes(), 1),
            "is_connected": nx.is_weakly_connected(self.graph) if self.graph.is_directed() else nx.is_connected(self.graph),
        }

        # Edge weight statistics
        if self.graph.number_of_edges() > 0:
            distances = [d.get("distance", 0) for _, _, d in self.graph.edges(data=True)]
            durations = [d.get("duration", 0) for _, _, d in self.graph.edges(data=True)]
            self.stats["avg_edge_distance_km"] = round(np.mean(distances), 2)
            self.stats["avg_edge_duration_min"] = round(np.mean(durations), 2)
            self.stats["max_edge_distance_km"] = round(max(distances), 2)
            self.stats["total_network_distance_km"] = round(sum(distances), 2)

        # Connected components
        if self.graph.is_directed():
            components = list(nx.weakly_connected_components(self.graph))
        else:
            components = list(nx.connected_components(self.graph))
        self.stats["num_components"] = len(components)
        self.stats["largest_component_size"] = len(max(components, key=len)) if components else 0

    def _print_statistics(self):
        """Print graph statistics."""
        print(f"\n  Graph Statistics:")
        print(f"  ├── Nodes: {self.stats['num_nodes']}")
        print(f"  ├── Edges: {self.stats['num_edges']}")
        print(f"  ├── Density: {self.stats['density']:.4f}")
        print(f"  ├── Avg Degree: {self.stats['avg_degree']:.1f}")
        print(f"  ├── Weakly Connected: {self.stats['is_connected']}")
        print(f"  ├── Connected Components: {self.stats['num_components']}")
        if "avg_edge_distance_km" in self.stats:
            print(f"  ├── Avg Edge Distance: {self.stats['avg_edge_distance_km']} km")
            print(f"  ├── Avg Edge Duration: {self.stats['avg_edge_duration_min']} min")
            print(f"  └── Total Network Distance: {self.stats['total_network_distance_km']} km")

    def get_node_info(self, node_id):
        """Get detailed information about a specific node."""
        if self.graph is None or node_id not in self.graph:
            return None
        data = dict(self.graph.nodes[node_id])
        data["in_degree"] = self.graph.in_degree(node_id)
        data["out_degree"] = self.graph.out_degree(node_id)
        data["neighbors"] = list(self.graph.successors(node_id))
        return data

    def get_edge_info(self, src, dst):
        """Get detailed information about a specific edge."""
        if self.graph is None or not self.graph.has_edge(src, dst):
            return None
        return dict(self.graph.edges[src, dst])

    def get_largest_component_subgraph(self):
        """Return the subgraph of the largest weakly connected component."""
        if self.graph is None:
            return None

        if self.graph.is_directed():
            components = list(nx.weakly_connected_components(self.graph))
        else:
            components = list(nx.connected_components(self.graph))

        largest = max(components, key=len)
        return self.graph.subgraph(largest).copy()
