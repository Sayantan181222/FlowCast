"""
Network Visualization Utilities.

Provides Plotly-based interactive visualizations of the transportation network:
- Full network graph with node/edge coloring
- Route highlighting on the network
- Demand heatmap overlays
- Algorithm comparison visualizations
"""

import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from config import NYC_ZONES


class NetworkVisualizer:
    """
    Interactive network visualization using Plotly.

    Creates rich, interactive maps of the transportation network with
    support for route overlays and demand-based coloring.
    """

    def __init__(self, graph):
        """
        Initialize visualizer with a transportation graph.

        Args:
            graph: NetworkX graph with lat/lon node attributes.
        """
        self.graph = graph
        self.zone_info = NYC_ZONES

    def plot_network(self, title="NYC Transportation Network",
                     show_edges=True, color_by="demand"):
        """
        Plot the full transportation network.

        Args:
            title: Plot title.
            show_edges: Whether to show edge lines.
            color_by: Node coloring attribute ('demand' or 'degree').

        Returns:
            Plotly Figure object.
        """
        fig = go.Figure()

        # Draw edges
        if show_edges:
            edge_x, edge_y = [], []
            for u, v in self.graph.edges():
                x0 = self.graph.nodes[u].get("lon", 0)
                y0 = self.graph.nodes[u].get("lat", 0)
                x1 = self.graph.nodes[v].get("lon", 0)
                y1 = self.graph.nodes[v].get("lat", 0)
                edge_x += [x0, x1, None]
                edge_y += [y0, y1, None]

            fig.add_trace(go.Scattergl(
                x=edge_x, y=edge_y,
                mode="lines",
                line=dict(width=0.5, color="rgba(150,150,150,0.3)"),
                hoverinfo="none",
                name="Roads",
            ))

        # Draw nodes
        node_x, node_y, node_text, node_sizes, node_colors = [], [], [], [], []
        for node in self.graph.nodes():
            data = self.graph.nodes[node]
            node_x.append(data.get("lon", 0))
            node_y.append(data.get("lat", 0))

            name = data.get("name", f"Zone {node}")
            demand = data.get("demand", 0)
            degree = self.graph.degree(node)

            node_text.append(f"{name}<br>Zone: {node}<br>Demand: {demand}<br>Connections: {degree}")

            if color_by == "demand":
                node_colors.append(data.get("demand", 0))
                node_sizes.append(max(5, min(20, data.get("demand_normalized", 0.1) * 20)))
            else:
                node_colors.append(degree)
                node_sizes.append(max(5, min(20, degree)))

        fig.add_trace(go.Scattergl(
            x=node_x, y=node_y,
            mode="markers",
            marker=dict(
                size=node_sizes,
                color=node_colors,
                colorscale="Viridis",
                colorbar=dict(title=color_by.capitalize()),
                line=dict(width=1, color="white"),
            ),
            text=node_text,
            hoverinfo="text",
            name="Zones",
        ))

        fig.update_layout(
            title=title,
            showlegend=True,
            xaxis=dict(title="Longitude", showgrid=False),
            yaxis=dict(title="Latitude", showgrid=False, scaleanchor="x"),
            template="plotly_dark",
            height=600,
            margin=dict(l=20, r=20, t=50, b=20),
        )

        return fig

    def plot_route(self, route_result, title=None):
        """
        Visualize a specific route on the network.

        Draws the full network in gray with the route highlighted in color.
        Shows step numbers and zone names along the route.

        Args:
            route_result: RouteResult object from RouteOptimizer.
            title: Override plot title.

        Returns:
            Plotly Figure object.
        """
        if not route_result.success or not route_result.path:
            fig = go.Figure()
            fig.add_annotation(text="No route found", xref="paper", yref="paper",
                              x=0.5, y=0.5, showarrow=False, font=dict(size=20))
            return fig

        path = route_result.path

        # Base network
        fig = self.plot_network(
            title=title or f"Route: {route_result.algorithm}",
            show_edges=True,
        )

        # Highlight route edges
        route_x, route_y = [], []
        for i in range(len(path) - 1):
            x0 = self.graph.nodes[path[i]].get("lon", 0)
            y0 = self.graph.nodes[path[i]].get("lat", 0)
            x1 = self.graph.nodes[path[i + 1]].get("lon", 0)
            y1 = self.graph.nodes[path[i + 1]].get("lat", 0)
            route_x += [x0, x1, None]
            route_y += [y0, y1, None]

        fig.add_trace(go.Scattergl(
            x=route_x, y=route_y,
            mode="lines",
            line=dict(width=4, color="#00ff88"),
            name="Route",
        ))

        # Highlight route nodes
        stop_x, stop_y, stop_text = [], [], []
        for i, node in enumerate(path):
            stop_x.append(self.graph.nodes[node].get("lon", 0))
            stop_y.append(self.graph.nodes[node].get("lat", 0))
            name = self.graph.nodes[node].get("name", f"Zone {node}")
            label = "START" if i == 0 else ("END" if i == len(path) - 1 else f"Step {i}")
            stop_text.append(f"{label}: {name}")

        fig.add_trace(go.Scattergl(
            x=stop_x, y=stop_y,
            mode="markers+text",
            marker=dict(size=12, color="#ff4444", line=dict(width=2, color="white")),
            text=[f"{i}" for i in range(len(path))],
            textposition="top center",
            textfont=dict(color="white", size=10),
            hovertext=stop_text,
            hoverinfo="text",
            name="Stops",
        ))

        # Add route info annotation
        info_text = (
            f"Distance: {route_result.total_distance_km:.2f} km | "
            f"Duration: {route_result.total_duration_min:.1f} min | "
            f"Stops: {len(path)}"
        )
        fig.add_annotation(
            text=info_text,
            xref="paper", yref="paper",
            x=0.5, y=-0.05,
            showarrow=False,
            font=dict(size=12, color="white"),
            bgcolor="rgba(0,0,0,0.7)",
            borderpad=4,
        )

        return fig

    def plot_algorithm_comparison(self, results_dict):
        """
        Create a comparison chart of algorithm performance.

        Args:
            results_dict: Dict of algorithm_name → RouteResult.

        Returns:
            Plotly Figure object.
        """
        algorithms = []
        distances = []
        durations = []
        times = []
        hops = []

        for name, result in results_dict.items():
            if result.success:
                algorithms.append(name)
                distances.append(result.total_distance_km)
                durations.append(result.total_duration_min)
                times.append(result.computation_time_ms)
                hops.append(len(result.path))

        fig = go.Figure()

        # Distance bars
        fig.add_trace(go.Bar(
            name="Distance (km)", x=algorithms, y=distances,
            marker_color="#4ecdc4", text=[f"{d:.2f}" for d in distances],
            textposition="auto",
        ))

        # Duration bars
        fig.add_trace(go.Bar(
            name="Duration (min)", x=algorithms, y=durations,
            marker_color="#ff6b6b", text=[f"{d:.1f}" for d in durations],
            textposition="auto",
        ))

        fig.update_layout(
            title="Algorithm Comparison — Route Metrics",
            barmode="group",
            template="plotly_dark",
            height=400,
            xaxis_title="Algorithm",
            yaxis_title="Value",
        )

        return fig

    def plot_demand_heatmap(self, demand_df):
        """
        Create a geographic heatmap of demand across zones.

        Args:
            demand_df: DataFrame with zone_id and demand columns.

        Returns:
            Plotly Figure object.
        """
        zone_demand = demand_df.groupby("zone_id")["demand"].sum().reset_index()

        lats, lons, demands, names = [], [], [], []
        for _, row in zone_demand.iterrows():
            zone_id = int(row["zone_id"])
            if zone_id in self.zone_info:
                name, lat, lon = self.zone_info[zone_id]
                lats.append(lat)
                lons.append(lon)
                demands.append(row["demand"])
                names.append(name)

        fig = go.Figure(go.Scattergl(
            x=lons, y=lats,
            mode="markers",
            marker=dict(
                size=[max(5, min(30, d / max(demands) * 30)) for d in demands],
                color=demands,
                colorscale="Hot",
                colorbar=dict(title="Total Demand"),
                opacity=0.8,
                line=dict(width=1, color="white"),
            ),
            text=[f"{n}<br>Demand: {d}" for n, d in zip(names, demands)],
            hoverinfo="text",
        ))

        fig.update_layout(
            title="Transportation Demand Heatmap",
            xaxis=dict(title="Longitude", showgrid=False),
            yaxis=dict(title="Latitude", showgrid=False, scaleanchor="x"),
            template="plotly_dark",
            height=600,
        )

        return fig
