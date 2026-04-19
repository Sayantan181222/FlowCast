"""
Route Optimizer Page — Module 3 UI.

Interactive route planning with algorithm selection, multi-stop optimization,
and network visualization.
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from config import DATA_CONFIG, NYC_ZONES
from ui.styles import section_header
from ui.components import render_metric_row, render_header


def render_route_optimizer():
    """Render the route optimization page."""
    render_header("Route Optimizer", "🗺️", "Module 3 — Find optimal routes using graph-based algorithms")

    # Build graph if needed
    if "transport_graph" not in st.session_state:
        _build_graph()

    graph = st.session_state.get("transport_graph")
    optimizer = st.session_state.get("route_optimizer")
    visualizer = st.session_state.get("network_visualizer")

    if graph is None:
        st.warning("⚠️ Transportation graph not available. Generate data first in Data Explorer.")
        return

    # Graph stats
    g_stats = st.session_state.get("graph_stats", {})
    st.markdown(section_header("📊 Network Statistics", "Transportation graph overview"), unsafe_allow_html=True)

    render_metric_row([
        ("🔵", str(g_stats.get("num_nodes", 0)), "Zones (Nodes)"),
        ("🔗", str(g_stats.get("num_edges", 0)), "Connections (Edges)"),
        ("📏", f"{g_stats.get('avg_edge_distance_km', 0):.1f} km", "Avg Edge Distance"),
        ("⏱️", f"{g_stats.get('avg_edge_duration_min', 0):.1f} min", "Avg Edge Duration"),
    ])

    st.markdown("<br>", unsafe_allow_html=True)

    # Network Visualization
    st.markdown(section_header("🌐 Network Map"), unsafe_allow_html=True)
    if visualizer:
        demand_df = st.session_state.get("demand_zone_df")
        if demand_df is not None:
            fig = visualizer.plot_demand_heatmap(demand_df)
        else:
            fig = visualizer.plot_network(color_by="degree")
        st.plotly_chart(fig, width="stretch")

    st.markdown("---")

    # Route Planning Interface
    st.markdown(section_header("🚗 Route Planning", "Select source and destination zones"), unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["🔀 Single Route", "📍 Multi-Stop", "⚡ Algorithm Comparison"])

    zone_options = {f"{zid} - {info[0]}": zid for zid, info in sorted(NYC_ZONES.items(), key=lambda x: x[1][0])}
    zone_list = list(zone_options.keys())

    with tab1:
        _render_single_route(zone_list, zone_options, optimizer, visualizer)

    with tab2:
        _render_multi_stop(zone_list, zone_options, optimizer, visualizer)

    with tab3:
        _render_comparison(zone_list, zone_options, optimizer, visualizer, graph)


def _build_graph():
    """Build or rebuild the transportation graph."""
    from module3_routing.graph_builder import TransportationGraphBuilder
    from module3_routing.route_optimizer import RouteOptimizer
    from module3_routing.network_visualizer import NetworkVisualizer

    builder = TransportationGraphBuilder()

    # Prefer raw data (has proper datetime columns) for graph building
    if os.path.exists(DATA_CONFIG["raw_data_file"]):
        df = pd.read_csv(DATA_CONFIG["raw_data_file"], parse_dates=["pickup_datetime", "dropoff_datetime"])
        graph = builder.build_from_trip_data(df, verbose=False)

        # Add demand data if available
        try:
            from module1_data.data_preprocessor import DataPreprocessor
            preprocessor = DataPreprocessor()
            df_featured = preprocessor.engineer_features(df, verbose=False)
            demand_zone = preprocessor.aggregate_demand(df_featured, verbose=False)
            builder.add_demand_to_nodes(demand_zone)
            st.session_state["demand_zone_df"] = demand_zone
        except Exception:
            pass  # Demand overlay is optional

    else:
        graph = builder.build_complete_graph(verbose=False)

    # Use largest connected component for routing
    graph = builder.get_largest_component_subgraph()

    st.session_state["transport_graph"] = graph
    st.session_state["graph_stats"] = builder.stats
    st.session_state["route_optimizer"] = RouteOptimizer(graph)
    st.session_state["network_visualizer"] = NetworkVisualizer(graph)


def _render_single_route(zone_list, zone_options, optimizer, visualizer):
    """Render single source-destination route planning."""
    col1, col2 = st.columns(2)
    with col1:
        source_name = st.selectbox("🟢 Source Zone", zone_list, index=0, key="sr_source")
    with col2:
        target_idx = min(5, len(zone_list) - 1)
        target_name = st.selectbox("🔴 Destination Zone", zone_list, index=target_idx, key="sr_target")

    source = zone_options[source_name]
    target = zone_options[target_name]

    algorithm = st.radio("Algorithm", ["Dijkstra", "A*", "Bellman-Ford"],
                         horizontal=True, key="sr_algo")

    weight = st.radio("Optimize for", ["distance", "duration"],
                      horizontal=True, key="sr_weight")

    if st.button("🔍 Find Route", key="sr_find", width="stretch"):
        if source == target:
            st.error("Source and destination must be different!")
            return

        with st.spinner(f"Running {algorithm}..."):
            if algorithm == "Dijkstra":
                result = optimizer.dijkstra(source, target, weight)
            elif algorithm == "A*":
                result = optimizer.astar(source, target, weight)
            else:
                result = optimizer.bellman_ford(source, target, weight)

        if result.success:
            render_metric_row([
                ("📏", f"{result.total_distance_km:.2f} km", "Total Distance"),
                ("⏱️", f"{result.total_duration_min:.1f} min", "Est. Duration"),
                ("🔢", str(len(result.path) - 1), "Hops"),
                ("⚡", f"{result.computation_time_ms:.3f} ms", "Comp. Time"),
            ])

            st.markdown("<br>", unsafe_allow_html=True)

            # Route visualization
            fig = visualizer.plot_route(result, title=f"Route via {algorithm}")
            st.plotly_chart(fig, width="stretch")

            # Route details
            with st.expander("📋 Route Details"):
                route_data = []
                for i, (zone_id, name) in enumerate(zip(result.path, result.path_names)):
                    label = "🟢 Start" if i == 0 else ("🔴 End" if i == len(result.path) - 1 else f"▶ Step {i}")
                    route_data.append({"Step": label, "Zone": name, "Zone ID": zone_id})
                st.dataframe(pd.DataFrame(route_data), width="stretch")
        else:
            st.error(f"❌ {result.message}")


def _render_multi_stop(zone_list, zone_options, optimizer, visualizer):
    """Render multi-stop route optimization."""
    st.markdown("Select multiple zones to visit. The optimizer will find the best order and route.")

    selected = st.multiselect(
        "📍 Select Stop Zones (2-10 zones)",
        zone_list,
        default=zone_list[:3] if len(zone_list) >= 3 else zone_list[:2],
        key="ms_stops",
    )

    if len(selected) < 2:
        st.warning("Select at least 2 zones.")
        return

    if len(selected) > 10:
        st.warning("Maximum 10 stops supported.")
        return

    start_zone = st.selectbox("🟢 Starting Zone", selected, key="ms_start")

    if st.button("🔍 Optimize Multi-Stop Route", key="ms_find", width="stretch"):
        stops = [zone_options[s] for s in selected]
        start = zone_options[start_zone]

        with st.spinner("Optimizing multi-stop route (NN + 2-opt)..."):
            result = optimizer.multi_stop_optimize(stops, start=start)

        if result.success:
            render_metric_row([
                ("📏", f"{result.total_distance_km:.2f} km", "Total Distance"),
                ("⏱️", f"{result.total_duration_min:.1f} min", "Est. Duration"),
                ("📍", str(len(selected)), "Stops"),
                ("⚡", f"{result.computation_time_ms:.3f} ms", "Comp. Time"),
            ])

            st.markdown("<br>", unsafe_allow_html=True)

            fig = visualizer.plot_route(result, title="Multi-Stop Optimized Route")
            st.plotly_chart(fig, width="stretch")

            with st.expander("📋 Optimized Route Order"):
                route_data = []
                for i, (zone_id, name) in enumerate(zip(result.path, result.path_names)):
                    label = "🟢 Start" if i == 0 else ("🔴 End" if i == len(result.path) - 1 else f"▶ {i}")
                    route_data.append({"Step": label, "Zone": name, "Zone ID": zone_id})
                st.dataframe(pd.DataFrame(route_data), width="stretch")
        else:
            st.error(f"❌ {result.message}")


def _render_comparison(zone_list, zone_options, optimizer, visualizer, graph):
    """Render algorithm comparison."""
    col1, col2 = st.columns(2)
    with col1:
        source_name = st.selectbox("🟢 Source Zone", zone_list, index=0, key="cmp_source")
    with col2:
        target_idx = min(10, len(zone_list) - 1)
        target_name = st.selectbox("🔴 Destination Zone", zone_list, index=target_idx, key="cmp_target")

    source = zone_options[source_name]
    target = zone_options[target_name]

    if st.button("⚡ Compare All Algorithms", key="cmp_run", width="stretch"):
        if source == target:
            st.error("Source and destination must be different!")
            return

        with st.spinner("Running algorithm comparison..."):
            results = optimizer.compare_algorithms(source, target, verbose=False)

        # Results table
        comparison_data = []
        for name, result in results.items():
            if result.success:
                from module4_evaluation.metrics import RouteMetrics
                efficiency = RouteMetrics.path_efficiency(result, graph)
                comparison_data.append({
                    "Algorithm": name,
                    "Distance (km)": round(result.total_distance_km, 3),
                    "Duration (min)": round(result.total_duration_min, 2),
                    "Hops": len(result.path) - 1,
                    "Comp. Time (ms)": round(result.computation_time_ms, 4),
                    "Efficiency": round(efficiency, 4),
                    "Optimal": "✅" if result.is_optimal else "❌",
                })

        if comparison_data:
            st.dataframe(pd.DataFrame(comparison_data), width="stretch")

            # Comparison chart
            fig = visualizer.plot_algorithm_comparison(results)
            st.plotly_chart(fig, width="stretch")

            # Show best route
            best_algo = min(
                [(name, r) for name, r in results.items() if r.success],
                key=lambda x: x[1].total_distance_km,
            )
            st.markdown(f"**🏆 Best Route: {best_algo[0]}**")
            fig = visualizer.plot_route(best_algo[1], title=f"Best Route — {best_algo[0]}")
            st.plotly_chart(fig, width="stretch")
