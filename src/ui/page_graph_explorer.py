"""Graph explorer page."""

from __future__ import annotations

import math
import random
from typing import Any

import numpy as np

from src.data.preprocessing import build_adjacency, iter_edge_pairs, make_scaled_split
from src.experiments.configs import SUPPORTED_SCALES
from src.ui.common import load_bundle, require_streamlit, split_count_rows

HARD_NODE_LIMIT = 4000
HARD_EDGE_LIMIT = 25000
FIXED_MAX_SAMPLED_EDGES = 6000
FIXED_LAYOUT_STEPS = 20
FIXED_NODE_SIZE = 6
FIXED_EDGE_WIDTH = 1


def _sample_edge_pairs(edges: Any, max_edges: int, seed: int) -> list[tuple[int, int]]:
    """Reservoir-sample edge pairs without loading all edges into memory."""
    rng = random.Random(seed)
    sample: list[tuple[int, int]] = []

    for edge_index, edge_pair in enumerate(iter_edge_pairs(edges)):
        if edge_index < max_edges:
            sample.append(edge_pair)
            continue

        replacement_index = rng.randint(0, edge_index)
        if replacement_index < max_edges:
            sample[replacement_index] = edge_pair

    return sample


def _build_graph_snapshot(
    dataset_scale: float,
    split_name: str,
    max_edges: int,
    seed: int,
) -> dict[str, Any]:
    bundle = load_bundle()
    scaled_split = make_scaled_split(
        split_edge=bundle.split_edge,
        scale=dataset_scale,
        seed=seed,
    )
    selected_split = scaled_split[split_name]
    sampled_edges = _sample_edge_pairs(
        edges=selected_split["edge"],
        max_edges=max_edges,
        seed=seed,
    )

    adjacency = build_adjacency(sampled_edges, make_undirected=True)
    degree_items = sorted(
        ((node_id, len(neighbors)) for node_id, neighbors in adjacency.items()),
        key=lambda item: item[1],
        reverse=True,
    )
    top_degree_rows = [
        {"node_id": node_id, "degree": degree}
        for node_id, degree in degree_items[:20]
    ]

    active_node_ids = [node_id for node_id, _ in degree_items]

    num_nodes = int(getattr(bundle.data, "num_nodes", 0))
    if num_nodes <= 0 and sampled_edges:
        num_nodes = max(max(source, target) for source, target in sampled_edges) + 1

    avg_degree = 0.0
    if adjacency:
        avg_degree = sum(len(neighbors) for neighbors in adjacency.values()) / len(adjacency)

    return {
        "num_nodes": num_nodes,
        "sampled_edge_count": len(sampled_edges),
        "active_node_count": len(adjacency),
        "avg_degree": avg_degree,
        "max_degree": degree_items[0][1] if degree_items else 0,
        "split_summary_rows": split_count_rows(scaled_split),
        "degree_values": [degree for _, degree in degree_items],
        "top_degree_rows": top_degree_rows,
        "active_node_ids": active_node_ids,
        "sampled_edges": sampled_edges,
        "adjacency": adjacency,
    }


def _degree_band(degree: int, band: str) -> bool:
    if band == "Tum node'lar":
        return True
    if band == "0-1 komsulu":
        return 0 <= degree <= 1
    if band == "2-3 komsulu":
        return 2 <= degree <= 3
    if band == "4-10 komsulu":
        return 4 <= degree <= 10
    if band == "11+ komsulu":
        return degree >= 11
    return True


def _compute_2d_layout(
    node_ids: list[int],
    adjacency: dict[int, set[int]],
    seed: int,
    layout_steps: int,
) -> dict[int, tuple[float, float]]:
    """Build deterministic 2D coordinates for interactive graph rendering."""
    if not node_ids:
        return {}

    rng = np.random.default_rng(seed)
    node_count = len(node_ids)
    index_by_node = {node_id: idx for idx, node_id in enumerate(node_ids)}
    positions = rng.uniform(-1.0, 1.0, size=(node_count, 2))

    edge_pairs: list[tuple[int, int]] = []
    for source in node_ids:
        for target in adjacency.get(source, set()):
            if source < target and target in index_by_node:
                edge_pairs.append((index_by_node[source], index_by_node[target]))

    if not edge_pairs:
        return {
            node_id: (float(positions[idx, 0]), float(positions[idx, 1]))
            for node_id, idx in index_by_node.items()
        }

    edges = np.asarray(edge_pairs, dtype=np.int64)
    ideal_length = 0.35
    attraction = 0.025
    center_pull = 0.01

    for _ in range(max(1, layout_steps)):
        displacement = np.zeros_like(positions)

        edge_delta = positions[edges[:, 0]] - positions[edges[:, 1]]
        edge_distance = np.sqrt(np.sum(edge_delta * edge_delta, axis=1) + 1e-6)
        stretch = edge_distance - ideal_length
        spring_force = (attraction * stretch / edge_distance)[:, None] * edge_delta
        np.add.at(displacement, edges[:, 0], -spring_force)
        np.add.at(displacement, edges[:, 1], spring_force)

        # Light center gravity keeps the layout bounded without O(N^2) repulsion.
        displacement -= center_pull * positions
        positions += 0.15 * displacement
        positions = np.clip(positions, -6.0, 6.0)

    return {
        node_id: (float(positions[idx, 0]), float(positions[idx, 1]))
        for node_id, idx in index_by_node.items()
    }


def _build_interactive_graph_view(
    node_ids: list[int],
    sampled_edges: list[tuple[int, int]],
    adjacency: dict[int, set[int]],
    seed: int,
    layout_steps: int,
    show_edges: bool,
    node_size: int,
    edge_width: int,
    color_mode: str,
) -> Any:
    import pydeck as pdk

    coordinates = _compute_2d_layout(
        node_ids=node_ids,
        adjacency=adjacency,
        seed=seed,
        layout_steps=layout_steps,
    )
    if not coordinates:
        return None

    degrees = {node_id: len(adjacency.get(node_id, set())) for node_id in node_ids}
    max_degree = max(degrees.values(), default=1)
    min_degree = min(degrees.values(), default=0)

    def degree_to_heat_color(degree: int) -> list[int]:
        """Map degree to a readable blue->green->yellow->orange->red palette."""
        if max_degree <= min_degree:
            return [44, 127, 184, 230]

        # Log normalization gives better visual spread for skewed degree distributions.
        norm = (math.log1p(degree) - math.log1p(min_degree)) / (
            math.log1p(max_degree) - math.log1p(min_degree)
        )
        norm = max(0.0, min(1.0, norm))

        color_stops: list[tuple[float, tuple[int, int, int]]] = [
            (0.00, (49, 54, 149)),   # deep blue
            (0.25, (69, 117, 180)),  # blue
            (0.50, (102, 189, 99)),  # green
            (0.70, (254, 224, 139)), # yellow
            (0.85, (244, 109, 67)),  # orange
            (1.00, (215, 48, 39)),   # red (highest degree)
        ]

        for stop_index in range(len(color_stops) - 1):
            left_pos, left_color = color_stops[stop_index]
            right_pos, right_color = color_stops[stop_index + 1]
            if left_pos <= norm <= right_pos:
                ratio = (norm - left_pos) / (right_pos - left_pos)
                red = int(left_color[0] + ratio * (right_color[0] - left_color[0]))
                green = int(left_color[1] + ratio * (right_color[1] - left_color[1]))
                blue = int(left_color[2] + ratio * (right_color[2] - left_color[2]))
                return [red, green, blue, 230]

        return [215, 48, 39, 230]

    node_rows = []
    for node_id in node_ids:
        degree = degrees[node_id]
        if color_mode == "degree":
            color = degree_to_heat_color(degree)
        elif color_mode == "component":
            bucket = node_id % 12
            color = [
                int(60 + (bucket * 17) % 180),
                int(70 + (bucket * 31) % 170),
                int(80 + (bucket * 43) % 160),
                220,
            ]
        else:
            color = [44, 127, 184, 220]

        node_rows.append(
            {
                "node_id": node_id,
                "x": coordinates[node_id][0],
                "y": coordinates[node_id][1],
                "color": color,
                "degree": degree,
            }
        )
    node_set = set(node_ids)
    edge_rows = []
    if show_edges:
        for source, target in sampled_edges:
            if source not in node_set or target not in node_set:
                continue
            source_xy = coordinates[source]
            target_xy = coordinates[target]
            edge_rows.append(
                {
                    "source": [source_xy[0], source_xy[1]],
                    "target": [target_xy[0], target_xy[1]],
                }
            )

    layers: list[Any] = []
    if edge_rows:
        layers.append(
            pdk.Layer(
                "LineLayer",
                data=edge_rows,
                get_source_position="source",
                get_target_position="target",
                get_color=[120, 120, 120, 120],
                get_width=edge_width,
                width_min_pixels=1,
            )
        )
    layers.append(
        pdk.Layer(
            "ScatterplotLayer",
            data=node_rows,
            get_position=["x", "y"],
            get_radius=node_size,
            radius_units="pixels",
            get_fill_color="color",
            pickable=True,
        )
    )

    xs = [row["x"] for row in node_rows]
    ys = [row["y"] for row in node_rows]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    center_x = (min_x + max_x) / 2.0
    center_y = (min_y + max_y) / 2.0
    span = max(max_x - min_x, max_y - min_y, 1e-6)
    # Heuristic fit: map span in "degree-like" space to a practical initial zoom.
    initial_zoom = max(1.0, min(15.0, math.log2(360.0 / span) - 1.0))

    return pdk.Deck(
        map_provider=None,
        initial_view_state=pdk.ViewState(
            longitude=center_x,
            latitude=center_y,
            zoom=initial_zoom,
            pitch=0,
            bearing=0,
        ),
        layers=layers,
        tooltip={"text": "node_id: {node_id}\ndegree: {degree}"},
    )


def render_graph_explorer() -> None:
    """Render graph-level dataset visualization tools."""
    st = require_streamlit()

    st.header("Graph Explorer")
    st.caption("Sade mod: sabit ayarlar + radio tabanli komsuluk/edge filtreleme.")

    controls_col1, controls_col2, controls_col3 = st.columns(3)
    dataset_scale = float(controls_col1.selectbox("Dataset scale", SUPPORTED_SCALES, index=2))
    split_name = controls_col2.selectbox("Split", ("train", "valid", "test"))
    color_mode = controls_col3.selectbox(
        "Node renklendirme",
        ("degree", "component", "single"),
        format_func=lambda value: {
            "degree": "Degree gradient",
            "component": "Pseudo-community palette",
            "single": "Single color",
        }[value],
    )

    radio_col1, radio_col2 = st.columns(2)
    neighbor_band = radio_col1.radio(
        "Komsuluk filtresi",
        ("Tum node'lar", "0-1 komsulu", "2-3 komsulu", "4-10 komsulu", "11+ komsulu"),
    )
    edge_mode = radio_col2.radio(
        "Edge gosterimi",
        ("Tum edge'ler", "Sadece filtrelenmis node edge'leri", "Edge kapali"),
    )

    seed = 42

    snapshot = _build_graph_snapshot(
        dataset_scale=dataset_scale,
        split_name=split_name,
        max_edges=FIXED_MAX_SAMPLED_EDGES,
        seed=seed,
    )

    degrees = {
        node_id: len(snapshot["adjacency"].get(node_id, set()))
        for node_id in snapshot["active_node_ids"]
    }
    render_node_ids = [
        node_id
        for node_id in snapshot["active_node_ids"]
        if _degree_band(degrees.get(node_id, 0), neighbor_band)
    ]
    if len(render_node_ids) > HARD_NODE_LIMIT:
        render_node_ids = render_node_ids[:HARD_NODE_LIMIT]
        st.warning(
            f"Stabilite icin en fazla {HARD_NODE_LIMIT} node render ediliyor. "
            "Daha fazla gorsel yogunluk icin max sampled edges degerini dusurebilirsin."
        )

    metric_columns = st.columns(5)
    metric_columns[0].metric("Total Nodes", snapshot["num_nodes"])
    metric_columns[1].metric("Sampled Edges", snapshot["sampled_edge_count"])
    metric_columns[2].metric("Active Nodes", snapshot["active_node_count"])
    metric_columns[3].metric("Avg Degree", f"{snapshot['avg_degree']:.2f}")
    metric_columns[4].metric("Max Degree", snapshot["max_degree"])

    st.subheader("2D Interactive Graph Canvas")
    st.caption(
        f"Render edilen node: {len(render_node_ids)} | "
        f"Node filtresi: {neighbor_band}"
    )

    render_node_set = set(render_node_ids)
    graph_edges = snapshot["sampled_edges"][:HARD_EDGE_LIMIT]
    if edge_mode == "Sadece filtrelenmis node edge'leri":
        graph_edges = [
            (source, target)
            for source, target in graph_edges
            if source in render_node_set and target in render_node_set
        ]
    if len(snapshot["sampled_edges"]) > HARD_EDGE_LIMIT:
        st.caption(f"Canvas edge limiti: {HARD_EDGE_LIMIT}")

    graph_deck = _build_interactive_graph_view(
        node_ids=render_node_ids,
        sampled_edges=graph_edges,
        adjacency=snapshot["adjacency"],
        seed=seed,
        layout_steps=FIXED_LAYOUT_STEPS,
        show_edges=edge_mode != "Edge kapali",
        node_size=FIXED_NODE_SIZE,
        edge_width=FIXED_EDGE_WIDTH,
        color_mode=color_mode,
    )
    if graph_deck is None:
        st.info("Filtreye uyan node bulunamadi.")
    else:
        st.pydeck_chart(graph_deck, width="stretch", height=760)

    st.subheader("Split-Aware Edge Composition")
    st.dataframe(snapshot["split_summary_rows"], width="stretch")
