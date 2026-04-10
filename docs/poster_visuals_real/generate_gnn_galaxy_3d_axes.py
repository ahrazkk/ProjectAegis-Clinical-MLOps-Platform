"""
Generate a clean 3D GNN galaxy render with visible X/Y/Z axes.
This script is intended for poster-friendly exports (light background).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = SCRIPT_DIR.parent.parent / "src" / "assets" / "gnn_real_data.json"
DEFAULT_OUTPUT = SCRIPT_DIR / "22_gnn_galaxy_3d_axes.png"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render GNN galaxy with 3D axes")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Path to gnn_real_data.json")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output PNG path")
    parser.add_argument("--max-edges", type=int, default=6000, help="Max number of edges to draw")
    parser.add_argument("--edge-length-percentile", type=float, default=90.0, help="Keep only edges shorter than this percentile")
    parser.add_argument("--node-size", type=float, default=3.4, help="Scatter marker size")
    parser.add_argument("--node-alpha", type=float, default=0.78, help="Node alpha")
    parser.add_argument("--node-color", type=str, default="#7dd3fc", help="Node color")
    parser.add_argument("--edge-alpha", type=float, default=0.03, help="Edge line alpha")
    parser.add_argument("--edge-width", type=float, default=0.18, help="Edge line width")
    parser.add_argument("--edge-color", type=str, default="#64748b", help="Edge line color")
    parser.add_argument("--dpi", type=int, default=300, help="Output DPI")
    parser.add_argument("--seed", type=int, default=42, help="Sampling RNG seed")
    parser.add_argument("--elev", type=float, default=22.0, help="3D camera elevation")
    parser.add_argument("--azim", type=float, default=35.0, help="3D camera azimuth")
    return parser.parse_args()


def load_graph(path: Path) -> tuple[np.ndarray, list[tuple[int, int]]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    nodes = data.get("nodes", [])
    adj = data.get("adj", None)
    edge_list = data.get("edges", None)

    xyz = []
    id_to_index: dict[str, int] = {}
    for node in nodes:
        node_id = str(node.get("id", len(id_to_index)))
        id_to_index[node_id] = len(id_to_index)

        if isinstance(node.get("position"), dict):
            p = node["position"]
            pos = [p.get("x", 0.0), p.get("y", 0.0), p.get("z", 0.0)]
        else:
            pos = node.get("pos", [0.0, 0.0, 0.0])

        if not isinstance(pos, list) or len(pos) < 3:
            pos = [0.0, 0.0, 0.0]
        xyz.append([float(pos[0]), float(pos[1]), float(pos[2])])

    # Format A: original gnn_real_data.json with adjacency dict
    if isinstance(adj, dict):
        return np.asarray(xyz, dtype=np.float64), unique_edges_from_adj(adj, id_to_index)

    # Format B: exported JSON with explicit edges list
    if isinstance(edge_list, list):
        return np.asarray(xyz, dtype=np.float64), unique_edges_from_list(edge_list, id_to_index)

    raise RuntimeError("Input file does not contain either 'adj' or 'edges' graph topology.")


def unique_edges_from_adj(adj: dict[str, list[str]], id_to_index: dict[str, int]) -> list[tuple[int, int]]:
    seen = set()
    edges: list[tuple[int, int]] = []

    for s, nbrs in adj.items():
        u = id_to_index.get(str(s))
        if u is None:
            continue

        for t in nbrs:
            v = id_to_index.get(str(t))
            if v is None:
                continue

            if u == v:
                continue

            a, b = (u, v) if u < v else (v, u)
            key = (a, b)
            if key in seen:
                continue
            seen.add(key)
            edges.append(key)

    return edges


def unique_edges_from_list(edge_list: list[dict], id_to_index: dict[str, int]) -> list[tuple[int, int]]:
    seen = set()
    edges: list[tuple[int, int]] = []

    for edge in edge_list:
        source = id_to_index.get(str(edge.get("source", "")))
        target = id_to_index.get(str(edge.get("target", "")))
        if source is None or target is None or source == target:
            continue

        a, b = (source, target) if source < target else (target, source)
        key = (a, b)
        if key in seen:
            continue
        seen.add(key)
        edges.append(key)

    return edges


def draw_axes(ax, center: np.ndarray, axis_len: float) -> None:
    cx, cy, cz = center.tolist()

    # X axis
    ax.plot([cx - axis_len, cx + axis_len], [cy, cy], [cz, cz], color="#dc2626", linewidth=2.2)
    # Y axis
    ax.plot([cx, cx], [cy - axis_len, cy + axis_len], [cz, cz], color="#16a34a", linewidth=2.2)
    # Z axis
    ax.plot([cx, cx], [cy, cy], [cz - axis_len, cz + axis_len], color="#2563eb", linewidth=2.2)

    ax.text(cx + axis_len * 1.05, cy, cz, "X", color="#dc2626", fontsize=11, weight="bold")
    ax.text(cx, cy + axis_len * 1.05, cz, "Y", color="#16a34a", fontsize=11, weight="bold")
    ax.text(cx, cy, cz + axis_len * 1.05, "Z", color="#2563eb", fontsize=11, weight="bold")


def edge_length(u: int, v: int, xyz: np.ndarray) -> float:
    dx = xyz[u, 0] - xyz[v, 0]
    dy = xyz[u, 1] - xyz[v, 1]
    dz = xyz[u, 2] - xyz[v, 2]
    return float(np.sqrt(dx * dx + dy * dy + dz * dz))


def select_clean_edges(
    edges: list[tuple[int, int]],
    xyz: np.ndarray,
    seed: int,
    max_edges: int,
    length_percentile: float,
) -> list[tuple[int, int]]:
    if not edges:
        return []

    lengths = np.array([edge_length(u, v, xyz) for u, v in edges], dtype=np.float64)
    p = float(np.clip(length_percentile, 1.0, 100.0))
    cutoff = float(np.percentile(lengths, p))

    short_edges = [e for e, d in zip(edges, lengths) if d <= cutoff]
    if len(short_edges) <= max_edges:
        return short_edges

    rng = np.random.default_rng(seed)
    idx = rng.choice(len(short_edges), size=max_edges, replace=False)
    return [short_edges[i] for i in idx]


def main() -> None:
    args = parse_args()

    xyz, edges = load_graph(args.input)
    if xyz.size == 0:
        raise RuntimeError(f"No node positions found in {args.input}")

    num_nodes = xyz.shape[0]

    sampled_edges = select_clean_edges(
        edges=edges,
        xyz=xyz,
        seed=args.seed,
        max_edges=max(0, int(args.max_edges)),
        length_percentile=args.edge_length_percentile,
    )

    center = xyz.mean(axis=0)
    span = xyz.max(axis=0) - xyz.min(axis=0)
    axis_len = float(np.max(span) * 0.58)

    fig = plt.figure(figsize=(11.5, 9.0), dpi=args.dpi)
    fig.patch.set_facecolor("white")
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor("white")

    # Light edge web
    for u, v in sampled_edges:
        if not (0 <= u < num_nodes and 0 <= v < num_nodes):
            continue
        x = [xyz[u, 0], xyz[v, 0]]
        y = [xyz[u, 1], xyz[v, 1]]
        z = [xyz[u, 2], xyz[v, 2]]
        ax.plot(x, y, z, color=args.edge_color, alpha=args.edge_alpha, linewidth=args.edge_width)

    # Node cloud
    ax.scatter(
        xyz[:, 0],
        xyz[:, 1],
        xyz[:, 2],
        c=args.node_color,
        s=args.node_size,
        alpha=args.node_alpha,
        linewidths=0.0,
    )

    draw_axes(ax, center, axis_len)

    ax.set_title("GNN Galaxy: Full Drug Interaction Topology (3D Embedding Space)", fontsize=13, pad=18)
    ax.set_xlabel("Embedding X", labelpad=15)
    ax.set_ylabel("Embedding Y", labelpad=15)
    ax.zaxis.set_rotate_label(False)
    ax.set_zlabel("", labelpad=25, rotation=90)

    # Move 2D fallback to act as the SOLE Z-axis label to avoid tick marker overlap
    ax.text2D(
        0.00,
        0.50,
        "Embedding Z",
        transform=ax.transAxes,
        rotation=90,
        va="center",
        ha="left",
        fontsize=11,
        color="#334155",
    )

    # Start configuring view
    # Keep proportions stable and readable
    max_range = float(np.max(span) / 2.0)
    cx, cy, cz = center.tolist()
    ax.set_xlim(cx - max_range, cx + max_range)
    ax.set_ylim(cy - max_range, cy + max_range)
    ax.set_zlim(cz - max_range, cz + max_range)

    ax.view_init(elev=args.elev, azim=args.azim)
    ax.grid(True, alpha=0.18)

    # White panes with subtle grid look
    for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
        axis.pane.set_facecolor((1.0, 1.0, 1.0, 1.0))
        axis.pane.set_edgecolor((0.85, 0.88, 0.92, 1.0))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print(f"Saved: {args.output}")
    print(f"Nodes: {num_nodes}")
    print(f"Edges total: {len(edges)}")
    print(f"Edges drawn: {len(sampled_edges)}")


if __name__ == "__main__":
    main()
