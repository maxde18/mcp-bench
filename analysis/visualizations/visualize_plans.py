#!/usr/bin/env python3
"""
Visualize DAG plans from a planning results JSON file.

Generates one PNG per repetition per task, showing each tool node with its
full metadata (tool name, description, parameters, condition) and dependency
edges in a top-down hierarchical layout.

Usage:
    python analysis/visualizations/visualize_plans.py
    python analysis/visualizations/visualize_plans.py path/to/results.json

Output saved alongside this script in the same directory.
"""

import json
import sys
import textwrap
from pathlib import Path
from collections import defaultdict, deque

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import networkx as nx
import numpy as np


# ---------------------------------------------------------------------------
# Color palette — one colour per server
# ---------------------------------------------------------------------------

_PALETTE = [
    "#4C72B0",  # blue
    "#DD8452",  # orange
    "#55A868",  # green
    "#C44E52",  # red
    "#8172B3",  # purple
    "#64B5CD",  # teal
    "#DA8BC3",  # pink
    "#CCB974",  # gold
    "#937860",  # brown
    "#8C8C8C",  # grey
]

BG_DARK = "#12122a"
BG_PANEL = "#1c1c3a"
EDGE_COLOR = "#8888aa"
TITLE_COLOR = "#e0e0f0"
LEGEND_BG = "#2a2a4e"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _server_from_tool(tool: str) -> str:
    """Extract server prefix from 'Server:func', 'Server__func', or bare name."""
    for sep in (":", "__"):
        if sep in tool:
            return tool.split(sep, 1)[0].strip()
    return tool


def _func_from_tool(tool: str) -> str:
    """Extract function part from tool string."""
    for sep in (":", "__"):
        if sep in tool:
            return tool.split(sep, 1)[1].strip()
    return tool


def _assign_colors(nodes: list[dict]) -> dict[str, str]:
    servers = sorted({_server_from_tool(n["tool"]) for n in nodes})
    return {s: _PALETTE[i % len(_PALETTE)] for i, s in enumerate(servers)}


def _wrap(text: str, width: int = 30) -> str:
    """Wrap text to given width, returning newline-joined string."""
    return "\n".join(textwrap.wrap(text, width=width))


# ---------------------------------------------------------------------------
# Hierarchical layout (no graphviz required)
# ---------------------------------------------------------------------------

def _hierarchical_layout(G: nx.DiGraph, x_gap: float = 3.0, y_gap: float = 2.8) -> dict:
    """
    Assign (x, y) positions using topological generations.
    Within each layer, nodes are spread evenly and sorted by their mean
    predecessor x-position to reduce edge crossings.
    """
    if len(G) == 0:
        return {}

    try:
        layers = list(nx.topological_generations(G))
    except nx.NetworkXUnfeasible:
        return nx.spring_layout(G, seed=42, k=2.5)

    # Assign initial x based on index in layer
    pos: dict = {}
    for depth, layer in enumerate(layers):
        layer = sorted(layer)  # deterministic initial order
        n = len(layer)
        for j, node in enumerate(layer):
            pos[node] = [(j - (n - 1) / 2.0) * x_gap, -depth * y_gap]

    # One pass: reorder each layer by mean x of predecessors to cut crossings
    for depth, layer in enumerate(layers):
        if depth == 0:
            continue
        scores = {}
        for node in layer:
            preds = list(G.predecessors(node))
            if preds:
                scores[node] = np.mean([pos[p][0] for p in preds if p in pos])
            else:
                scores[node] = 0.0
        ordered = sorted(layer, key=lambda n: scores.get(n, 0.0))
        n = len(ordered)
        for j, node in enumerate(ordered):
            pos[node][0] = (j - (n - 1) / 2.0) * x_gap

    return {k: tuple(v) for k, v in pos.items()}


# ---------------------------------------------------------------------------
# Node label builder
# ---------------------------------------------------------------------------

def _build_label(node: dict, server_line: bool = True) -> str:
    """Build a multi-line label for a DAG node."""
    nid = node["id"]
    func = _func_from_tool(node["tool"])
    server = _server_from_tool(node["tool"])

    # Header: id + function name
    header = f"#{nid}  {func}"

    lines = [header]

    if server_line:
        lines.append(f"[{server}]")

    # Description (wrapped)
    desc = node.get("description", "")
    if desc:
        lines.append("")
        for wrapped_line in textwrap.wrap(desc, width=32):
            lines.append(wrapped_line)

    # Parameters
    params = node.get("parameters", {})
    if params:
        lines.append("")
        lines.append("params:")
        for k, v in list(params.items())[:4]:
            v_str = str(v)
            if len(v_str) > 24:
                v_str = v_str[:21] + "…"
            entry = f"  {k}={v_str}"
            if len(entry) > 36:
                entry = entry[:35] + "…"
            lines.append(entry)
        if len(params) > 4:
            lines.append(f"  (+{len(params) - 4} more params)")

    # Condition
    cond = node.get("condition", "")
    if cond:
        lines.append("")
        cond_short = cond if len(cond) <= 36 else cond[:33] + "…"
        lines.append(f"if: {cond_short}")

    # Filter
    filt = node.get("filter", "")
    if filt:
        filt_short = filt if len(filt) <= 36 else filt[:33] + "…"
        lines.append(f"filter: {filt_short}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Single DAG drawing
# ---------------------------------------------------------------------------

def _draw_single_dag(
    ax: plt.Axes,
    nodes: list[dict],
    title: str,
    server_colors: dict[str, str],
) -> None:
    if not nodes:
        ax.text(0.5, 0.5, "No nodes", ha="center", va="center",
                transform=ax.transAxes, color="white")
        ax.set_title(title, color=TITLE_COLOR, fontsize=9, fontweight="bold")
        ax.axis("off")
        return

    # Build graph over node IDs
    G = nx.DiGraph()
    id_map = {n["id"]: n for n in nodes}
    for node in nodes:
        G.add_node(node["id"])
    for node in nodes:
        for dep in node.get("depends_on", []):
            if dep in id_map:
                G.add_edge(dep, node["id"])

    pos = _hierarchical_layout(G)

    # --- Draw edges first (behind nodes) ---
    for u, v in G.edges():
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        ax.annotate(
            "",
            xy=(x1, y1),
            xytext=(x0, y0),
            arrowprops=dict(
                arrowstyle="->",
                color=EDGE_COLOR,
                lw=1.4,
                connectionstyle="arc3,rad=0.06",
                shrinkA=18,
                shrinkB=18,
            ),
            zorder=1,
        )

    # --- Draw nodes ---
    for node in nodes:
        nid = node["id"]
        x, y = pos[nid]
        server = _server_from_tool(node["tool"])
        color = server_colors.get(server, "#888888")
        label = _build_label(node, server_line=False)

        # Header line gets bold via splitting at first newline
        ax.text(
            x, y,
            label,
            ha="center",
            va="center",
            fontsize=6.5,
            color="white",
            bbox=dict(
                boxstyle="round,pad=0.5",
                facecolor=color,
                alpha=0.90,
                edgecolor="white",
                linewidth=1.2,
            ),
            zorder=3,
            multialignment="center",
            linespacing=1.35,
        )

    ax.set_title(title, color=TITLE_COLOR, fontsize=8.5, fontweight="bold",
                 pad=10, loc="left")
    ax.set_facecolor(BG_PANEL)
    ax.axis("off")

    # Auto-scale axes with padding
    if pos:
        xs = [p[0] for p in pos.values()]
        ys = [p[1] for p in pos.values()]
        pad_x = max(1.5, (max(xs) - min(xs)) * 0.15) if len(xs) > 1 else 2.0
        pad_y = max(1.5, (max(ys) - min(ys)) * 0.20) if len(ys) > 1 else 2.0
        ax.set_xlim(min(xs) - pad_x, max(xs) + pad_x)
        ax.set_ylim(min(ys) - pad_y, max(ys) + pad_y)


# ---------------------------------------------------------------------------
# Per-repetition figure
# ---------------------------------------------------------------------------

def _make_rep_figure(
    task: dict,
    rep: dict,
    model: str,
    server_colors: dict[str, str],
    output_dir: Path,
) -> Path:
    task_id = task.get("task_id", "unknown")
    server_name = task.get("server_name", "")
    rep_num = rep.get("repetition", 0)
    temp = rep.get("temperature")
    nodes = rep.get("generated_plan", {}).get("nodes", [])
    val = rep.get("validation", {})
    tok = rep.get("token_usage", {})

    n_nodes = len(nodes)
    # Scale figure: wider for more nodes, taller for deeper graphs
    depth = max(
        (len(list(nx.topological_generations(
            _build_graph_from_nodes(nodes)
        ))) if nodes else 1),
        1
    )
    fig_w = max(14, n_nodes * 2.2)
    fig_h = max(9, depth * 3.0 + 3)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h), facecolor=BG_DARK)

    temp_str = f"temp={temp}" if temp is not None else "temp=default"
    valid_badge = "VALID ✓" if val.get("is_valid") else "INVALID ✗"
    title = (
        f"model: {model}   task: {task_id}   servers: {server_name}\n"
        f"rep {rep_num}   {temp_str}   nodes: {n_nodes}   "
        f"{valid_badge}   "
        f"tokens: {tok.get('total_tokens','?')} "
        f"(prompt {tok.get('prompt_tokens','?')} + "
        f"completion {tok.get('completion_tokens','?')})"
    )

    _draw_single_dag(ax, nodes, title, server_colors)

    # Server colour legend
    legend_handles = [
        mpatches.Patch(facecolor=c, edgecolor="white", linewidth=0.8, label=s)
        for s, c in sorted(server_colors.items())
    ]
    leg = ax.legend(
        handles=legend_handles,
        loc="upper right",
        fontsize=7.5,
        framealpha=0.85,
        facecolor=LEGEND_BG,
        edgecolor="#6666aa",
        labelcolor="white",
        title="Servers",
        title_fontsize=8,
    )
    leg.get_title().set_color("white")

    plt.tight_layout(pad=1.5)

    fname = f"{task_id}_rep{rep_num:02d}.png"
    out_path = output_dir / fname
    fig.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    return out_path


def _build_graph_from_nodes(nodes: list[dict]) -> nx.DiGraph:
    G = nx.DiGraph()
    id_map = {n["id"] for n in nodes}
    for node in nodes:
        G.add_node(node["id"])
    for node in nodes:
        for dep in node.get("depends_on", []):
            if dep in id_map:
                G.add_edge(dep, node["id"])
    return G


# ---------------------------------------------------------------------------
# Overview figure — all repetitions in a grid
# ---------------------------------------------------------------------------

def _make_overview_figure(
    task: dict,
    reps: list[dict],
    model: str,
    server_colors: dict[str, str],
    output_dir: Path,
) -> Path:
    task_id = task.get("task_id", "unknown")
    server_name = task.get("server_name", "")
    n = len(reps)

    cols = min(4, n)
    rows = (n + cols - 1) // cols

    fig_w = cols * 9
    fig_h = rows * 7 + 1.5

    fig, axes = plt.subplots(rows, cols, figsize=(fig_w, fig_h),
                             facecolor=BG_DARK)
    axes = np.array(axes).reshape(-1)  # flatten

    for i, rep in enumerate(reps):
        ax = axes[i]
        rep_num = rep.get("repetition", 0)
        temp = rep.get("temperature")
        nodes = rep.get("generated_plan", {}).get("nodes", [])
        val = rep.get("validation", {})
        tok = rep.get("token_usage", {})
        temp_str = f"T={temp}" if temp is not None else "T=default"
        valid_badge = "✓" if val.get("is_valid") else "✗"
        title = (
            f"rep {rep_num}  {temp_str}  nodes={len(nodes)}  "
            f"{valid_badge}  tok={tok.get('total_tokens','?')}"
        )
        _draw_single_dag(ax, nodes, title, server_colors)

    # Hide unused axes
    for j in range(n, len(axes)):
        axes[j].set_visible(False)

    # Super-title
    fig.suptitle(
        f"Model: {model}  |  Task: {task_id}  |  Servers: {server_name}\n"
        f"All {n} repetitions overview",
        color=TITLE_COLOR, fontsize=11, fontweight="bold", y=1.01,
    )

    # Shared legend
    legend_handles = [
        mpatches.Patch(facecolor=c, edgecolor="white", linewidth=0.8, label=s)
        for s, c in sorted(server_colors.items())
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=len(server_colors),
        fontsize=8,
        framealpha=0.85,
        facecolor=LEGEND_BG,
        edgecolor="#6666aa",
        labelcolor="white",
        bbox_to_anchor=(0.5, -0.02),
    )

    plt.tight_layout(pad=1.2)

    fname = f"{task_id}_overview.png"
    out_path = output_dir / fname
    fig.savefig(out_path, dpi=130, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def visualize_results(json_path: Path, output_dir: Path) -> None:
    print(f"Loading: {json_path}")
    with open(json_path) as f:
        data = json.load(f)

    model = data.get("model", "unknown")
    runs = data.get("runs", [])

    print(f"Model: {model}  |  Tasks: {len(runs)}")

    for task in runs:
        task_id = task.get("task_id", "unknown")
        reps = [r for r in task.get("repetitions", [])
                if r.get("status") == "success"]

        if not reps:
            print(f"  [skip] {task_id} — no successful repetitions")
            continue

        print(f"\n  Task: {task_id}  ({len(reps)} successful reps)")

        # Collect all nodes across all reps for consistent colour assignment
        all_nodes = []
        for rep in reps:
            all_nodes.extend(rep.get("generated_plan", {}).get("nodes", []))
        server_colors = _assign_colors(all_nodes)

        # Individual rep figures
        for rep in reps:
            nodes = rep.get("generated_plan", {}).get("nodes", [])
            if not nodes:
                continue
            out = _make_rep_figure(task, rep, model, server_colors, output_dir)
            print(f"    saved: {out.name}")

        # Overview grid (all reps on one canvas)
        out = _make_overview_figure(task, reps, model, server_colors, output_dir)
        print(f"    saved: {out.name}  (overview)")

    print(f"\nAll images written to: {output_dir}")


if __name__ == "__main__":
    here = Path(__file__).parent

    if len(sys.argv) > 1:
        json_path = Path(sys.argv[1])
    else:
        # Default: the latest minimax-m2.7 result copied into this directory
        candidates = sorted(here.glob("*.json"))
        if not candidates:
            print("No JSON files found in", here)
            sys.exit(1)
        json_path = candidates[-1]
        print(f"No path given — using: {json_path.name}")

    import datetime
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = json_path.stem.replace(" ", "_")
    output_dir = here / f"{stem}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    visualize_results(json_path, output_dir)
