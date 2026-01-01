"""
Generate topology visualizations from NCCL benchmark summary.

Features
- Process all tests and G values automatically, or specify individual test/G
- Auto-organized output: topology/{test_name}/G{n}.png and topology/{test_name}/allG.png
- Respects directory structure (preserves with-debug/, without-debug/)
- Customizable layouts, colors, edge widths, and label positioning
- Optional automatic label adjustment to avoid overlaps

Behavior
- Default (--all): Process all tests and G values from CSV, organized by test_name
- --test specified: Process only that test, all G values
- --test and --g specified: Single image (backward compatible)

Usage
  # Process all tests and G values (recommended)
  python generate_topology.py --csv ./summary.csv --all

  # Process single test, all G values
  python generate_topology.py --csv ./summary.csv --test alltoall_perf

  # Single G image (backward compatible)
  python generate_topology.py --csv ./summary.csv --test alltoall_perf --g 1

  # With custom styling and label adjustment
  python generate_topology.py --csv ./summary.csv --all \
    --vmin 0 --vmax 80 --layout shell --adjust-labels

References
  NetworkX examples: https://networkx.org/documentation/stable/auto_examples/
  Colormaps: https://matplotlib.org/stable/users/explain/colors/colormaps.html
"""

import argparse
import math
import pathlib
import warnings
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import networkx.algorithms.community as nx_comm
import numpy as np
import pandas as pd
from adjustText import adjust_text
from matplotlib import cm
from matplotlib.colors import Normalize

# ===========================================================
# Constants
# ===========================================================

REQUIRED_COLS = {
    "file",
    "test",
    "N",
    "G",
    "nodes",
    "minBytes",
    "maxBytes",
    "avg_bus_bw_GBs",
    "peak_busbw_GBs",
}

DEFAULT_NODE_SIZE = 2400
DEFAULT_FONT_SIZE = 7
DEFAULT_EDGE_LABEL_FONT_SIZE = 9
DEFAULT_CMAP = "YlGn"
CMAP_VMIN_RATIO = 0.25  # Trim the lightest 25% of colormap for better visibility
CMAP_VMAX_RATIO = 1.0  # Keep the darkest colors

# ===========================================================
# Arg parsing & validation
# ===========================================================


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate professional topology image(s) from NCCL link CSV."
    )
    p.add_argument("--csv", required=True, help="Input CSV path.")
    p.add_argument(
        "--test",
        default=None,
        help="Test type to filter (e.g., alltoall_perf, sendrecv_perf). If omitted with --all, processes all tests.",
    )
    p.add_argument(
        "--g", type=int, default=None, help="GPU count G to filter (e.g., 1,2,4,8)."
    )
    p.add_argument(
        "--all",
        action="store_true",
        help="Process all tests and G values. Ignored if --test and --g are both specified.",
    )
    p.add_argument(
        "--out",
        default=None,
        help="Output PNG path for single-G mode. Deprecated in favor of --out-dir.",
    )
    p.add_argument(
        "--out-dir",
        default=None,
        help="Output directory for images. Default: ./topology or {csv_dir}/topology",
    )
    p.add_argument(
        "--combine-out",
        default=None,
        help="Combined image output path (only used in multi-G mode). Default: {test_name}/allG.png",
    )
    p.add_argument(
        "--title", default=None, help="Optional figure title (single-G only)."
    )
    p.add_argument(
        "--layout",
        choices=["kamada", "spring", "circular", "shell", "bipartite", "cluster"],
        default="kamada",
        help="Graph layout algorithm.",
    )
    p.add_argument(
        "--seed", type=int, default=42, help="Seed for deterministic layouts."
    )
    p.add_argument(
        "--vmin",
        type=float,
        default=None,
        help="Color scale minimum (GB/s). Auto if omitted.",
    )
    p.add_argument(
        "--vmax",
        type=float,
        default=None,
        help="Color scale maximum (GB/s). Auto if omitted.",
    )
    p.add_argument("--min-width", type=float, default=1.5, help="Minimum edge width.")
    p.add_argument("--max-width", type=float, default=3.0, help="Maximum edge width.")
    p.add_argument(
        "--decimals", type=int, default=1, help="Decimals for edge label numbers."
    )
    p.add_argument("--dpi", type=int, default=300, help="Output DPI.")

    # Multi-G presentation controls
    p.add_argument(
        "--combine-cols",
        type=int,
        default=4,
        help="Number of columns in the combined grid (multi-G).",
    )
    p.add_argument(
        "--no-combine",
        action="store_true",
        help="If set, do not create combined multi-G image (still emits per-G PNGs).",
    )
    p.add_argument(
        "--global-scale",
        action="store_true",
        help="Force shared vmin/vmax and edge-width scale across all G subplots.",
    )
    p.add_argument(
        "--adjust-labels",
        action="store_true",
        help="Enable automatic label adjustment to avoid overlaps. May move labels away from edge centers.",
    )
    p.add_argument(
        "--heatmap",
        dest="heatmap",
        action="store_true",
        default=True,
        help="Also emit NxN bandwidth heatmap PNG(s). (default: enabled)",
    )
    p.add_argument(
        "--no-heatmap",
        dest="heatmap",
        action="store_false",
        help="Disable heatmap output.",
    )
    p.add_argument(
        "--heatmap-only",
        action="store_true",
        help="Emit only heatmap PNG(s) (skip topology graph). Implies --heatmap.",
    )
    return p.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    """Validate command line arguments."""
    if args.vmin is not None and args.vmax is not None:
        if args.vmin >= args.vmax:
            raise ValueError("--vmin must be less than --vmax")
    if args.dpi <= 0:
        raise ValueError("--dpi must be positive")
    if args.heatmap_only:
        args.heatmap = True

    # Determine mode
    if args.test is None and args.g is None and not args.all:
        warnings.warn(
            "No --test, --g, or --all specified. Defaulting to --all mode.", UserWarning
        )
        args.all = True


def validate_df(df: pd.DataFrame) -> None:
    missing = REQUIRED_COLS - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {sorted(missing)}")


# ===========================================================
# Data utilities
# ===========================================================


def normalize_pair(nodes_str: str) -> Tuple[str, str]:
    s = nodes_str.strip().strip('"').strip("'")
    parts = [x.strip() for x in s.split(",") if x.strip()]
    if len(parts) != 2:
        raise ValueError(
            f"Invalid nodes field (expect two comma-separated names): {nodes_str!r}"
        )
    a, b = sorted(parts)
    return (a, b)


def aggregate_links(df: pd.DataFrame) -> pd.DataFrame:
    # Expect df already filtered by test and (optionally) G
    pairs = df["nodes"].apply(normalize_pair)
    df = df.copy()
    df[["node_a", "node_b"]] = pd.DataFrame(pairs.tolist(), index=df.index)
    # Average over duplicate pairs if present
    agg = df.groupby(["node_a", "node_b"], as_index=False).agg(
        avg_bus_bw_GBs=("avg_bus_bw_GBs", "mean"),
        count=("avg_bus_bw_GBs", "size"),
    )
    return agg


# ===========================================================
# Graph utilities
# ===========================================================


def bandwidth_clustering(G: nx.Graph, threshold: float) -> List[List[str]]:
    G_filtered = nx.Graph()
    G_filtered.add_nodes_from(G.nodes())

    for u, v, data in G.edges(data=True):
        if data["bw"] >= threshold:
            G_filtered.add_edge(u, v, bw=data["bw"])

    communities = list(nx.connected_components(G_filtered))

    if len(communities) == 1 and threshold > 0:
        print(f"  Threshold {threshold} too high, trying {threshold * 0.8}")
        return bandwidth_clustering(G, threshold * 0.8)

    return [sorted(list(comm)) for comm in communities]


def pick_layout(G: nx.Graph, which: str, seed: int) -> Dict[str, Tuple[float, float]]:
    if which == "kamada":
        has_bw = all("bw" in G[u][v] for u, v in G.edges()) if G.edges() else False
        if has_bw:
            return nx.kamada_kawai_layout(G, weight="bw", scale=1.0)
        else:
            return nx.kamada_kawai_layout(G, weight=None, scale=1.0)
    if which == "spring":
        return nx.spring_layout(G, seed=seed, k=None)  # automatic k
    if which == "circular":
        return nx.circular_layout(G)
    if which == "shell":
        return nx.shell_layout(G, nlist=[list(G.nodes())])
    if which == "bipartite":
        nodes = sorted(G.nodes())
        left = nodes[0 : (len(nodes) + 1) // 2]
        return nx.bipartite_layout(
            G, left, align="vertical", aspect_ratio=0.5, scale=1.0
        )
    if which in ["cluster"]:
        has_bw = all("bw" in G[u][v] for u, v in G.edges())
        weight = "bw" if has_bw else None
        if not has_bw:
            print(
                "Warning: Graph edges missing 'bw' attribute, using unweighted clustering."
            )

        communities = list(nx_comm.greedy_modularity_communities(G, weight=weight))
        print(
            f"  Detected {len(communities)} communities using modularity (weighted={has_bw})."
        )

        if len(communities) == 1 and has_bw:
            print("Graph is too dense, trying bandwidth-based clustering...")
            bws = [G[u][v]["bw"] for u, v in G.edges()]
            threshold = np.percentile(bws, 50)
            communities = bandwidth_clustering(G, threshold)
            print(
                f"  Bandwidth clustering found {len(communities)} groups (threshold={threshold:.1f})"
            )

        if len(communities) == 1:
            print("Still only 1 community, using spring layout.")
            return nx.spring_layout(G, seed=seed)

        supergraph = nx.cycle_graph(len(communities))
        superpos = nx.spring_layout(supergraph, scale=3, seed=seed)
        centers = list(superpos.values())
        pos = {}
        for center, comm in zip(centers, communities):
            pos.update(nx.spring_layout(nx.subgraph(G, comm), center=center, seed=seed))
        return pos

    raise ValueError(f"Unsupported layout: {which}")


def dynamic_figsize(n_nodes: int) -> Tuple[float, float]:
    base = 6.0
    if n_nodes <= 6:
        return (base, base)
    s = base + 0.6 * math.sqrt(max(0, n_nodes - 6))
    if n_nodes > 12:
        s *= 1.5
    max_size = 20.0 if n_nodes > 12 else 14.0
    return (min(max_size, s), min(max_size, s))


def compute_edge_widths(values: List[float], wmin: float, wmax: float) -> List[float]:
    if not values:
        return []
    lo, hi = min(values), max(values)
    if math.isclose(lo, hi, rel_tol=1e-12, abs_tol=1e-12):
        return [0.5 * (wmin + wmax)] * len(values)
    return [wmin + (v - lo) * (wmax - wmin) / (hi - lo) for v in values]


def build_graph(agg_df: pd.DataFrame) -> nx.Graph:
    G = nx.Graph()
    nodes = sorted(set(agg_df["node_a"]).union(set(agg_df["node_b"])))
    G.add_nodes_from(nodes)
    for _, row in agg_df.iterrows():
        G.add_edge(
            row["node_a"],
            row["node_b"],
            bw=float(row["avg_bus_bw_GBs"]),
            n=int(row["count"]),
        )
    return G


def build_bw_matrix(
    agg_df: pd.DataFrame, nodes: List[str]
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (z, customdata) matrices for an NxN heatmap."""
    n = len(nodes)
    idx = {node: i for i, node in enumerate(nodes)}
    z = np.full((n, n), np.nan, dtype=float)
    custom = np.full((n, n), "missing", dtype=object)

    for _, row in agg_df.iterrows():
        i = idx[row["node_a"]]
        j = idx[row["node_b"]]
        bw = float(row["avg_bus_bw_GBs"])
        z[i, j] = bw
        z[j, i] = bw
        custom[i, j] = f"{bw:.2f} GB/s"
        custom[j, i] = f"{bw:.2f} GB/s"

    for i in range(n):
        custom[i, i] = "self"
    return z, custom


# ===========================================================
# Drawing
# ===========================================================


def get_trimmed_colormap():
    """
    Get a trimmed colormap that excludes the lightest colors for better visibility.

    Returns:
        Trimmed colormap with better contrast.
    """
    from matplotlib.colors import LinearSegmentedColormap

    base_cmap = plt.colormaps.get_cmap(DEFAULT_CMAP)
    # Sample the colormap but skip the lightest part
    colors = base_cmap(np.linspace(CMAP_VMIN_RATIO, CMAP_VMAX_RATIO, 256))
    return LinearSegmentedColormap.from_list(f"{DEFAULT_CMAP}_trimmed", colors)


def add_colorbar(
    fig: plt.Figure,
    ax,
    vmin: float,
    vmax: float,
    label: str = "Average bus BW (GB/s)",
    **kwargs,
):
    """Add colorbar to figure."""
    sm = cm.ScalarMappable(
        norm=Normalize(vmin=vmin, vmax=vmax, clip=True), cmap=get_trimmed_colormap()
    )
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, **kwargs)
    cbar.set_label(label)
    return cbar


def dynamic_heatmap_figsize(n_nodes: int) -> Tuple[float, float]:
    base = 6.0
    size = max(base, min(0.35 * n_nodes, 40.0))
    return (size, size)


def heatmap_label_step(n_nodes: int) -> int:
    if n_nodes <= 20:
        return 1
    if n_nodes <= 40:
        return 2
    if n_nodes <= 80:
        return 4
    return 6


def draw_heatmap_ax(
    z: np.ndarray,
    nodes: List[str],
    ax: plt.Axes,
    *,
    vmin: float,
    vmax: float,
    title: Optional[str] = None,
):
    ax.set_aspect("equal")
    n = len(nodes)
    cmap = get_trimmed_colormap()
    if hasattr(cmap, "copy"):
        cmap = cmap.copy()
    cmap.set_bad(color="lightgray")

    z_masked = np.ma.masked_invalid(z)
    im = ax.imshow(z_masked, vmin=vmin, vmax=vmax, cmap=cmap, interpolation="none")

    step = heatmap_label_step(n)
    ticks = np.arange(0, n, step)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)

    font_size = 8 if n <= 30 else 6 if n <= 60 else 5
    ax.set_xticklabels([nodes[i] for i in ticks], rotation=90, fontsize=font_size)
    ax.set_yticklabels([nodes[i] for i in ticks], fontsize=font_size)

    if title:
        ax.set_title(title, fontsize=11, pad=10)

    return im


def draw_topology_ax(
    G: nx.Graph,
    pos: Dict[str, Tuple[float, float]],
    ax: plt.Axes,
    *,
    vmin: float,
    vmax: float,
    min_width: float,
    max_width: float,
    decimals: int,
    title: Optional[str] = None,
    adjust_labels: bool = False,
):
    ax.set_axis_off()

    # Get all nodes
    all_nodes = list(G.nodes())

    # Separate edges into normal (with valid bw) and missing/NaN edges
    normal_edges = []
    missing_edges = []
    normal_bws = []

    # Check all possible node pairs
    for i, u in enumerate(all_nodes):
        for v in all_nodes[i + 1 :]:  # Only check upper triangle (undirected graph)
            if G.has_edge(u, v):
                bw = G[u][v]["bw"]
                if pd.isna(bw) or (isinstance(bw, float) and math.isnan(bw)):
                    missing_edges.append((u, v))
                else:
                    normal_edges.append((u, v))
                    normal_bws.append(float(bw))
            else:
                # Edge doesn't exist in graph - mark as missing
                missing_edges.append((u, v))

    # Draw nodes and labels
    nx.draw_networkx_nodes(
        G,
        pos,
        ax=ax,
        node_color="white",
        edgecolors="black",
        node_size=DEFAULT_NODE_SIZE,
        linewidths=1.2,
    )
    nx.draw_networkx_labels(
        G, pos, ax=ax, font_size=DEFAULT_FONT_SIZE, font_weight="semibold"
    )

    # Draw normal edges (using colormap)
    if normal_edges:
        norm = Normalize(vmin=vmin, vmax=vmax, clip=True)
        cmap = get_trimmed_colormap()
        edge_colors = [cmap(norm(G[u][v]["bw"])) for u, v in normal_edges]
        edge_widths = compute_edge_widths(normal_bws, min_width, max_width)

        nx.draw_networkx_edges(
            G,
            pos,
            ax=ax,
            edgelist=normal_edges,
            width=edge_widths,
            edge_color=edge_colors,
        )

    # Draw missing/NaN edges (red dashed lines, no labels)
    if missing_edges:
        nx.draw_networkx_edges(
            G if len(G.edges()) > 0 else nx.complete_graph(all_nodes),
            pos,
            ax=ax,
            edgelist=missing_edges,
            width=2.5,
            edge_color="red",
            style="dashed",
            alpha=0.6,
        )

    # Create edge labels (only for normal edges with valid values)
    fmt = f"{{:.{decimals}f}}"

    if adjust_labels:
        # Use adjustText for automatic overlap avoidance
        texts = []
        for u, v in normal_edges:  # Only label normal edges
            x = (pos[u][0] + pos[v][0]) / 2
            y = (pos[u][1] + pos[v][1]) / 2
            label = fmt.format(G[u][v]["bw"])

            text = ax.text(
                x,
                y,
                label,
                fontsize=DEFAULT_EDGE_LABEL_FONT_SIZE,
                ha="center",
                va="center",
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.7),
            )
            texts.append(text)

        if len(texts) > 1:
            adjust_text(
                texts,
                only_move={"text": "xy"},
                arrowprops=dict(
                    arrowstyle="-",
                    color="gray",
                    lw=0.5,
                    alpha=0.3,
                    shrinkA=5,
                    shrinkB=5,
                ),
                expand_text=(1.01, 1.05),
                expand_points=(1.01, 1.05),
                force_text=(0.1, 0.1),
                force_points=(0.05, 0.05),
                lim=100,
                ax=ax,
            )
    else:
        # Use default networkx edge labels (only for normal edges)
        edge_labels = {(u, v): fmt.format(G[u][v]["bw"]) for u, v in normal_edges}
        if edge_labels:
            nx.draw_networkx_edge_labels(
                G,
                pos,
                ax=ax,
                edge_labels=edge_labels,
                font_size=DEFAULT_EDGE_LABEL_FONT_SIZE,
                rotate=False,
                label_pos=0.5,
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.7),
                verticalalignment="bottom",
            )

    if title:
        ax.set_title(title, fontsize=11, pad=10)


def compute_scales_for_groups(
    agg_per_g: Dict[int, pd.DataFrame],
    user_vmin: Optional[float],
    user_vmax: Optional[float],
    min_width: float,
    max_width: float,
    global_scale: bool,
):
    """Return (vmin, vmax, width_min, width_max, perG_bw_minmax)."""
    perG_minmax = {}
    all_bws = []
    for g, agg in agg_per_g.items():
        bws = agg["avg_bus_bw_GBs"].tolist()
        if bws:
            perG_minmax[g] = (min(bws), max(bws))
            all_bws.extend(bws)
        else:
            perG_minmax[g] = (0.0, 1.0)

    # Color scale
    if user_vmin is not None and user_vmax is not None:
        vmin, vmax = float(user_vmin), float(user_vmax)
    elif global_scale and all_bws:
        vmin, vmax = min(all_bws), max(all_bws)
    else:
        # Will set per-subplot later using perG_minmax.
        vmin = vmax = None

    # Edge-width scale
    width_min, width_max = min_width, max_width
    return vmin, vmax, width_min, width_max, perG_minmax


# ===========================================================
# Output path management
# ===========================================================


def determine_output_dir(
    csv_path: pathlib.Path, out_dir: Optional[str]
) -> pathlib.Path:
    """
    Determine the base output directory.

    Priority:
    1. User-specified --out-dir
    2. {csv_dir}/topology (preserves with-debug/without-debug structure)
    3. ./topology (fallback)
    """
    if out_dir:
        return pathlib.Path(out_dir)

    # Auto-detect: place topology/ in same directory as CSV
    csv_dir = csv_path.parent
    return csv_dir / "topology"


def get_output_paths(
    base_dir: pathlib.Path,
    test_name: str,
    g: Optional[int] = None,
    combined: bool = False,
) -> pathlib.Path:
    """
    Generate output path following the structure: topology/{test_name}/G{g}.png

    Args:
        base_dir: Base topology directory
        test_name: Test name (e.g., alltoall_perf)
        g: G value (None for combined)
        combined: True for allG.png

    Returns:
        Full path to output file
    """
    test_dir = base_dir / test_name
    test_dir.mkdir(parents=True, exist_ok=True)

    if combined:
        return test_dir / "allG.png"
    elif g is not None:
        return test_dir / f"G{g}.png"
    else:
        raise ValueError("Either g or combined must be specified")


def get_output_paths_heatmap(
    base_dir: pathlib.Path, test_name: str, g: Optional[int] = None
) -> pathlib.Path:
    test_dir = base_dir / test_name
    test_dir.mkdir(parents=True, exist_ok=True)
    if g is None:
        raise ValueError("g must be specified for heatmap output")
    return test_dir / f"G{g}_heatmap.png"


def get_output_paths_heatmap_combined(
    base_dir: pathlib.Path, test_name: str
) -> pathlib.Path:
    test_dir = base_dir / test_name
    test_dir.mkdir(parents=True, exist_ok=True)
    return test_dir / "allG_heatmap.png"


# ===========================================================
# Processing logic
# ===========================================================


def process_single_g(
    df: pd.DataFrame,
    test_name: str,
    g_value: int,
    args: argparse.Namespace,
    output_path: pathlib.Path,
) -> None:
    """Process and save a single G value for a test."""
    filtered = df[(df["test"] == test_name) & (df["G"] == g_value)]
    if filtered.empty:
        warnings.warn(
            f"No data for test={test_name}, G={g_value}. Skipping.", UserWarning
        )
        return

    agg = aggregate_links(filtered)
    nodes = sorted(set(agg["node_a"]).union(set(agg["node_b"])))
    Gx = build_graph(agg)
    pos = pick_layout(Gx, args.layout, args.seed)

    bws = [float(x) for x in agg["avg_bus_bw_GBs"].tolist()]
    if bws:
        auto_vmin = min(bws)
        auto_vmax = max(bws)
    else:
        auto_vmin, auto_vmax = 0.0, 1.0

    vmin = args.vmin if args.vmin is not None else auto_vmin
    vmax = args.vmax if args.vmax is not None else auto_vmax
    if math.isclose(vmin, vmax, rel_tol=1e-12, abs_tol=1e-12):
        vmin -= 0.5
        vmax += 0.5

    if args.heatmap:
        z, _ = build_bw_matrix(agg, nodes)
        fig, ax = plt.subplots(
            figsize=dynamic_heatmap_figsize(len(nodes)), dpi=args.dpi
        )
        draw_heatmap_ax(
            z,
            nodes,
            ax,
            vmin=vmin,
            vmax=vmax,
            title=(
                args.title if args.title else f"{test_name} — heatmap, N=2, G={g_value}"
            ),
        )
        add_colorbar(fig, ax, vmin, vmax, fraction=0.046, pad=0.04)
        heatmap_path = output_path.with_name(f"{output_path.stem}_heatmap.png")
        fig.tight_layout()
        fig.savefig(heatmap_path, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {heatmap_path}")
        if args.heatmap_only:
            return

    figsize = dynamic_figsize(len(Gx.nodes()))
    fig, ax = plt.subplots(figsize=figsize, dpi=args.dpi)
    draw_topology_ax(
        Gx,
        pos,
        ax,
        vmin=vmin,
        vmax=vmax,
        min_width=args.min_width,
        max_width=args.max_width,
        decimals=args.decimals,
        title=(args.title if args.title else f"{test_name} — N=2, G={g_value}"),
        adjust_labels=args.adjust_labels,
    )

    add_colorbar(fig, ax, vmin, vmax, fraction=0.046, pad=0.04)

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


def process_multi_g(
    df: pd.DataFrame,
    test_name: str,
    g_values: List[int],
    args: argparse.Namespace,
    base_dir: pathlib.Path,
) -> None:
    """Process multiple G values for a single test."""
    print(f"\nProcessing test: {test_name}")

    # Aggregate per G
    agg_per_g: Dict[int, pd.DataFrame] = {}
    union_edges = []
    for g in g_values:
        gdf = df[(df["test"] == test_name) & (df["G"] == g)]
        if gdf.empty:
            warnings.warn(
                f"No data for test={test_name}, G={g}. Skipping.", UserWarning
            )
            continue
        agg = aggregate_links(gdf)
        agg_per_g[g] = agg
        union_edges.append(agg[["node_a", "node_b"]])

    if not agg_per_g:
        warnings.warn(
            f"No data for test={test_name}. Skipping entire test.", UserWarning
        )
        return

    g_values_with_data = sorted(agg_per_g.keys())

    union_nodes = set()
    for agg in agg_per_g.values():
        union_nodes.update(agg["node_a"].tolist())
        union_nodes.update(agg["node_b"].tolist())

    # Build a union graph for layout stability across subplots
    G_union = nx.Graph()
    G_union.add_nodes_from(sorted(union_nodes))
    edge_bws = {}  # (u, v) -> [bw1, bw2, ...]
    for agg in agg_per_g.values():
        for _, row in agg.iterrows():
            u, v = row["node_a"], row["node_b"]
            edge_key = tuple(sorted([u, v]))
            if edge_key not in edge_bws:
                edge_bws[edge_key] = []
            edge_bws[edge_key].append(float(row["avg_bus_bw_GBs"]))

    for (u, v), bws in edge_bws.items():
        G_union.add_edge(u, v, bw=float(np.mean(bws)))

    pos_union = pick_layout(G_union, args.layout, args.seed)

    # Scales - always use global scale for multi-G comparison
    global_scale = True
    vmin_glob, vmax_glob, wmin, wmax, perG_minmax = compute_scales_for_groups(
        agg_per_g,
        args.vmin,
        args.vmax,
        args.min_width,
        args.max_width,
        global_scale=global_scale,
    )

    # Emit heatmaps (per-G)
    if args.heatmap:
        for g in g_values_with_data:
            agg = agg_per_g[g]
            nodes = sorted(set(agg["node_a"]).union(set(agg["node_b"])))
            z, _ = build_bw_matrix(agg, nodes)
            if vmin_glob is None or vmax_glob is None:
                vmin, vmax = perG_minmax[g]
                if math.isclose(vmin, vmax, rel_tol=1e-12, abs_tol=1e-12):
                    vmin -= 0.5
                    vmax += 0.5
            else:
                vmin, vmax = vmin_glob, vmax_glob

            fig, ax = plt.subplots(
                figsize=dynamic_heatmap_figsize(len(nodes)), dpi=args.dpi
            )
            draw_heatmap_ax(
                z,
                nodes,
                ax,
                vmin=vmin,
                vmax=vmax,
                title=f"{test_name} — heatmap, N=2, G={g}",
            )
            add_colorbar(fig, ax, vmin, vmax, fraction=0.046, pad=0.04)

            out_path = get_output_paths_heatmap(base_dir, test_name, g=g)
            fig.tight_layout()
            fig.savefig(out_path, bbox_inches="tight")
            plt.close(fig)
            print(f"  Saved: {out_path}")

        # Combined heatmap grid
        if len(g_values_with_data) > 1:
            n = len(g_values_with_data)
            cols = max(1, int(args.combine_cols))
            cols_eff = min(cols, n)
            rows = (n + cols_eff - 1) // cols_eff

            panel_w, panel_h = dynamic_heatmap_figsize(len(union_nodes))
            panel_w = max(4.0, panel_w * 0.9)
            panel_h = max(4.0, panel_h * 0.9)

            fig_w = cols_eff * panel_w
            fig_h = rows * panel_h

            fig, axes = plt.subplots(
                rows, cols_eff, figsize=(fig_w, fig_h), dpi=args.dpi
            )
            if not isinstance(axes, (list, tuple, pd.Series, np.ndarray)):
                axes = [[axes]]
            else:
                if isinstance(axes, np.ndarray):
                    if axes.ndim == 1:
                        axes = np.expand_dims(axes, axis=0)
                    axes = axes.tolist()

            if vmin_glob is None or vmax_glob is None:
                all_bws = []
                for agg in agg_per_g.values():
                    all_bws.extend(agg["avg_bus_bw_GBs"].tolist())
                if all_bws:
                    vmin_c, vmax_c = min(all_bws), max(all_bws)
                    if math.isclose(vmin_c, vmax_c, rel_tol=1e-12, abs_tol=1e-12):
                        vmin_c -= 0.5
                        vmax_c += 0.5
                else:
                    vmin_c, vmax_c = 0.0, 1.0
            else:
                vmin_c, vmax_c = vmin_glob, vmax_glob

            ax_list = [ax for row_axes in axes for ax in row_axes]
            for idx, g in enumerate(g_values_with_data):
                ax = ax_list[idx]
                agg = agg_per_g[g]
                nodes = sorted(set(agg["node_a"]).union(set(agg["node_b"])))
                z, _ = build_bw_matrix(agg, nodes)
                draw_heatmap_ax(
                    z,
                    nodes,
                    ax,
                    vmin=vmin_c,
                    vmax=vmax_c,
                    title=f"G={g}",
                )

            for j in range(len(g_values_with_data), len(ax_list)):
                ax_list[j].axis("off")

            add_colorbar(
                fig,
                ax_list[: len(g_values_with_data)],
                vmin_c,
                vmax_c,
                fraction=0.02,
                pad=0.02,
            )
            fig.suptitle(f"{test_name} — heatmap, all G", fontsize=13, y=0.995)

            combined_path = get_output_paths_heatmap_combined(base_dir, test_name)
            fig.savefig(combined_path, bbox_inches="tight")
            plt.close(fig)
            print(f"  Saved: {combined_path}")

        if args.heatmap_only:
            return

    # Emit per-G images
    for g in g_values_with_data:
        agg = agg_per_g[g]
        Gx = build_graph(agg)
        figsize = dynamic_figsize(len(G_union.nodes()))
        fig, ax = plt.subplots(figsize=figsize, dpi=args.dpi)

        if vmin_glob is None or vmax_glob is None:
            lo, hi = perG_minmax[g]
            vmin, vmax = lo, hi
            if math.isclose(vmin, vmax, rel_tol=1e-12, abs_tol=1e-12):
                vmin -= 0.5
                vmax += 0.5
        else:
            vmin, vmax = vmin_glob, vmax_glob

        draw_topology_ax(
            Gx,
            pos_union,
            ax,
            vmin=vmin,
            vmax=vmax,
            min_width=wmin,
            max_width=wmax,
            decimals=args.decimals,
            title=f"{test_name} — N=2, G={g}",
            adjust_labels=args.adjust_labels,
        )

        add_colorbar(fig, ax, vmin, vmax, fraction=0.046, pad=0.04)

        out_path = get_output_paths(base_dir, test_name, g=g)
        fig.tight_layout()
        fig.savefig(out_path, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {out_path}")

    # Combined grid
    if not args.no_combine and len(g_values_with_data) > 1:
        n = len(g_values_with_data)
        cols = max(1, int(args.combine_cols))
        cols_eff = min(cols, n)
        rows = (n + cols_eff - 1) // cols_eff

        panel_w, panel_h = dynamic_figsize(len(G_union.nodes()))
        # Scale panels to match individual G images (remove max size cap)
        panel_w = max(4.0, panel_w * 0.9)
        panel_h = max(4.0, panel_h * 0.9)

        fig_w = cols_eff * panel_w
        fig_h = rows * panel_h

        fig, axes = plt.subplots(rows, cols_eff, figsize=(fig_w, fig_h), dpi=args.dpi)
        if not isinstance(axes, (list, tuple, pd.Series, np.ndarray)):
            axes = [[axes]]
        else:
            if isinstance(axes, np.ndarray):
                if axes.ndim == 1:
                    axes = np.expand_dims(axes, axis=0)
                axes = axes.tolist()

        # Decide global color scale for the combined figure
        if vmin_glob is None or vmax_glob is None:
            all_bws = []
            for agg in agg_per_g.values():
                all_bws.extend(agg["avg_bus_bw_GBs"].tolist())
            if all_bws:
                vmin_c, vmax_c = min(all_bws), max(all_bws)
                if math.isclose(vmin_c, vmax_c, rel_tol=1e-12, abs_tol=1e-12):
                    vmin_c -= 0.5
                    vmax_c += 0.5
            else:
                vmin_c, vmax_c = 0.0, 1.0
        else:
            vmin_c, vmax_c = vmin_glob, vmax_glob

        # Draw each subplot
        ax_list = [ax for row_axes in axes for ax in row_axes]
        for idx, g in enumerate(g_values_with_data):
            ax = ax_list[idx]
            agg = agg_per_g[g]
            Gx = build_graph(agg)
            draw_topology_ax(
                Gx,
                pos_union,
                ax,
                vmin=vmin_c,
                vmax=vmax_c,
                min_width=wmin,
                max_width=wmax,
                decimals=args.decimals,
                title=f"G={g}",
                adjust_labels=args.adjust_labels,
            )

        # Hide any extra axes
        for j in range(len(g_values_with_data), len(ax_list)):
            ax_list[j].axis("off")

        # Shared colorbar
        add_colorbar(
            fig,
            ax_list[: len(g_values_with_data)],
            vmin_c,
            vmax_c,
            fraction=0.02,
            pad=0.02,
        )

        # Suptitle
        fig.suptitle(f"{test_name} — all G", fontsize=13, y=0.995)

        # Save combined
        combined_path = get_output_paths(base_dir, test_name, combined=True)
        fig.savefig(combined_path, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {combined_path}")


# ===========================================================
# Main
# ===========================================================


def main():
    args = parse_args()
    validate_args(args)

    csv_path = pathlib.Path(args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)

    df = pd.read_csv(csv_path)
    validate_df(df)

    # Determine output directory
    base_output_dir = determine_output_dir(csv_path, args.out_dir)

    # Determine processing mode
    single_g_mode = args.test is not None and args.g is not None
    single_test_mode = args.test is not None and args.g is None
    all_mode = args.all or (args.test is None and args.g is None)

    if single_g_mode:
        # Single G, single test (backward compatible)
        print(f"Mode: Single test ({args.test}), single G ({args.g})")

        if args.out:
            output_path = pathlib.Path(args.out)
            output_path.parent.mkdir(parents=True, exist_ok=True)
        else:
            output_path = get_output_paths(base_output_dir, args.test, g=args.g)

        process_single_g(df, args.test, args.g, args, output_path)
        print(f"\nDone! Output: {output_path}")

    elif single_test_mode:
        # Single test, all G values
        print(f"Mode: Single test ({args.test}), all G values")

        df_test = df[df["test"] == args.test]
        if df_test.empty:
            raise SystemExit(f"No rows for test={args.test}")

        g_values = sorted(df_test["G"].dropna().astype(int).unique().tolist())
        if not g_values:
            raise SystemExit(f"No G values found for test={args.test}")

        process_multi_g(df, args.test, g_values, args, base_output_dir)
        print(f"\nDone! Output directory: {base_output_dir / args.test}")

    elif all_mode:
        # All tests, all G values
        print("Mode: All tests, all G values")

        test_names = sorted(df["test"].dropna().unique().tolist())
        if not test_names:
            raise SystemExit("No test names found in CSV")

        print(f"Found {len(test_names)} test(s): {', '.join(test_names)}")

        for test_name in test_names:
            df_test = df[df["test"] == test_name]
            g_values = sorted(df_test["G"].dropna().astype(int).unique().tolist())

            if not g_values:
                warnings.warn(
                    f"No G values for test={test_name}. Skipping.", UserWarning
                )
                continue

            process_multi_g(df, test_name, g_values, args, base_output_dir)

        print(f"\nDone! Output directory: {base_output_dir}")
        print(f"Structure: {base_output_dir}/{{test_name}}/G{{n}}.png")


if __name__ == "__main__":
    main()
