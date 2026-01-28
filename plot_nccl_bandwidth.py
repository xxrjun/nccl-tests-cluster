"""
Generate bandwidth plots from NCCL benchmark logs.

Features
- Parse per-message-size data from NCCL test logs
- Generate line plots: message size (x-axis) vs bandwidth (y-axis)
- Support single-node and multi-node results
- Per-node individual plots and combined comparison plots
- Supports multiple test types (all_reduce, all_gather, etc.)

Usage
  # Generate plots from single-node logs
  python plot_nccl_bandwidth.py --input benchmarks/cluster00/single-node/latest/without-debug/logs

  # Specify output directory
  python plot_nccl_bandwidth.py --input ./logs --out-dir ./plots

  # Filter by test type
  python plot_nccl_bandwidth.py --input ./logs --test all_reduce_perf

  # Specify GPU count filter
  python plot_nccl_bandwidth.py --input ./logs --g 8
"""

import argparse
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ===========================================================
# Regex patterns (same as summarize_nccl_logs.py)
# ===========================================================

RE_SECTION_START = re.compile(r"^#\s*Collective test starting:\s*(\S+)", re.MULTILINE)
RE_HEADER_BYTES = re.compile(
    r"^#\s*nThread\s+\d+\s+nGpus\s+\d+\s+minBytes\s+(\d+)\s+maxBytes\s+(\d+)\s+step:",
    re.MULTILINE,
)

# Table rows: size count type redop root time algbw busbw #wrong time algbw busbw #wrong
RE_TABLE_ROW = re.compile(
    r"^\s*(\d+)\s+\d+\s+\S+\s+\S+\s+-?\d+\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+\d+\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)",
    re.MULTILINE,
)

# Fallback for logs with only out-of-place results
RE_TABLE_ROW_OOP_ONLY = re.compile(
    r"^\s*(\d+)\s+\d+\s+\S+\s+\S+\s+-?\d+\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+\d+\s*$",
    re.MULTILINE,
)

# Filename pattern: _N{N}_G{G}[_node1_node2_...].log
RE_FILE_NG_NODES = re.compile(
    r"_N(\d+)_G(\d+)(?:_([^.]+?)(?<!_debug))?\.log$", re.IGNORECASE
)


# ===========================================================
# Utility Functions
# ===========================================================


def human_bytes(n: int) -> str:
    """Convert bytes to human-readable format."""
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    v = float(n)
    for u in units:
        if v < 1024.0 or u == units[-1]:
            if u == "B":
                return f"{int(v)}{u}"
            elif v >= 100:
                return f"{v:.0f}{u}"
            elif v >= 10:
                return f"{v:.1f}{u}"
            else:
                return f"{v:.2f}{u}"
        v /= 1024.0
    return f"{n}B"


def split_sections(text: str) -> List[Tuple[str, str]]:
    """Split file into sections per 'Collective test starting:'."""
    sections: List[Tuple[str, str]] = []
    starts = [(m.start(), m.group(1)) for m in RE_SECTION_START.finditer(text)]
    if not starts:
        return sections
    for i, (pos, test_name) in enumerate(starts):
        end = starts[i + 1][0] if i + 1 < len(starts) else len(text)
        sections.append((test_name, text[pos:end]))
    return sections


def parse_section_data(section_text: str) -> List[Dict]:
    """
    Extract per-message-size data from a test section.
    Returns list of dicts with: size_bytes, time_us, algbw_GBs, busbw_GBs
    """
    rows = []

    # Try rich format first (both out-of-place and in-place)
    for m in RE_TABLE_ROW.finditer(section_text):
        try:
            size = int(m.group(1))
            # Use in-place results (columns 5-7) if available, typically more stable
            time_us = float(m.group(5))
            algbw = float(m.group(6))
            busbw = float(m.group(7))
            rows.append({
                "size_bytes": size,
                "time_us": time_us,
                "algbw_GBs": algbw,
                "busbw_GBs": busbw,
            })
        except (ValueError, IndexError):
            continue

    # Fallback to out-of-place only format
    if not rows:
        for m in RE_TABLE_ROW_OOP_ONLY.finditer(section_text):
            try:
                size = int(m.group(1))
                time_us = float(m.group(2))
                algbw = float(m.group(3))
                busbw = float(m.group(4))
                rows.append({
                    "size_bytes": size,
                    "time_us": time_us,
                    "algbw_GBs": algbw,
                    "busbw_GBs": busbw,
                })
            except (ValueError, IndexError):
                continue

    return rows


def infer_ng_nodes(file_path: str) -> Tuple[Optional[int], Optional[int], List[str]]:
    """
    Infer N, G, and node names from filename.
    Returns (N, G, nodes_list).
    """
    base = os.path.basename(file_path)
    m = RE_FILE_NG_NODES.search(base)
    if not m:
        return None, None, []
    N = int(m.group(1))
    G = int(m.group(2))
    nodes_list: List[str] = []
    if m.group(3):
        nodes_list = [tok for tok in m.group(3).split("_") if tok]
    return N, G, nodes_list


def walk_logs(root: str) -> List[str]:
    """Recursively find all .log files."""
    logs: List[str] = []
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            if fn.lower().endswith(".log") and not fn.endswith("_debug.log"):
                logs.append(os.path.join(dirpath, fn))
    logs.sort()
    return logs


# ===========================================================
# Data Processing
# ===========================================================


def process_logs(root: str) -> pd.DataFrame:
    """
    Process all logs and return a DataFrame with per-message-size data.
    Columns: file, test, N, G, nodes, size_bytes, time_us, algbw_GBs, busbw_GBs
    """
    logs = walk_logs(root)
    all_rows = []

    for path in logs:
        basename = os.path.basename(path)
        N, G, nodes_list = infer_ng_nodes(path)

        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as fh:
                text = fh.read()
        except Exception as e:
            print(f"Warning: Failed to read {path}: {e}")
            continue

        sections = split_sections(text)
        if not sections:
            continue

        for test_name, section_text in sections:
            data_rows = parse_section_data(section_text)
            for dr in data_rows:
                all_rows.append({
                    "file": basename,
                    "test": test_name,
                    "N": N,
                    "G": G,
                    "nodes": ",".join(nodes_list) if nodes_list else "",
                    "node": nodes_list[0] if len(nodes_list) == 1 else ",".join(nodes_list),
                    **dr,
                })

    if not all_rows:
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)
    # Sort for consistent output
    df = df.sort_values(["test", "N", "G", "file", "size_bytes"]).reset_index(drop=True)
    return df


# ===========================================================
# Plotting Functions
# ===========================================================


def setup_plot_style():
    """Configure matplotlib style for professional plots."""
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update({
        "figure.figsize": (12, 7),
        "font.size": 11,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    })


def plot_bandwidth_single_node(
    df: pd.DataFrame,
    test_name: str,
    g_value: int,
    out_dir: str,
    metric: str = "busbw_GBs",
) -> Optional[str]:
    """
    Generate individual plots for each node and a combined comparison plot.
    
    Args:
        df: DataFrame filtered by test and G
        test_name: Name of the test (e.g., "all_reduce_perf")
        g_value: GPU count
        out_dir: Output directory
        metric: Which metric to plot ("busbw_GBs" or "algbw_GBs")
    
    Returns:
        Path to combined plot, or None if no data
    """
    if df.empty:
        return None

    # Filter for single-node results
    df_single = df[df["N"] == 1].copy()
    if df_single.empty:
        return None

    nodes = df_single["node"].unique()
    test_short = test_name.replace("_perf", "")
    metric_label = "Bus Bandwidth" if metric == "busbw_GBs" else "Algorithm Bandwidth"

    # Create output subdirectory
    plot_dir = Path(out_dir) / "plots" / test_name
    plot_dir.mkdir(parents=True, exist_ok=True)

    # Color palette with darker, more distinguishable colors
    DARK_COLORS = [
        '#1f77b4',  # blue
        '#d62728',  # red
        '#2ca02c',  # green
        '#ff7f0e',  # orange
        '#9467bd',  # purple
        '#8c564b',  # brown
        '#e377c2',  # pink
        '#17becf',  # cyan
        '#bcbd22',  # olive
        '#7f7f7f',  # gray
    ]
    MARKERS = ['o', 's', '^', 'D', 'v', 'p', 'h', '*', 'X', 'P']
    node_colors = {node: DARK_COLORS[i % len(DARK_COLORS)] for i, node in enumerate(sorted(nodes))}
    node_markers = {node: MARKERS[i % len(MARKERS)] for i, node in enumerate(sorted(nodes))}

    # Individual node plots
    for node in nodes:
        node_df = df_single[df_single["node"] == node].sort_values("size_bytes")
        if node_df.empty:
            continue

        fig, ax = plt.subplots(figsize=(10, 6))
        
        sizes = node_df["size_bytes"].values
        bw = node_df[metric].values

        ax.plot(sizes, bw, marker=node_markers[node], linewidth=2, markersize=6, color=node_colors[node])
        
        ax.set_xscale("log", base=2)
        ax.set_xlabel("Message Size")
        ax.set_ylabel(f"{metric_label} (GB/s)")
        ax.set_title(f"{test_short} - {node} (G={g_value})")
        
        # Custom x-axis labels
        ax.set_xticks(sizes)
        ax.set_xticklabels([human_bytes(s) for s in sizes], rotation=45, ha="right")
        
        ax.grid(True, alpha=0.3)
        
        # Add peak bandwidth annotation
        peak_bw = bw.max()
        peak_idx = bw.argmax()
        ax.annotate(
            f"Peak: {peak_bw:.1f} GB/s",
            xy=(sizes[peak_idx], peak_bw),
            xytext=(10, 10),
            textcoords="offset points",
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7),
        )

        node_safe = node.replace("/", "_").replace("\\", "_")
        out_path = plot_dir / f"G{g_value}_{node_safe}.png"
        fig.savefig(out_path)
        plt.close(fig)
        print(f"Saved: {out_path}")

    # Combined comparison plot
    fig, ax = plt.subplots(figsize=(12, 7))

    for node in sorted(nodes):
        node_df = df_single[df_single["node"] == node].sort_values("size_bytes")
        if node_df.empty:
            continue

        sizes = node_df["size_bytes"].values
        bw = node_df[metric].values

        ax.plot(
            sizes, bw,
            marker=node_markers[node],
            linewidth=2,
            markersize=6,
            label=node,
            color=node_colors[node],
        )

    ax.set_xscale("log", base=2)
    ax.set_xlabel("Message Size")
    ax.set_ylabel(f"{metric_label} (GB/s)")
    ax.set_title(f"{test_short} - All Nodes Comparison (G={g_value})")

    # Set x-axis ticks based on actual data
    all_sizes = df_single["size_bytes"].unique()
    all_sizes.sort()
    ax.set_xticks(all_sizes)
    ax.set_xticklabels([human_bytes(s) for s in all_sizes], rotation=45, ha="right")

    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", ncol=2 if len(nodes) > 10 else 1, framealpha=0.9)

    combined_path = plot_dir / f"G{g_value}_combined.png"
    fig.savefig(combined_path)
    plt.close(fig)
    print(f"Saved: {combined_path}")

    return str(combined_path)


def plot_bandwidth_pairwise(
    df: pd.DataFrame,
    test_name: str,
    g_value: int,
    out_dir: str,
    metric: str = "busbw_GBs",
) -> Optional[str]:
    """
    Generate bandwidth plots for pairwise (N=2) test results.
    Shows bandwidth across message sizes for different node pairs.
    """
    if df.empty:
        return None

    # Filter for pairwise results
    df_pair = df[df["N"] == 2].copy()
    if df_pair.empty:
        return None

    pairs = df_pair["nodes"].unique()
    test_short = test_name.replace("_perf", "")
    metric_label = "Bus Bandwidth" if metric == "busbw_GBs" else "Algorithm Bandwidth"

    # Create output directory
    plot_dir = Path(out_dir) / "plots" / test_name
    plot_dir.mkdir(parents=True, exist_ok=True)

    # Color palette with darker colors
    DARK_COLORS = [
        '#1f77b4', '#d62728', '#2ca02c', '#ff7f0e', '#9467bd',
        '#8c564b', '#e377c2', '#17becf', '#bcbd22', '#7f7f7f',
        '#393b79', '#637939', '#8c6d31', '#843c39', '#7b4173',
        '#5254a3', '#8ca252', '#bd9e39', '#ad494a', '#a55194',
    ]
    MARKERS = ['o', 's', '^', 'D', 'v', 'p', 'h', '*', 'X', 'P']

    # Combined plot for all pairs
    fig, ax = plt.subplots(figsize=(14, 8))

    for i, pair in enumerate(sorted(pairs)):
        pair_df = df_pair[df_pair["nodes"] == pair].sort_values("size_bytes")
        if pair_df.empty:
            continue

        sizes = pair_df["size_bytes"].values
        bw = pair_df[metric].values

        ax.plot(
            sizes, bw,
            marker=MARKERS[i % len(MARKERS)],
            linewidth=1.5,
            markersize=5,
            label=pair,
            color=DARK_COLORS[i % len(DARK_COLORS)],
        )

    ax.set_xscale("log", base=2)
    ax.set_xlabel("Message Size")
    ax.set_ylabel(f"{metric_label} (GB/s)")
    ax.set_title(f"{test_short} - Pairwise Comparison (G={g_value})")

    # Set x-axis ticks
    all_sizes = df_pair["size_bytes"].unique()
    all_sizes.sort()
    ax.set_xticks(all_sizes)
    ax.set_xticklabels([human_bytes(s) for s in all_sizes], rotation=45, ha="right")

    ax.grid(True, alpha=0.3)
    
    # Legend handling for many pairs
    if len(pairs) <= 15:
        ax.legend(loc="best", ncol=2, framealpha=0.9, fontsize=8)
    else:
        ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), ncol=1, framealpha=0.9, fontsize=7)
        fig.subplots_adjust(right=0.75)

    combined_path = plot_dir / f"G{g_value}_pairwise_combined.png"
    fig.savefig(combined_path)
    plt.close(fig)
    print(f"Saved: {combined_path}")

    return str(combined_path)


def plot_bandwidth_multi_node(
    df: pd.DataFrame,
    test_name: str,
    out_dir: str,
    metric: str = "busbw_GBs",
) -> Optional[str]:
    """
    Generate bandwidth plots for multi-node (N>1) test results.
    Shows bandwidth across message sizes for different G values.
    Each line represents a different GPU count (G) value.
    """
    if df.empty:
        return None

    # Filter for multi-node results (N > 1)
    df_multi = df[df["N"] > 1].copy()
    if df_multi.empty:
        return None

    # Get N value (should be same for all rows in a multi-node run)
    n_value = df_multi["N"].iloc[0]
    g_values = sorted(df_multi["G"].unique())
    test_short = test_name.replace("_perf", "")
    metric_label = "Bus Bandwidth" if metric == "busbw_GBs" else "Algorithm Bandwidth"

    # Create output directory
    plot_dir = Path(out_dir) / "plots" / test_name
    plot_dir.mkdir(parents=True, exist_ok=True)

    # Color palette with darker, distinguishable colors
    DARK_COLORS = ['#1f77b4', '#d62728', '#2ca02c', '#ff7f0e', '#9467bd', '#8c564b', '#17becf', '#bcbd22']
    MARKERS = ['o', 's', '^', 'D', 'v', 'p', 'h', '*']

    # Combined plot for all G values
    fig, ax = plt.subplots(figsize=(12, 7))

    for i, g in enumerate(g_values):
        g_df = df_multi[df_multi["G"] == g].sort_values("size_bytes")
        if g_df.empty:
            continue

        sizes = g_df["size_bytes"].values
        bw = g_df[metric].values

        ax.plot(
            sizes, bw,
            marker=MARKERS[i % len(MARKERS)],
            linewidth=2,
            markersize=7,
            label=f"G={int(g)}",
            color=DARK_COLORS[i % len(DARK_COLORS)],
        )

        # Annotate peak
        if len(bw) > 0:
            peak_bw = bw.max()
            peak_idx = bw.argmax()
            ax.annotate(
                f"{peak_bw:.1f}",
                xy=(sizes[peak_idx], peak_bw),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=8,
                color=DARK_COLORS[i % len(DARK_COLORS)],
            )

    ax.set_xscale("log", base=2)
    ax.set_xlabel("Message Size")
    ax.set_ylabel(f"{metric_label} (GB/s)")
    ax.set_title(f"{test_short} - Multi-Node (N={int(n_value)}) Bandwidth vs Message Size")

    # Set x-axis ticks
    all_sizes = df_multi["size_bytes"].unique()
    all_sizes.sort()
    ax.set_xticks(all_sizes)
    ax.set_xticklabels([human_bytes(s) for s in all_sizes], rotation=45, ha="right")

    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", title="GPUs per Node", framealpha=0.9)

    combined_path = plot_dir / f"N{int(n_value)}_multi_node.png"
    fig.savefig(combined_path)
    plt.close(fig)
    print(f"Saved: {combined_path}")

    return str(combined_path)


def plot_all(
    df: pd.DataFrame,
    out_dir: str,
    test_filter: Optional[str] = None,
    g_filter: Optional[int] = None,
    metric: str = "busbw_GBs",
) -> List[str]:
    """
    Generate all applicable plots from the DataFrame.
    
    Args:
        df: Full DataFrame with all parsed data
        out_dir: Output directory for plots
        test_filter: Optional test name filter
        g_filter: Optional GPU count filter
        metric: Metric to plot
    
    Returns:
        List of generated plot paths
    """
    if df.empty:
        print("No data to plot.")
        return []

    setup_plot_style()
    generated_plots = []

    # Get unique test/G combinations
    tests = [test_filter] if test_filter else df["test"].unique()
    g_values = [g_filter] if g_filter else sorted(df["G"].dropna().unique())

    for test in tests:
        # Single-node and pairwise: iterate by G value
        for g in g_values:
            df_filtered = df[(df["test"] == test) & (df["G"] == g)]
            if df_filtered.empty:
                continue

            print(f"\nProcessing: {test}, G={int(g)}")

            # Check if single-node or pairwise data
            if 1 in df_filtered["N"].values:
                path = plot_bandwidth_single_node(df_filtered, test, int(g), out_dir, metric)
                if path:
                    generated_plots.append(path)

            if 2 in df_filtered["N"].values:
                path = plot_bandwidth_pairwise(df_filtered, test, int(g), out_dir, metric)
                if path:
                    generated_plots.append(path)

        # Multi-node: plot all G values together for each test
        df_test = df[df["test"] == test]
        if any(df_test["N"] > 1):
            df_multi = df_test[df_test["N"] > 1]
            if g_filter:
                df_multi = df_multi[df_multi["G"] == g_filter]
            if not df_multi.empty:
                print(f"\nProcessing multi-node: {test}")
                path = plot_bandwidth_multi_node(df_multi, test, out_dir, metric)
                if path:
                    generated_plots.append(path)

    return generated_plots


# ===========================================================
# Main
# ===========================================================


def main():
    parser = argparse.ArgumentParser(
        description="Generate bandwidth plots from NCCL benchmark logs."
    )
    parser.add_argument(
        "-i", "--input",
        required=True,
        help="Input directory containing NCCL log files",
    )
    parser.add_argument(
        "-o", "--out-dir",
        default=None,
        help="Output directory for plots. Default: same as input or parent of 'logs' dir",
    )
    parser.add_argument(
        "--test",
        default=None,
        help="Filter by test name (e.g., all_reduce_perf)",
    )
    parser.add_argument(
        "--g",
        type=int,
        default=None,
        help="Filter by GPU count (e.g., 8)",
    )
    parser.add_argument(
        "--metric",
        choices=["busbw", "algbw"],
        default="busbw",
        help="Bandwidth metric to plot: busbw (bus bandwidth) or algbw (algorithm bandwidth)",
    )
    parser.add_argument(
        "--save-csv",
        default=None,
        help="Save parsed per-message-size data to CSV",
    )
    args = parser.parse_args()

    # Use realpath to resolve symlinks, ensuring the path works from any directory
    input_path = os.path.realpath(args.input)
    
    # Determine output directory
    if args.out_dir:
        out_dir = os.path.realpath(args.out_dir)
    elif os.path.basename(input_path) == "logs":
        out_dir = os.path.dirname(input_path)
    else:
        out_dir = input_path

    print(f"Input: {input_path}")
    print(f"Output: {out_dir}")

    # Process logs
    print("\nParsing logs...")
    df = process_logs(input_path)

    if df.empty:
        print("No valid log data found.")
        return

    print(f"Parsed {len(df)} data points from {df['file'].nunique()} files")
    print(f"Tests: {df['test'].unique().tolist()}")
    print(f"G values: {sorted(df['G'].dropna().unique().tolist())}")

    # Save CSV if requested
    if args.save_csv:
        df.to_csv(args.save_csv, index=False)
        print(f"\nSaved data to: {args.save_csv}")

    # Generate plots
    metric_col = "busbw_GBs" if args.metric == "busbw" else "algbw_GBs"
    plots = plot_all(df, out_dir, args.test, args.g, metric_col)

    print(f"\n{'='*50}")
    print(f"Generated {len(plots)} plot(s)")
    for p in plots:
        print(f"  - {p}")


if __name__ == "__main__":
    main()
