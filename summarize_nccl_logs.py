"""
Parse NCCL benchmark logs and output a summary table.

Features
- Recursively scans for *.log files under --input.
- For each "Collective test starting: ..." section in a file, extracts:
  * test_name (e.g., alltoall_perf, sendrecv_perf, etc.)
  * minBytes, maxBytes (from test header)
  * Avg bus bandwidth (from summary line)
  * Peak bus bandwidth across sizes in the table (max of out-of-place/in-place busbw)
- Infers N (nodes) and G (GPUs-per-node or GPU-group) from filename suffix "_N{N}_G{G}[_node1_node2_...].log".
  Also extracts any number of node names appended after G and shows them in a "nodes" column.
- Includes "group" (top-level directory under root) and basename for traceability.
- Prints a pretty aligned table; optionally saves CSV/Markdown.

Usage
  Single directory mode:
    python summarize_nccl_logs.py --input benchmarks/cluster00/nccl-tests-pairs/with-debug/logs
    -> Generates summary.csv and summary.md in with-debug/ directory

  Batch mode:
    python summarize_nccl_logs.py --input benchmarks/cluster00/nccl-tests-pairs/
    -> Scans for with-debug/logs and without-debug/logs subdirectories
    -> Generates summary.csv and summary.md in each parent directory
"""

import argparse
import csv
import os
import re
from typing import Dict, List, Optional, Tuple

# ===========================================================
# Regex patterns
# ===========================================================

RE_SECTION_START = re.compile(r"^#\s*Collective test starting:\s*(\S+)", re.MULTILINE)
RE_HEADER_BYTES = re.compile(
    r"^#\s*nThread\s+\d+\s+nGpus\s+\d+\s+minBytes\s+(\d+)\s+maxBytes\s+(\d+)\s+step:",
    re.MULTILINE,
)
RE_AVG_BUS_BW = re.compile(r"^#\s*Avg bus bandwidth\s*:\s*([0-9.]+)", re.MULTILINE)

# Table lines look like (spaces may vary):
# size(B) count type redop root time(us) algbw(GB/s) busbw(GB/s) ... then in-place triplet
# We'll capture both out-of-place and in-place busbw columns when present.
RE_TABLE_ROW = re.compile(
    r"^\s*(\d+)\s+\d+\s+\S+\s+\S+\s+-?\d+\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+\d+\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)",
    re.MULTILINE,
)

# Some logs (different NCCL versions/ops) may omit the in-place triplet; be tolerant:
RE_TABLE_ROW_OOP_ONLY = re.compile(
    r"^\s*(\d+)\s+\d+\s+\S+\s+\S+\s+-?\d+\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+\d+\s*$",
    re.MULTILINE,
)

# Allow optional nodes suffix after _N{N}_G{G}, e.g. _N2_G8_cnode4-012_cnode4-013.log
RE_FILE_NG_NODES = re.compile(
    r"_N(\d+)_G(\d+)(?:_([^.]+?)(?<!_debug))?\.log$", re.IGNORECASE
)


# ===========================================================
# Helper functions
# ===========================================================


def human_bytes(n: Optional[int]) -> str:
    if n is None:
        return ""
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    v = float(n)
    for u in units:
        if v < 1024.0 or u == units[-1]:
            return f"{v:.0f}{u}" if u == "B" else f"{v:.1f}{u}"
        v /= 1024.0
    return f"{n}B"


def split_sections(text: str) -> List[Tuple[str, str]]:
    """
    Split the file into sections per 'Collective test starting:'.
    Returns list of (test_name, section_text).
    """
    sections: List[Tuple[str, str]] = []
    starts = [(m.start(), m.group(1)) for m in RE_SECTION_START.finditer(text)]
    if not starts:
        return sections
    for i, (pos, test_name) in enumerate(starts):
        end = starts[i + 1][0] if i + 1 < len(starts) else len(text)
        sections.append((test_name, text[pos:end]))
    return sections


def parse_section_metrics(section_text: str) -> Dict[str, Optional[float]]:
    """
    From a single test section extract:
      minBytes, maxBytes, avg_bus_bw, peak_busbw (max across sizes, considering both in-place and out-of-place)
    """
    min_b = max_b = None
    m_hdr = RE_HEADER_BYTES.search(section_text)
    if m_hdr:
        min_b = int(m_hdr.group(1))
        max_b = int(m_hdr.group(2))

    avg_bw = None
    m_avg = RE_AVG_BUS_BW.search(section_text)
    if m_avg:
        try:
            avg_bw = float(m_avg.group(1))
        except ValueError:
            avg_bw = None

    # Peak bus bandwidth from table rows
    peak_busbw = None

    def upd(val: float):
        nonlocal peak_busbw
        if val is None:
            return
        if peak_busbw is None or val > peak_busbw:
            peak_busbw = val

    # Rich rows with both OOP and in-place
    for m in RE_TABLE_ROW.finditer(section_text):
        # groups: size, time_oop, algbw_oop, busbw_oop, time_ip, algbw_ip, busbw_ip
        try:
            bus_oop = float(m.group(4))
            bus_ip = float(m.group(7))
            upd(bus_oop)
            upd(bus_ip)
        except ValueError:
            pass

    # Fallback rows that only have OOP
    for m in RE_TABLE_ROW_OOP_ONLY.finditer(section_text):
        try:
            bus_oop = float(m.group(4))
            upd(bus_oop)
        except ValueError:
            pass

    return {
        "minBytes": min_b,
        "maxBytes": max_b,
        "avg_bus_bw": avg_bw,
        "peak_busbw": peak_busbw,
    }


def infer_ng_nodes(file_path: str) -> Tuple[Optional[int], Optional[int], List[str]]:
    """
    Infer N, G, and a list of node names from filename pattern:
      ..._N{N}_G{G}[_node1_node2_...].log
    Returns (N, G, nodes_list). If absent, N/G are None and nodes_list is [].
    """
    base = os.path.basename(file_path)
    m = RE_FILE_NG_NODES.search(base)
    if not m:
        return None, None, []
    N = int(m.group(1))
    G = int(m.group(2))
    nodes_list: List[str] = []
    if m.group(3):
        # Split the whole tail by underscores; keep non-empty tokens
        # (Allow hyphens etc. inside a token; we do not further split.)
        nodes_list = [tok for tok in m.group(3).split("_") if tok]
    return N, G, nodes_list


def walk_logs(root: str) -> List[str]:
    logs: List[str] = []
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            if fn.lower().endswith(".log"):
                logs.append(os.path.join(dirpath, fn))
    logs.sort()
    return logs


def format_table(rows: List[Dict[str, object]]) -> str:
    """
    Create a simple aligned table using only stdlib.
    """
    if not rows:
        return "No rows parsed."

    headers = [
        "file",
        "test",
        "N",
        "G",
        "nodes",
        "minBytes",
        "maxBytes",
        "AvgBusBW(GB/s)",
        "PeakBusBW(GB/s)",
    ]

    def nodes_to_str(ns: Optional[List[str]]) -> str:
        if not ns:
            return ""
        return ",".join(ns)

    # Transform rows -> strings
    str_rows: List[List[str]] = []
    for r in rows:
        str_rows.append(
            [
                str(r.get("file", "")),
                str(r.get("test", "")),
                str(r.get("N", "") if r.get("N") is not None else ""),
                str(r.get("G", "") if r.get("G") is not None else ""),
                nodes_to_str(r.get("nodes")),
                human_bytes(r.get("minBytes")),
                human_bytes(r.get("maxBytes")),
                f"{r.get('avg_bus_bw'):.3f}" if r.get("avg_bus_bw") is not None else "",
                f"{r.get('peak_busbw'):.3f}" if r.get("peak_busbw") is not None else "",
            ]
        )

    cols = list(zip(*([headers] + str_rows)))
    col_widths = [max(len(x) for x in col) for col in cols]

    def fmt_row(vals: List[str]) -> str:
        return " | ".join(val.ljust(w) for val, w in zip(vals, col_widths))

    sep = "-+-".join("-" * w for w in col_widths)
    lines = [fmt_row(headers), sep]
    for sr in str_rows:
        lines.append(fmt_row(sr))
    return "\n".join(lines)


def save_csv(rows: List[Dict[str, object]], path: str) -> None:
    headers = [
        "file",
        "test",
        "N",
        "G",
        "nodes",
        "minBytes",
        "maxBytes",
        "avg_bus_bw_GBs",
        "peak_busbw_GBs",
    ]
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(headers)
        for r in rows:
            nodes_str = ",".join(r.get("nodes", [])) if r.get("nodes") else ""
            w.writerow(
                [
                    r.get("file", ""),
                    r.get("test", ""),
                    r.get("N", ""),
                    r.get("G", ""),
                    nodes_str,
                    r.get("minBytes", ""),
                    r.get("maxBytes", ""),
                    f"{r.get('avg_bus_bw'):.6f}"
                    if r.get("avg_bus_bw") is not None
                    else "",
                    f"{r.get('peak_busbw'):.6f}"
                    if r.get("peak_busbw") is not None
                    else "",
                ]
            )


def save_md(rows: List[Dict[str, object]], path: str) -> None:
    headers = [
        "file",
        "test",
        "N",
        "G",
        "nodes",
        "minBytes",
        "maxBytes",
        "AvgBusBW (GB/s)",
        "PeakBusBW (GB/s)",
    ]

    def nodes_to_str(ns: Optional[List[str]]) -> str:
        if not ns:
            return ""
        return ",".join(ns)

    def row_to_list(r: Dict[str, object]) -> List[str]:
        return [
            str(r.get("file", "")),
            str(r.get("test", "")),
            str(r.get("N", "")) if r.get("N") is not None else "",
            str(r.get("G", "")) if r.get("G") is not None else "",
            nodes_to_str(r.get("nodes")),
            human_bytes(r.get("minBytes")),
            human_bytes(r.get("maxBytes")),
            f"{r.get('avg_bus_bw'):.3f}" if r.get("avg_bus_bw") is not None else "",
            f"{r.get('peak_busbw'):.3f}" if r.get("peak_busbw") is not None else "",
        ]

    with open(path, "w") as f:
        f.write("| " + " | ".join(headers) + " |\n")
        f.write("|" + "|".join(["---"] * len(headers)) + "|\n")
        for r in rows:
            f.write("| " + " | ".join(row_to_list(r)) + " |\n")


def process_logs_in_dir(root: str) -> List[Dict[str, object]]:
    """
    Process all logs in a directory and return parsed rows.
    """
    logs = walk_logs(root)
    if not logs:
        return []

    rows: List[Dict[str, object]] = []

    for path in logs:
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as fh:
                text = fh.read()
        except Exception as e:
            print(f"Failed to read {path}: {e}")
            continue

        sections = split_sections(text)
        if not sections:
            continue

        N, G, nodes_list = infer_ng_nodes(path)

        for test_name, section_text in sections:
            metrics = parse_section_metrics(section_text)

            row = {
                "file": os.path.basename(path),
                "test": test_name,
                "N": N,
                "G": G,
                "nodes": nodes_list,
                "minBytes": metrics["minBytes"],
                "maxBytes": metrics["maxBytes"],
                "avg_bus_bw": metrics["avg_bus_bw"],
                "peak_busbw": metrics["peak_busbw"],
            }
            rows.append(row)

    # Sort for stable output: by test -> N -> G -> file
    def sort_key(r: Dict[str, object]):
        return (
            str(r.get("test", "")),
            int(r["N"]) if r.get("N") is not None else 0,
            int(r["G"]) if r.get("G") is not None else 0,
            str(r.get("file", "")),
        )

    rows.sort(key=sort_key)
    return rows


def save_results(rows: List[Dict[str, object]], output_dir: str, label: str = ""):
    """
    Save results to CSV and Markdown in the specified directory.
    """
    if not rows:
        print(f"No logs found for {label}")
        return

    csv_path = os.path.join(output_dir, "summary.csv")
    md_path = os.path.join(output_dir, "summary.md")

    try:
        save_csv(rows, csv_path)
        print(f"Saved CSV: {csv_path}")
    except Exception as e:
        print(f"Failed to save CSV to {csv_path}: {e}")

    try:
        save_md(rows, md_path)
        print(f"Saved Markdown: {md_path}")
    except Exception as e:
        print(f"Failed to save Markdown to {md_path}: {e}")


# ===========================================================
# Main function
# ===========================================================


def main():
    ap = argparse.ArgumentParser(description="Parse NCCL logs and summarize results.")
    ap.add_argument(
        "-i", "--input", default=".", help="Input directory to scan (default: .)"
    )
    ap.add_argument(
        "--save-csv",
        default=None,
        help="Optional path to save CSV (overrides auto-detection)",
    )
    ap.add_argument(
        "--save-md",
        default=None,
        help="Optional path to save Markdown table (overrides auto-detection)",
    )
    args = ap.parse_args()

    input_path = os.path.abspath(args.input)

    with_debug_logs = os.path.join(input_path, "with-debug", "logs")
    without_debug_logs = os.path.join(input_path, "without-debug", "logs")

    batch_mode = (
        os.path.isdir(with_debug_logs) or os.path.isdir(without_debug_logs)
    ) and os.path.basename(input_path) != "logs"

    if batch_mode:
        if os.path.isdir(with_debug_logs):
            print("Processing with-debug logs:\n")
            rows = process_logs_in_dir(with_debug_logs)
            print(format_table(rows))
            print()
            save_results(rows, os.path.join(input_path, "with-debug"), "with-debug")
            print()
        if os.path.isdir(without_debug_logs):
            print("Processing without-debug logs:\n")
            rows = process_logs_in_dir(without_debug_logs)
            print(format_table(rows))
            print()
            save_results(
                rows, os.path.join(input_path, "without-debug"), "without-debug"
            )
    else:
        if os.path.basename(input_path) == "logs":
            output_dir = os.path.dirname(input_path)
        else:
            output_dir = input_path

        rows = process_logs_in_dir(input_path)

        if not rows:
            print("No .log files found.")
            return

        print(format_table(rows))
        print()

        if args.save_csv or args.save_md:
            if args.save_csv:
                try:
                    save_csv(rows, args.save_csv)
                    print(f"Saved CSV: {args.save_csv}")
                except Exception as e:
                    print(f"Failed to save CSV: {e}")

            if args.save_md:
                try:
                    save_md(rows, args.save_md)
                    print(f"Saved Markdown: {args.save_md}")
                except Exception as e:
                    print(f"Failed to save Markdown: {e}")
        else:
            save_results(rows, output_dir)


if __name__ == "__main__":
    main()
