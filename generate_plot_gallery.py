#!/usr/bin/env python3
"""
Generate an HTML or Markdown gallery for NCCL plot artifacts.

Scans benchmarks/<cluster>/... for plots under:
  - */plots/<test_name>/*.png
  - */topology/<test_name>/*.png

Supports both layouts:
  benchmarks/<cluster>/<test-type>/runs/<run-id>/...
  benchmarks/<cluster>/nccl-benchmark-results/<test-type>/runs/<run-id>/...
"""

from __future__ import annotations

import argparse
import os
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


PLOT_DIR_NAMES = {"plots", "topology"}
IMAGE_EXTS = {".png", ".jpg", ".jpeg"}


@dataclass(frozen=True)
class PlotItem:
    cluster: str
    test_type: str
    run_id: str
    debug: str
    kind: str
    test_name: str
    file_name: str
    abs_path: Path
    rel_path: str

    @property
    def title(self) -> str:
        parts = [self.test_name, self.file_name]
        return " / ".join([p for p in parts if p])

    @property
    def tag_string(self) -> str:
        tokens = [
            self.cluster,
            self.test_type,
            self.run_id,
            self.debug,
            self.kind,
            self.test_name,
            self.file_name,
        ]
        return " ".join(t for t in tokens if t).lower()


def parse_metadata(path: Path, benchmarks_dir: Path) -> Optional[Dict[str, str]]:
    try:
        rel = path.relative_to(benchmarks_dir)
    except ValueError:
        return None

    parts = list(rel.parts)
    if len(parts) < 4:
        return None

    cluster = parts[0]
    rest = parts[1:]
    if rest and rest[0] == "nccl-benchmark-results":
        rest = rest[1:]
        if not rest:
            return None

    test_type = rest[0]
    run_id = ""
    debug = ""
    kind = ""
    test_name = ""

    if "runs" in rest:
        idx = rest.index("runs")
        if idx + 1 < len(rest):
            run_id = rest[idx + 1]

    if "without-debug" in rest:
        debug = "without-debug"
    elif "with-debug" in rest:
        debug = "with-debug"

    for k in PLOT_DIR_NAMES:
        if k in rest:
            kind = k
            k_idx = rest.index(k)
            if k_idx + 1 < len(rest):
                test_name = rest[k_idx + 1]
            break

    if not kind:
        return None

    return {
        "cluster": cluster,
        "test_type": test_type,
        "run_id": run_id or "unknown-run",
        "debug": debug or "unknown-debug",
        "kind": kind,
        "test_name": test_name or "unknown-test",
    }


def discover_plot_files(benchmarks_dir: Path, clusters: Optional[List[str]]) -> List[Path]:
    paths: List[Path] = []
    if not benchmarks_dir.exists():
        return paths

    wanted = {c.strip() for c in clusters} if clusters else None

    for root, _, files in os.walk(benchmarks_dir):
        root_path = Path(root)
        if wanted:
            try:
                rel = root_path.relative_to(benchmarks_dir)
            except ValueError:
                continue
            if not rel.parts or rel.parts[0] not in wanted:
                continue
        for fn in files:
            ext = Path(fn).suffix.lower()
            if ext not in IMAGE_EXTS:
                continue
            paths.append(root_path / fn)
    return paths


def build_items(
    benchmarks_dir: Path,
    output_path: Path,
    clusters: Optional[List[str]],
    include_plots: bool,
    include_topology: bool,
) -> List[PlotItem]:
    items: List[PlotItem] = []
    output_dir = output_path.parent
    plot_paths = discover_plot_files(benchmarks_dir, clusters)
    for path in plot_paths:
        meta = parse_metadata(path, benchmarks_dir)
        if not meta:
            continue
        if meta["kind"] == "plots" and not include_plots:
            continue
        if meta["kind"] == "topology" and not include_topology:
            continue
        rel_path = os.path.relpath(path, output_dir)
        items.append(
            PlotItem(
                cluster=meta["cluster"],
                test_type=meta["test_type"],
                run_id=meta["run_id"],
                debug=meta["debug"],
                kind=meta["kind"],
                test_name=meta["test_name"],
                file_name=path.name,
                abs_path=path,
                rel_path=rel_path,
            )
        )
    items.sort(key=lambda x: (x.cluster, x.test_type, x.run_id, x.kind, x.test_name, x.file_name))
    return items


def group_items(items: List[PlotItem]) -> Dict[Tuple[str, str, str, str], Dict[str, List[PlotItem]]]:
    grouped: Dict[Tuple[str, str, str, str], Dict[str, List[PlotItem]]] = {}
    for item in items:
        top_key = (item.cluster, item.test_type, item.run_id, item.debug)
        grouped.setdefault(top_key, {}).setdefault(item.test_name, []).append(item)
    return grouped


def render_html(items: List[PlotItem], output_path: Path) -> None:
    grouped = group_items(items)
    clusters = sorted({i.cluster for i in items})
    test_types = sorted({i.test_type for i in items})
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    html: List[str] = []
    html.append("<!doctype html>")
    html.append("<html lang=\"en\">")
    html.append("<head>")
    html.append("  <meta charset=\"utf-8\">")
    html.append("  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">")
    html.append("  <title>NCCL Plot Gallery</title>")
    html.append("  <style>")
    html.append("    :root {")
    html.append("      --bg: #f6f4ef;")
    html.append("      --card: #ffffff;")
    html.append("      --ink: #1e1e1e;")
    html.append("      --muted: #666;")
    html.append("      --accent: #0f766e;")
    html.append("    }")
    html.append("    body { margin: 0; font-family: \"Source Sans 3\", \"Helvetica Neue\", Arial, sans-serif; color: var(--ink); background: var(--bg); }")
    html.append("    header { padding: 24px 28px 16px; background: linear-gradient(120deg, #fef7ed, #ecfeff); border-bottom: 1px solid #e5e7eb; }")
    html.append("    h1 { margin: 0 0 8px; font-size: 28px; }")
    html.append("    .controls { display: flex; flex-wrap: wrap; gap: 12px; align-items: center; margin-top: 12px; }")
    html.append("    .controls label { font-size: 14px; color: var(--muted); }")
    html.append("    .controls input, .controls select { padding: 6px 8px; border-radius: 6px; border: 1px solid #d1d5db; }")
    html.append("    .meta { color: var(--muted); font-size: 13px; }")
    html.append("    main { padding: 20px 28px 40px; }")
    html.append("    section.cluster { margin-bottom: 32px; }")
    html.append("    section.cluster > h2 { margin: 0 0 6px; font-size: 22px; }")
    html.append("    .group { margin: 12px 0 20px; padding: 12px 14px; background: var(--card); border-radius: 12px; box-shadow: 0 2px 6px rgba(0,0,0,0.05); }")
    html.append("    .group h3 { margin: 0 0 6px; font-size: 18px; }")
    html.append("    .group h4 { margin: 6px 0 10px; font-size: 14px; color: var(--muted); }")
    html.append("    details { margin-bottom: 10px; }")
    html.append("    summary { cursor: pointer; font-weight: 600; margin-bottom: 6px; }")
    html.append("    .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 12px; }")
    html.append("    figure.card { margin: 0; background: #fff; border-radius: 10px; border: 1px solid #e5e7eb; overflow: hidden; }")
    html.append("    figure.card img { width: 100%; height: auto; display: block; }")
    html.append("    figure.card figcaption { padding: 8px 10px; font-size: 12px; color: var(--muted); }")
    html.append("    figure.card a { color: inherit; text-decoration: none; }")
    html.append("    .tagline { font-size: 12px; color: var(--muted); }")
    html.append("    .hidden { display: none !important; }")
    html.append("  </style>")
    html.append("</head>")
    html.append("<body>")
    html.append("  <header>")
    html.append("    <h1>NCCL Plot Gallery</h1>")
    html.append(f"    <div class=\"meta\">Generated {now}</div>")
    html.append("    <div class=\"controls\">")
    html.append("      <label>Search <input id=\"search\" type=\"text\" placeholder=\"cluster, test, G8...\" /></label>")
    html.append("      <label>Cluster <select id=\"clusterFilter\">")
    html.append("        <option value=\"all\">All</option>")
    for c in clusters:
        html.append(f"        <option value=\"{c}\">{c}</option>")
    html.append("      </select></label>")
    html.append("      <label>Test type <select id=\"typeFilter\">")
    html.append("        <option value=\"all\">All</option>")
    for t in test_types:
        html.append(f"        <option value=\"{t}\">{t}</option>")
    html.append("      </select></label>")
    html.append("    </div>")
    html.append("  </header>")
    html.append("  <main id=\"gallery\">")

    for (cluster, test_type, run_id, debug), tests in grouped.items():
        html.append(f"    <section class=\"cluster\" data-cluster=\"{cluster}\">")
        html.append(f"      <h2>{cluster}</h2>")
        html.append(f"      <div class=\"group\" data-type=\"{test_type}\" data-run=\"{run_id}\">")
        html.append(f"        <h3>{test_type} / {run_id}</h3>")
        html.append(f"        <h4>{debug}</h4>")
        for test_name, items_in_test in tests.items():
            html.append("        <details open>")
            html.append(f"          <summary>{test_name}</summary>")
            html.append("          <div class=\"grid\">")
            for item in items_in_test:
                html.append(
                    "            <figure class=\"card\" data-tags=\"{}\">".format(
                        item.tag_string
                    )
                )
                html.append(f"              <a href=\"{item.rel_path}\" target=\"_blank\" rel=\"noopener\">")
                html.append(f"                <img src=\"{item.rel_path}\" loading=\"lazy\" alt=\"{item.title}\">")
                html.append("              </a>")
                html.append(
                    f"              <figcaption>{item.test_name} · {item.file_name}</figcaption>"
                )
                html.append("            </figure>")
            html.append("          </div>")
            html.append("        </details>")
        html.append("      </div>")
        html.append("    </section>")

    html.append("  </main>")
    html.append("  <script>")
    html.append("    const searchInput = document.getElementById('search');")
    html.append("    const clusterFilter = document.getElementById('clusterFilter');")
    html.append("    const typeFilter = document.getElementById('typeFilter');")
    html.append("    function applyFilters() {")
    html.append("      const term = searchInput.value.trim().toLowerCase();")
    html.append("      const cluster = clusterFilter.value;")
    html.append("      const type = typeFilter.value;")
    html.append("      document.querySelectorAll('figure.card').forEach(card => {")
    html.append("        const tags = card.dataset.tags || '';")
    html.append("        const clusterOk = cluster === 'all' || tags.includes(cluster.toLowerCase());")
    html.append("        const typeOk = type === 'all' || tags.includes(type.toLowerCase());")
    html.append("        const termOk = !term || tags.includes(term);")
    html.append("        card.classList.toggle('hidden', !(clusterOk && typeOk && termOk));")
    html.append("      });")
    html.append("    }")
    html.append("    searchInput.addEventListener('input', applyFilters);")
    html.append("    clusterFilter.addEventListener('change', applyFilters);")
    html.append("    typeFilter.addEventListener('change', applyFilters);")
    html.append("  </script>")
    html.append("</body>")
    html.append("</html>")

    output_path.write_text("\n".join(html), encoding="utf-8")


def render_markdown(items: List[PlotItem], output_path: Path) -> None:
    grouped = group_items(items)
    lines: List[str] = []
    lines.append(f"# NCCL Plot Gallery")
    lines.append("")
    lines.append(f"Generated {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")

    for (cluster, test_type, run_id, debug), tests in grouped.items():
        lines.append(f"## {cluster}")
        lines.append(f"### {test_type} / {run_id} ({debug})")
        lines.append("")
        for test_name, items_in_test in tests.items():
            lines.append(f"#### {test_name}")
            lines.append("")
            for item in items_in_test:
                lines.append(f"![{item.title}]({item.rel_path})")
                lines.append(f"*{item.file_name}*")
                lines.append("")
        lines.append("")

    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a gallery of NCCL plots.")
    parser.add_argument(
        "--benchmarks-dir",
        default="benchmarks",
        help="Path to benchmarks directory (default: benchmarks)",
    )
    parser.add_argument(
        "--clusters",
        default=None,
        help="Comma-separated cluster names to include (default: all)",
    )
    parser.add_argument(
        "--output",
        default="benchmarks/plot-gallery.html",
        help="Output file (default: benchmarks/plot-gallery.html)",
    )
    parser.add_argument(
        "--format",
        choices=["html", "md"],
        default="html",
        help="Output format: html or md (default: html)",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Exclude bandwidth plots (plots/)",
    )
    parser.add_argument(
        "--no-topology",
        action="store_true",
        help="Exclude topology heatmaps (topology/)",
    )
    args = parser.parse_args()

    benchmarks_dir = Path(args.benchmarks_dir).resolve()
    output_path = Path(args.output).resolve()

    clusters = None
    if args.clusters:
        clusters = [c.strip() for c in args.clusters.split(",") if c.strip()]

    items = build_items(
        benchmarks_dir=benchmarks_dir,
        output_path=output_path,
        clusters=clusters,
        include_plots=not args.no_plots,
        include_topology=not args.no_topology,
    )

    if not items:
        print("No plot images found.")
        return

    if args.format == "html":
        render_html(items, output_path)
    else:
        render_markdown(items, output_path)

    print(f"Wrote gallery: {output_path}")


if __name__ == "__main__":
    main()
