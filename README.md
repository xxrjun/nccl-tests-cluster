<h1 align="center">
NCCL Tests Cluster
</h1>

<p align="center">
Automated inter-node bandwidth testing and visualization for GPU clusters using NCCL.
</p>

<p align="center">
  <img src="./assets/17node_heatmap_alltoall_allG.png"
       alt="Example heatmap of an 17-node H100 cluster"
       width="700" />
  <br/>
  <sub>Example: 17  -node H100 cluster bandwidth heatmap (alltoall_perf)</sub>
</p>

**Key Features:**

- Run **[NCCL Tests](https://github.com/NVIDIA/nccl-tests)** in single-node, multi-node, and pairwise modes
- Parse logs into structured CSV/Markdown reports
- Visualize bandwidth with heatmaps and plots
- Full **[SLURM](https://slurm.schedmd.com/documentation.html) **integration

### Test Types at a Glance

| Test Type       | Purpose                       | Best For                                  |
| --------------- | ----------------------------- | ----------------------------------------- |
| **Single-node** | Intra-node GPU communication  | Verify each node works correctly          |
| **Pairwise**    | All N×(N-1)/2 node pairs      | **Diagnose network issues** between nodes |
| **Multi-node**  | N nodes in one collective job | Benchmark overall cluster performance     |
| **Smoke**       | Quick 2-node sanity check     | Fast validation before full tests         |

### Feature Support Matrix

| Feature         | Single-Node | Pairwise | Multi-Node | Smoke |
| --------------- | :---------: | :------: | :--------: | :---: |
| Run NCCL Tests  |     ✅      |    ✅    |     ✅     |  ✅   |
| Summarize Logs  |     ✅      |    ✅    |     ✅     |  ✅   |
| Bandwidth Plots |     ✅      |    —     |     ✅     |   —   |
| Heatmaps        |      —      |    ✅    |     —      |   —   |

> **Note:** Bandwidth plots show bandwidth vs message size. Single-node compares different nodes; multi-node compares different G values. Heatmaps visualize inter-node bandwidth matrix (requires pairwise data).

## Table of Contents <!-- omit in toc -->

- [Quick Start](#quick-start)
  - [Setup](#setup)
  - [Run Tests](#run-tests)
  - [Wait \& Process](#wait--process)
- [Workflow](#workflow)
- [Output Structure](#output-structure)
- [Prerequisites](#prerequisites)
  - [Clone Repository and Build NCCL](#clone-repository-and-build-nccl)
  - [Python Environment](#python-environment)
- [Usage](#usage)
  - [Run NCCL Tests (Single-Node)](#run-nccl-tests-single-node)
  - [Run NCCL Tests (Pairs)](#run-nccl-tests-pairs)
  - [Run NCCL Tests (Multi-Node)](#run-nccl-tests-multi-node)
  - [Quick Smoke Test](#quick-smoke-test)
  - [Summarize Logs](#summarize-logs)
  - [Generate Bandwidth Plots](#generate-bandwidth-plots)
  - [Generate Heatmaps](#generate-heatmaps)
  - [Generate Plot Gallery](#generate-plot-gallery)
- [Configuration](#configuration)
  - [Default Test Parameters](#default-test-parameters)
  - [Default Test Binaries](#default-test-binaries)
  - [Environment Variable Overrides](#environment-variable-overrides)
- [Limitations](#limitations)
- [Links](#links)
- [Troubleshooting](#troubleshooting)
  - [Common Issues](#common-issues)

## Quick Start

> **All commands run from repository root** (`nccl-tests-cluster/`).

### Setup

```bash
git clone https://github.com/xxrjun/nccl-tests-cluster.git
cd nccl-tests-cluster

# Build NCCL and tests
bash build_nccl_and_tests.sh

# Python environment
uv venv && source .venv/bin/activate && uv pip install -r requirements.txt
# Or: python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
```

### Run Tests

Choose the test type that fits your goal:

```bash
# Optional: reduce repetition in commands
export PARTITION="<partition>"
export CLUSTER="<cluster>"
```

**Pairwise (recommended for network diagnostics):**

```bash
bash sbatch_run_nccl_tests_pairs.sh -p "$PARTITION" -c "$CLUSTER" -n "node[01-04]"
```

**Single-node (intra-node GPU verification):**

```bash
bash sbatch_run_nccl_tests_single.sh -p "$PARTITION" -c "$CLUSTER"
```

**Multi-node (collective benchmark):**

```bash
bash sbatch_run_nccl_tests_multi.sh -p "$PARTITION" -c "$CLUSTER" --num-nodes 4 --gpn "8"
```

**Smoke test (quick sanity check):**

```bash
bash sbatch_run_nccl_tests_smoke.sh -p "$PARTITION" -c "$CLUSTER" -n "node1,node2"
```

### Wait & Process

```bash
# Monitor jobs
squeue -u $USER
watch -n 30 squeue -u $USER  # Auto-refresh

# After jobs complete:

# 1. Summarize logs (all test types)
python3 summarize_nccl_logs.py \
  --input benchmarks/$CLUSTER/nccl-benchmark-results/<test-type>/latest/without-debug/logs

# 2. Visualizations (test-type specific)
# Single-node -> bandwidth plots:
python3 plot_nccl_bandwidth.py \
  --input benchmarks/$CLUSTER/nccl-benchmark-results/single-node/latest/without-debug/logs

# Pairwise -> heatmaps:
python3 generate_topology.py \
  --csv benchmarks/$CLUSTER/nccl-benchmark-results/pairwise/latest/without-debug/summary.csv --all

# Plot gallery (bandwidth plots + heatmaps):
python3 generate_plot_gallery.py \
  --clusters "$CLUSTER" --output benchmarks/plot-gallery.html
```

## Workflow

The workflow requires **three separate steps** because SLURM jobs run asynchronously:

1. **Submit** — Run `sbatch_run_nccl_tests_*.sh` to submit SLURM jobs
2. **Wait** — Monitor with `squeue -u $USER` until jobs complete (minutes to hours)
3. **Process** — Run Python scripts to summarize logs and generate visualizations

> Jobs run asynchronously, so you can submit and return later to process results.

## Output Structure

Results are organized by **test type first, then run ID**, so you can browse or cleanly remove entire test classes. Each test type has its own `latest` symlink pointing to the most recent run.

```
nccl-tests-cluster/                    # Repository root (run all commands from here)
├── benchmarks/
│   └── {cluster_name}/
│       └── nccl-benchmark-results/
│           ├── single-node/
│           │   ├── runs/
│           │   │   └── <RUN_ID>/
│           │   │       ├── without-debug/
│           │   │       │   ├── logs/           # Raw NCCL test outputs
│           │   │       │   ├── summary.csv     # Parsed results
│           │   │       │   ├── summary.md      # Markdown table
│           │   │       │   └── plots/          # Bandwidth plots
│           │   │       └── with-debug/
│           │   │           └── ...
│           │   └── latest -> runs/<RUN_ID>     # Relative symlink
│           ├── pairwise/
│           │   ├── runs/<RUN_ID>/
│           │   │   └── without-debug/
│           │   │       ├── logs/
│           │   │       ├── summary.csv
│           │   │       ├── summary.md
│           │   │       ├── failures.txt        # If any tests failed
│           │   │       └── topology/           # Heatmaps
│           │   └── latest -> runs/<RUN_ID>
│           ├── multi-node/
│           │   ├── runs/<RUN_ID>/...
│           │   └── latest -> runs/<RUN_ID>
│           └── smoke/
│               ├── runs/<RUN_ID>/logs
│               └── latest -> runs/<RUN_ID>
├── nccl/
│   ├── build/                         # NCCL build (NCCL_HOME)
│   └── nccl-tests/
│       └── build/                     # NCCL test binaries (NCCL_TEST)
├── lib/
│   └── nccl_common.sh                 # Shared shell functions
├── sbatch_run_nccl_tests_single.sh    # Single-node test script
├── sbatch_run_nccl_tests_pairs.sh     # Pairwise test script
├── sbatch_run_nccl_tests_multi.sh     # Multi-node test script
├── sbatch_run_nccl_tests_smoke.sh     # Quick smoke test script
├── summarize_nccl_logs.py             # Log parser
├── plot_nccl_bandwidth.py             # Bandwidth plot generator
├── generate_topology.py               # Heatmap/topology generator
└── build_nccl_and_tests.sh            # Build script
```

**Notes:**

- The `latest` symlink uses a **relative path** (`runs/<run-id>`) to work correctly from any directory
- Reuse `--run-id` to resume and fill in missing logs for a prior run
- Each test type maintains its own independent `latest` symlink

## Prerequisites

### Clone Repository and Build NCCL

For convenience, it is recommended to clone this repository into `$HOME/` by default. Otherwise, you might need to modify the paths in `sbatch_run_nccl_tests_pairs.sh` accordingly.

```bash
cd $HOME
git clone https://github.com/xxrjun/nccl-tests-cluster.git
cd nccl-tests-cluster
```

> [!TIP]
> This project is build on [NVIDIA/nccl](https://github.com/nvidia/nccl) and [NVIDIA/nccl-tests](https://github.com/NVIDIA/nccl-tests). Please refer to their README files for more information about NCCL and NCCL tests.
>
> Or you can run with the provided build script `build_nccl_and_tests.sh` to build NCCL and NCCL tests automatically.

```bash
bash build_nccl_and_tests.sh
```

### Python Environment

Install required packages for log parsing and visualization.

**Option 1: Using [uv](https://docs.astral.sh/uv/) (recommended)**

If you don't have `uv` installed, you can install it via

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
```

Create and activate a virtual environment, then install the required packages

```bash
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

**Option 2: Using pip**:

```bash
pip install -r requirements.txt
```

## Usage

### Run NCCL Tests (Single-Node)

Test intra-node GPU communication performance on individual nodes.

**View help:**

```bash
bash sbatch_run_nccl_tests_single.sh --help
```

**Basic usage:**

```bash
# Test all nodes in a partition with default GPU counts (4, 8)
bash sbatch_run_nccl_tests_single.sh -p <partition> -c cluster00

# Test specific nodes
bash sbatch_run_nccl_tests_single.sh -p <partition> -c cluster00 -n "cnode-[001-004]"

# Custom GPU counts
bash sbatch_run_nccl_tests_single.sh -p <partition> -c cluster00 --gpn "2 4 8"

# Dry run (preview without submitting)
bash sbatch_run_nccl_tests_single.sh -p <partition> -c cluster00 --dry-run

# Enable debug mode
bash sbatch_run_nccl_tests_single.sh -p <partition> -c cluster00 --debug

# Resume a prior run (skips existing logs)
bash sbatch_run_nccl_tests_single.sh -p <partition> -c cluster00 --run-id 20250114-153012
```

**Example output:**

```bash
Submitting 4 single-node jobs...
  cnode-001
  cnode-002
  cnode-003
  cnode-004
Submit: NCCL_N1_G4_cnode-001  --nodelist=cnode-001  --gpus-per-node=4
Submitted batch job 1234
# ...
==========================================
Submission Summary
==========================================
Total nodes:    4
Jobs per node:  2
Total jobs:     8
Submitted:      8
Skipped:        0
DRY RUN:        0
NCCL DEBUG:     0
==========================================
```

### Run NCCL Tests (Pairs)

Test inter-node GPU communication performance across all node pairs.

**View help:**

```bash
bash sbatch_run_nccl_tests_pairs.sh --help
```

**Basic usage:**

```bash
# Test all node pairs in a partition with default GPU counts (1, 2, 4, 8)
bash sbatch_run_nccl_tests_pairs.sh -p <partition> -c cluster00

# Test specific nodes
bash sbatch_run_nccl_tests_pairs.sh -p <partition> -c cluster00 -n "cnode-[001-004]"

# Custom GPU counts
bash sbatch_run_nccl_tests_pairs.sh -p <partition> -c cluster00 --gpn "2 4 8"

# Dry run (preview without submitting)
bash sbatch_run_nccl_tests_pairs.sh -p <partition> -c cluster00 --dry-run

# Enable debug mode
bash sbatch_run_nccl_tests_pairs.sh -p <partition> -c cluster00 --debug

# Resume a prior run (skips existing logs)
bash sbatch_run_nccl_tests_pairs.sh -p <partition> -c cluster00 --run-id 20250114-153012
```

> [!TIP]
> It is highly recommended to first test with only two nodes to verify that your NCCL environment is working correctly:
>
> ```bash
> bash sbatch_run_nccl_tests_pairs.sh -p <partition> -c cluster00 -n "cnode-[001-002]"
> ```

**Example output:**

```bash
Submitting 6 pairs...
  cnode-001,cnode-002
  cnode-001,cnode-003
  # ...
==========================================
Submission Summary
==========================================
Total pairs:    6
Jobs per pair:  4
Total jobs:     24
Submitted:      24
Skipped:        0
DRY RUN:        0
NCCL DEBUG:     0
==========================================
```

**Cancel jobs if needed:**

```bash
scancel -u $USER
```

**Common CLI Options:**
| Option | Description | Default |
| ------------------ | --------------------------------------------------------- | ------------------------------------------------------------------------------------------ |
| `-p, --partition` | SLURM partition name | Required |
| `-c, --cluster` | Cluster name for log organization | `cluster00` |
| `-n, --nodelist` | Compressed nodelist (e.g., `"cnode-[001-004]"`) | All nodes in partition |
| `-r, --run-id` | Run ID for timestamped results | `YYYYMMDD-HHMMSS` |
| `-l, --log-dir` | Custom log directory | `benchmarks/<CLUSTER>/nccl-benchmark-results/<test-type>/runs/<RUN_ID>/without-debug/logs` |
| `--gpn` | Space-separated GPU counts | Single: `"4 8"`, Pairs: `"1 2 4 8"` |
| `--dry-run` | Preview commands without submitting | `false` |
| `--debug` | Enable NCCL debug mode (affects performance) | `false` |
| `--gpn` (comma ok) | GPU counts can also be comma-separated, e.g., `"1,2,4,8"` |

### Run NCCL Tests (Multi-Node)

Run one NCCL job across N>=2 nodes (not all pair combinations).

```bash
# Use first 4 nodes in the partition, 8 GPUs per node
bash sbatch_run_nccl_tests_multi.sh -p <partition> -c cluster00 --num-nodes 4 --gpn "8"

# Explicit nodelist
bash sbatch_run_nccl_tests_multi.sh -p <partition> -c cluster00 -n "cnode-[001-004]"

# Debug mode and custom tests (space-separated list)
RUN_BIN_LIST="all_reduce_perf all_gather_perf" \
bash sbatch_run_nccl_tests_multi.sh -p <partition> -c cluster00 --num-nodes 8 --gpn "4"

# Dry run
bash sbatch_run_nccl_tests_multi.sh -p <partition> --dry-run
```

### Quick Smoke Test

Fast two-node sanity check (all_reduce_perf + sendrecv_perf, small message sizes).

```bash
# Default: first two nodes in the partition, 1 GPU per node
bash sbatch_run_nccl_tests_smoke.sh -p <partition> -c cluster00

# Explicit nodes and debug
bash sbatch_run_nccl_tests_smoke.sh -p <partition> -c cluster00 -n "cnode-[001-002]" --debug
```

### Summarize Logs

Parse NCCL test logs and generate summary reports (CSV + Markdown). All paths are resolved automatically (symlinks included), so commands work from the repository root.

```bash
# Process single-node test logs (latest run)
python3 summarize_nccl_logs.py \
  --input benchmarks/<cluster-name>/nccl-benchmark-results/single-node/latest/without-debug/logs

# Process pairwise test logs (latest run)
python3 summarize_nccl_logs.py \
  --input benchmarks/<cluster-name>/nccl-benchmark-results/pairwise/latest/without-debug/logs

# Batch mode: process both with-debug/ and without-debug/ for a run
python3 summarize_nccl_logs.py \
  --input benchmarks/<cluster-name>/nccl-benchmark-results/pairwise/latest

# Custom output paths
python3 summarize_nccl_logs.py \
  --input benchmarks/.../logs \
  --save-csv /path/to/summary.csv \
  --save-md  /path/to/summary.md
```

**Filename Format:**

- Single-node: `..._N1_G{G}_node.log` (e.g., `nccl_N1_G8_cnode-001.log`)
- Pairs: `..._N2_G{G}_node1_node2.log` (e.g., `nccl_N2_G8_cnode-005_cnode-006.log`)
- Multi-node: `..._N{N}_G{G}.log` (e.g., `nccl_N4_G8.log`)
- The `_debug` suffix is automatically ignored

### Generate Bandwidth Plots

Generate bandwidth vs message size plots for visualizing NCCL performance trends. All paths are resolved automatically (symlinks included), so commands work from the repository root.

```bash
# Generate plots from single-node logs (all tests and G values)
python3 plot_nccl_bandwidth.py \
  --input benchmarks/<cluster-name>/nccl-benchmark-results/single-node/latest/without-debug/logs

# Filter by specific test
python3 plot_nccl_bandwidth.py --input ./logs --test all_reduce_perf

# Filter by GPU count
python3 plot_nccl_bandwidth.py --input ./logs --g 8

# Use algorithm bandwidth instead of bus bandwidth
python3 plot_nccl_bandwidth.py --input ./logs --metric algbw

# Save parsed data to CSV for further analysis
python3 plot_nccl_bandwidth.py --input ./logs --save-csv detailed_data.csv
```

**Output:** `plots/{test_name}/G{n}_{node}.png` (individual) + `G{n}_combined.png` (comparison)

**Key Options:**

- `--test NAME`: Filter by test name (e.g., `all_reduce_perf`)
- `--g N`: Filter by GPU count
- `--metric {busbw|algbw}`: Bandwidth metric (default: `busbw`)
- `--out-dir DIR`: Custom output directory
- `--save-csv FILE`: Export per-message-size data to CSV

Run `python3 plot_nccl_bandwidth.py --help` for all options.

### Generate Heatmaps

Visualize network bandwidth with heatmaps from `summary.csv`. By default, only heatmaps are generated; topology graphs require the `--topology` flag. All paths are resolved automatically (symlinks included), so commands work from the repository root.

```bash
# Process all tests and G values (generates heatmaps only by default)
python3 generate_topology.py \
  --csv benchmarks/<cluster-name>/nccl-benchmark-results/pairwise/latest/without-debug/summary.csv \
  --all

# Single test, all G values
python3 generate_topology.py --csv ./summary.csv --test alltoall_perf

# Also generate topology graphs (in addition to heatmaps)
python3 generate_topology.py --csv ./summary.csv --all --topology

# With custom styling
python3 generate_topology.py --csv ./summary.csv --all --topology \
  --vmin 0 --vmax 80 --layout shell --adjust-labels
```

**Output:** `topology/{test_name}/G{n}_heatmap.png` and `allG_heatmap.png` by default; add `--topology` to also generate `G{n}.png` + `allG.png` (combined grid)

**Key Options:**

- `--all`: Process all tests and G values
- `--test NAME`: Process specific test only
- `--topology`: Also generate topology graph PNG(s) in addition to heatmaps
- `--topology-only`: Generate only topology graphs (skip heatmaps)
- `--adjust-labels`: Auto-adjust overlapping labels (useful for dense graphs)
- `--layout`: Algorithm (`kamada`, `shell`, `spring`, `circular`, `bipartite`, `cluster`)
- `--heatmap-values {auto|on|off}`: Show numbers in heatmap cells (default `auto` for ≤20 nodes)
- `--vmin/--vmax`: Bandwidth color scale range (default: 0 to auto-detected max)

### Generate Plot Gallery

Create an HTML or Markdown gallery to browse plots across clusters, test types, and runs. The gallery scans
`benchmarks/<cluster>/...` for images under `plots/` (bandwidth) and `topology/` (heatmaps).

```bash
# HTML gallery (default: benchmarks/plot-gallery.html)
python3 generate_plot_gallery.py

# Filter to specific clusters
python3 generate_plot_gallery.py --clusters cluster01,cluster02 \
  --output benchmarks/plot-gallery.html

# Markdown output
python3 generate_plot_gallery.py --format md --output benchmarks/plot-gallery.md

# Only bandwidth plots (exclude heatmaps)
python3 generate_plot_gallery.py --no-topology --output benchmarks/plot-gallery.html
```

**Key Options:**

- `--clusters LIST`: Comma-separated cluster names to include
- `--format {html|md}`: Output format (default: `html`)
- `--output FILE`: Output path
- `--no-plots`: Exclude bandwidth plots
- `--no-topology`: Exclude topology heatmaps
- `--dpi`: Resolution (default: 300)

Run `python3 generate_topology.py --help` for all options.

## Configuration

Each script has sensible defaults that can be overridden via environment variables or CLI options.

### Default Test Parameters

| Parameter               | Single-Node | Pairwise   | Multi-Node | Smoke    |
| ----------------------- | ----------- | ---------- | ---------- | -------- |
| `MAXIMUM_TRANSFER_SIZE` | 16G         | 16G        | 16G        | 512M     |
| `MINIMUM_TRANSFER_SIZE` | 32M         | 4G         | 32M        | 32M      |
| `STEP_FACTOR`           | 2           | 2          | 2          | 2        |
| `ITERS_COUNT`           | 20          | 20         | 20         | 5        |
| `WARMUP_ITERS`          | 5           | 5          | 5          | 2        |
| `JOB_TIME_LIMIT`        | 00:30:00    | 00:50:00   | 00:50:00   | 00:05:00 |
| GPU counts (`--gpn`)    | 4, 8        | 1, 2, 4, 8 | 1, 2, 4, 8 | 1        |

> **Note**: `JOB_TIME_LIMIT` format is `HH:MM:SS` (hours:minutes:seconds).

### Default Test Binaries

| Script      | Default Binaries                                                                              |
| ----------- | --------------------------------------------------------------------------------------------- |
| Single-Node | `all_reduce_perf`, `all_gather_perf`, `reduce_scatter_perf`, `alltoall_perf`, `sendrecv_perf` |
| Pairwise    | `alltoall_perf`, `sendrecv_perf`                                                              |
| Multi-Node  | `all_reduce_perf`, `all_gather_perf`, `reduce_scatter_perf`, `alltoall_perf`, `sendrecv_perf` |
| Smoke       | `all_reduce_perf`, `sendrecv_perf`                                                            |

### Environment Variable Overrides

Override any default by setting environment variables before running scripts:

```bash
# Example: Custom transfer sizes and iterations
MAXIMUM_TRANSFER_SIZE=8G MINIMUM_TRANSFER_SIZE=1G ITERS_COUNT=50 \
  bash sbatch_run_nccl_tests_pairs.sh -p <partition> -c cluster00

# Example: Custom test binaries
RUN_BIN_LIST="all_reduce_perf alltoall_perf" \
  bash sbatch_run_nccl_tests_single.sh -p <partition> -c cluster00
```

## Limitations

- **Scheduler**: SLURM only
- **GPU/NIC selection**: Manual configuration via environment variables (e.g., `CUDA_VISIBLE_DEVICES`, `NCCL_SOCKET_IFNAME`)
- **Large clusters**: Heatmaps become crowded (>20 nodes)

## Links

- [NVIDIA NCCL Documentation](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/index.html)
- [NCCL GitHub Issues](https://github.com/NVIDIA/nccl/issues)
- [NCCL Tests GitHub](https://github.com/NVIDIA/nccl-tests)
- [[2507.04786] Demystifying NCCL: An In-depth Analysis of GPU Communication Protocols and Algorithms](https://arxiv.org/abs/2507.04786)
- [[2510.20171] Collective Communication for 100k+ GPUs](https://arxiv.org/abs/2510.20171)

## Troubleshooting

> [!TIP]
> If you encounter issues related to NCCL, it is highly recommended to search for or post your questions on [NCCL GitHub Issues](https://github.com/NVIDIA/nccl/issues) and [NCCL Tests GitHub Issues](https://github.com/NVIDIA/nccl-tests/issues).

### Common Issues

**"No .log files found" when running Python scripts:**

This typically means the path doesn't exist or the symlink is broken. Verify the path:

```bash
# Check if the path exists and symlinks are valid
ls -la benchmarks/<cluster-name>/nccl-benchmark-results/pairwise/latest

# If the symlink is broken, it should point to "runs/<run-id>" (relative path)
# Correct symlink example: latest -> runs/20260128-010702
# Broken symlink example: latest -> benchmarks/.../runs/20260128-010702 (full path)

# To fix a broken symlink:
cd benchmarks/<cluster-name>/nccl-benchmark-results/pairwise
rm latest
ls runs/  # Find the correct run ID
ln -sfn runs/<run-id> latest
```

The Python scripts use `os.path.realpath()` to resolve symlinks, so they will work correctly as long as the symlink target exists. You can also use the absolute path directly:

```bash
python3 summarize_nccl_logs.py \
  --input benchmarks/<cluster-name>/nccl-benchmark-results/pairwise/runs/<run-id>/without-debug/logs
```

**Single-node tests succeed but multi-node tests fail:**

Try specifying the network interface used for communication:

```bash
export NCCL_SOCKET_IFNAME=<iface>
```

**Low bandwidth with small transfer sizes:**

If average bus bandwidth is significantly below theoretical limits when using small transfer sizes (e.g., 32 MB), consider increasing `MINIMUM_TRANSFER_SIZE` in scripts (default: 32M). Larger transfer sizes typically achieve higher sustained bandwidth.

**Gray blocks in heatmap (or red lines in topology graphs):**

These indicate failed tests or missing data. Check the corresponding log files for detailed error messages:

```bash
# Check for error files
ls -la benchmarks/<cluster-name>/nccl-benchmark-results/pairwise/latest/without-debug/logs/*.err

# View specific error
cat benchmarks/<cluster-name>/nccl-benchmark-results/pairwise/latest/without-debug/logs/<job>.err
```

**Multiple processes using the same Rank:**

If you see multiple processes using the same Rank in the logs, ensure that you compile NCCL Tests with MPI support enabled:

```bash
# Example log output showing the problem:
# Rank  0 Group  0 Pid 223120 on cnode2-002 device  0 [0000:1b:00] NVIDIA H100 80GB HBM3
# Rank  0 Group  0 Pid 223121 on cnode2-002 device  0 [0000:1b:00] NVIDIA H100 80GB HBM3
# ...

# Fix: Rebuild NCCL Tests with MPI support
module load openmpi
cd nccl/nccl-tests
make clean
make MPI=1
```
