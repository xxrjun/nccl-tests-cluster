#!/bin/bash
# ===========================================================
# NCCL Tests Common Library
# Shared functions and defaults for NCCL test scripts
# ===========================================================

# ===========================================================
# Default NCCL Environment
# ===========================================================

# Setup NCCL environment variables and verify paths
# Args:
#   $1 - debug mode (0=off, 1=on). Optional, defaults to 0.
# Returns:
#   0 on success, 1 if NCCL_TEST path not found
setup_nccl_env() {
  local debug=${1:-0}

  export NCCL_HOME="${NCCL_HOME:-$HOME/nccl-tests-cluster/nccl/build}"
  export NCCL_TEST="${NCCL_TEST:-$HOME/nccl-tests-cluster/nccl/nccl-tests}"
  export LD_LIBRARY_PATH="$NCCL_HOME/lib:${LD_LIBRARY_PATH:-}"

  if [[ ! -d "${NCCL_TEST}/build" ]]; then
    echo "Error: NCCL_TEST path not found: ${NCCL_TEST}/build" >&2
    return 1
  fi

  if [[ "$debug" -eq 1 ]]; then
    echo "NCCL DEBUG: Enabled"
    export NCCL_DEBUG=INFO
    export NCCL_DEBUG_SUBSYS=ALL,^CALL,^PROXY
  else
    echo "NCCL DEBUG: Disabled"
  fi

  return 0
}

# ===========================================================
# Directory Management
# ===========================================================

# Create directories and set up latest symlink
# Args:
#   $1 - results_root: Base directory for results (e.g., benchmarks/cluster/pairwise)
#   $2 - run_dir: Full path to this run's directory
#   $3 - log_dir: Directory for log files
#   $4 - dry_run: Skip creation if 1. Optional, defaults to 0.
#   $5 - log_dir_set: If 1, skip latest symlink update. Optional, defaults to 0.
setup_directories() {
  local results_root="$1"
  local run_dir="$2"
  local log_dir="$3"
  local dry_run="${4:-0}"
  local log_dir_set="${5:-0}"

  if [[ "$dry_run" -eq 1 ]]; then
    return 0
  fi

  mkdir -p "${log_dir}"

  if [[ "$log_dir_set" -eq 0 ]]; then
    mkdir -p "${results_root}/runs"
    ln -sfn "$run_dir" "$results_root/latest"
  fi
}

# ===========================================================
# Node List Helpers
# ===========================================================

# Get node list from a SLURM partition
# Args:
#   $1 - partition: SLURM partition name
#   $2 - state: Optional filter (e.g., "idle"). If empty, returns all nodes.
# Output: One node name per line, sorted
get_nodes_from_partition() {
  local partition="$1"
  local state="${2:-}"  # Optional: "idle" to filter only idle nodes

  if [[ -n "$state" ]]; then
    sinfo -p "${partition}" -t "$state" -h -N -o %N | sort -V
  else
    sinfo -p "${partition}" -h -N -o %N | sort -V
  fi
}

# Expand compressed SLURM nodelist to individual node names
# Args:
#   $1 - nodelist: Compressed nodelist (e.g., "node[01-03]")
# Output: One node name per line
expand_nodelist() {
  local nodelist="$1"
  scontrol show hostnames "$nodelist"
}

# ===========================================================
# NCCL Test Command Builder
# ===========================================================

# Build NCCL test command string for srun execution
# Args:
#   $1 - nccl_test_path: Path to nccl-tests directory
#   $2 - iters: Number of iterations
#   $3 - warmup: Number of warmup iterations
#   $4 - step_factor: Step factor for message sizes
#   $5 - min_bytes: Minimum transfer size
#   $6 - max_bytes: Maximum transfer size
#   $7+ - bins: Test binary names (e.g., all_reduce_perf)
# Output: Command string ready for sbatch --wrap
build_nccl_cmd() {
  local nccl_test_path="$1"
  local iters="$2"
  local warmup="$3"
  local step_factor="$4"
  local min_bytes="$5"
  local max_bytes="$6"
  shift 6
  local bins=("$@")

  local cmd=""
  for bin in "${bins[@]}"; do
    cmd+="srun ${nccl_test_path}/build/${bin}"
    cmd+=" --iters ${iters}"
    cmd+=" --warmup_iters ${warmup}"
    cmd+=" -f ${step_factor}"
    cmd+=" --datatype double"
    cmd+=" --minbytes ${min_bytes}"
    cmd+=" --maxbytes ${max_bytes}"
    cmd+="; "
  done
  # Remove trailing "; "
  echo "${cmd%"; "}"
}

# ===========================================================
# Job Submission
# ===========================================================

# Submit a job via sbatch, with dry-run support
# Args:
#   $1 - dry_run: If 1, run sbatch --test-only instead of actual submission
#   $2+ - sbatch_args: All arguments to pass to sbatch
# Returns: sbatch exit code, or 0 for dry run
submit_sbatch_job() {
  local dry_run="$1"
  shift
  local sbatch_args=("$@")

  if [[ "$dry_run" -eq 1 ]]; then
    echo "=========================================="
    echo "DRY RUN: sbatch --test-only preview (no submission):"
    printf 'sbatch --test-only %q ' "${sbatch_args[@]}"
    echo
    sbatch --test-only "${sbatch_args[@]}" || true
    echo "=========================================="
    echo ""
    return 0
  else
    sbatch "${sbatch_args[@]}"
    return $?
  fi
}

# ===========================================================
# Argument Parsing Helpers
# ===========================================================

# Check if an option has a required value
# Args:
#   $1 - opt: Option name (for error message)
#   $2 - num_remaining: Number of remaining arguments
# Returns: 0 if value exists, 1 otherwise
require_value() {
  local opt="$1"
  local num_remaining="$2"
  if [[ "$num_remaining" -lt 2 ]]; then
    echo "Missing value for $opt" >&2
    return 1
  fi
  return 0
}

# ===========================================================
# Summary Helpers
# ===========================================================

# Print a formatted summary box with key-value pairs
# Args:
#   $1 - title: Header text for the summary box
#   $2+ - pairs: Alternating key/value pairs
# Example: print_submission_summary "Job Info" "Partition" "gpu" "Nodes" "4"
print_submission_summary() {
  local title="$1"
  shift
  # Accept key-value pairs
  local -a pairs=("$@")

  echo "=========================================="
  echo "$title"
  echo "=========================================="
  local i=0
  while [[ $i -lt ${#pairs[@]} ]]; do
    printf "%-20s %s\n" "${pairs[$i]}:" "${pairs[$((i+1))]}"
    ((i+=2))
  done
  echo "=========================================="
}
