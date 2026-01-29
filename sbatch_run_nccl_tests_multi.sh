#!/bin/bash

# We avoid `set -e` here to allow continuing other GPN submissions even if one fails.
set -uo pipefail

# Source common library
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/lib/nccl_common.sh"

# =============================================================
# Parse Arguments
# =============================================================

usage() {
  cat <<'USAGE'
Submit NCCL tests on multi-node groups (N >= 2) with configurable GPUs per node.

Usage:
  sbatch_run_nccl_tests_multi.sh [options]

Options:
  -p, --partition <PART>     Slurm partition name (required).
  -c, --cluster <NAME>       Cluster name for organizing logs. Default: cluster00
  -n, --nodelist "<string>"  Compressed nodelist, e.g. "cnode-[001-004]". If not set, top --num-nodes from the partition are used.
      --num-nodes <N>        Number of nodes to use when --nodelist not provided. Default: 4
  -r, --run-id <ID>          Run ID. Default: YYYYMMDD-HHMMSS
  -l, --log-dir <DIR>        Custom log directory. Default: benchmarks/<CLUSTER>/nccl-benchmark-results/multi-node/runs/<RUN_ID>/without-debug/logs (or with-debug)
      --gpn "<list>"         Space-separated GPUs-per-node list. Default: "1 2 4 8"
      --dry-run              Show commands without executing them
      --debug                Enable NCCL debug mode (may affect performance)
  -h, --help                 Show this help

Environment:
  NCCL_HOME                  Path to NCCL installation (default: $HOME/nccl-tests-cluster/nccl/build)
  NCCL_TEST                  Path to nccl-tests repo (default: $HOME/nccl-tests-cluster/nccl/nccl-tests)
  LD_LIBRARY_PATH            Path to CUDA libraries (default: $HOME/nccl-tests-cluster/nccl/build/lib)

  (optional) NCCL_SOCKET_IFNAME  Specify network interface for NCCL communication.
  (optional) NCCL_IB_HCA         Specify InfiniBand HCA for NCCL
  (optional) RUN_BIN_LIST        Space-separated list of NCCL test binaries to run (default set below)
  (optional) MINIMUM_TRANSFER_SIZE, MAXIMUM_TRANSFER_SIZE, STEP_FACTOR, ITERS_COUNT, WARMUP_ITERS
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -p|--partition)
      [[ $# -ge 2 ]] || { echo "Missing value for $1" >&2; usage; exit 2; }
      PARTITION="$2"; shift 2;;
    -c|--cluster)
      [[ $# -ge 2 ]] || { echo "Missing value for $1" >&2; usage; exit 2; }
      CLUSTER="$2"; shift 2;;
    -n|--nodelist)
      [[ $# -ge 2 ]] || { echo "Missing value for $1" >&2; usage; exit 2; }
      NODELIST="$2"; shift 2;;
    --num-nodes)
      [[ $# -ge 2 ]] || { echo "Missing value for $1" >&2; usage; exit 2; }
      NUM_NODES_CLI="$2"; shift 2;;
    -r|--run-id)
      [[ $# -ge 2 ]] || { echo "Missing value for $1" >&2; usage; exit 2; }
      RUN_ID="$2"; shift 2;;
    -l|--log-dir)
      [[ $# -ge 2 ]] || { echo "Missing value for $1" >&2; usage; exit 2; }
      LOG_DIR="$2"; LOG_DIR_SET=1; shift 2;;
    --gpn)
      [[ $# -ge 2 ]] || { echo "Missing value for $1" >&2; usage; exit 2; }
      GPN_LIST="$2"; shift 2;;
    --dry-run)
      DRY_RUN=1; shift;;
    --debug)
      DEBUG=1; shift;;
    -h|--help)
      usage; exit 0;;
    *)
      echo "Unknown option: $1" >&2
      usage; exit 2;;
  esac
done

# =============================================================
# Configuration
# =============================================================

PARTITION=${PARTITION:-}
CLUSTER=${CLUSTER:-cluster00}
NODELIST=${NODELIST:-}
NUM_NODES_CLI=${NUM_NODES_CLI:-}
GPN_LIST=${GPN_LIST:-"1 2 4 8"}
GPN_LIST=${GPN_LIST//,/ }
CPUS_PER_TASK=${CPUS_PER_TASK:-2}
RUN_ID=${RUN_ID:-$(date +%Y%m%d-%H%M%S)}
RESULTS_ROOT_BASE=${RESULTS_ROOT_BASE:-"benchmarks/$CLUSTER/nccl-benchmark-results"}
RESULTS_ROOT=${RESULTS_ROOT:-"$RESULTS_ROOT_BASE/multi-node"}
RUN_DIR=${RUN_DIR:-"$RESULTS_ROOT/runs/$RUN_ID"}
LOG_DIR_SET=${LOG_DIR_SET:-0}

DEBUG=${DEBUG:-0}
DRY_RUN=${DRY_RUN:-0}

if [[ -z "${PARTITION}" ]]; then
  echo "Error: --partition is required." >&2
  usage
  exit 2
fi

# Multi-node test defaults
MAXIMUM_TRANSFER_SIZE=${MAXIMUM_TRANSFER_SIZE:-16G}
MINIMUM_TRANSFER_SIZE=${MINIMUM_TRANSFER_SIZE:-32M}
STEP_FACTOR=${STEP_FACTOR:-2}
ITERS_COUNT=${ITERS_COUNT:-20}
WARMUP_ITERS=${WARMUP_ITERS:-5}
JOB_TIME_LIMIT=${JOB_TIME_LIMIT:-"00:50:00"}

# Default binaries
DEFAULT_RUN_BIN=(
  all_reduce_perf
  all_gather_perf
  reduce_scatter_perf
  alltoall_perf
  sendrecv_perf
)

if [[ -n "${RUN_BIN_LIST:-}" ]]; then
  IFS=' ' read -r -a RUN_BIN <<< "${RUN_BIN_LIST}"
else
  RUN_BIN=("${DEFAULT_RUN_BIN[@]}")
fi

# =============================================================
# Setup Environment and Directories
# =============================================================

if [[ $DEBUG -eq 1 ]]; then
  LOG_DIR=${LOG_DIR:-"$RUN_DIR/with-debug/logs"}
else
  LOG_DIR=${LOG_DIR:-"$RUN_DIR/without-debug/logs"}
fi

setup_nccl_env "$DEBUG" || exit 1
setup_directories "$RESULTS_ROOT" "$RUN_DIR" "$LOG_DIR" "$DRY_RUN" "$LOG_DIR_SET"

# =============================================================
# Resolve Node List
# =============================================================

declare -a NODES
if [[ -n "${NODELIST}" ]]; then
  mapfile -t NODES < <(expand_nodelist "${NODELIST}")
else
  mapfile -t NODES < <(get_nodes_from_partition "${PARTITION}")
fi

if [[ ${#NODES[@]} -lt 2 ]]; then
  echo "Error: need at least 2 nodes, found ${#NODES[@]}." >&2
  exit 2
fi

NUM_NODES=${NUM_NODES_CLI:-${#NODES[@]}}
if [[ ${NUM_NODES} -gt ${#NODES[@]} ]]; then
  echo "Error: requested --num-nodes=${NUM_NODES} but only ${#NODES[@]} nodes available." >&2
  exit 2
fi

# Trim to requested count
NODES=("${NODES[@]:0:${NUM_NODES}}")
NODELIST_SLURM=$(IFS=,; echo "${NODES[*]}")

# Normalize GPN list into array
read -r -a GPN_ARRAY <<< "${GPN_LIST}"

echo "Submitting multi-node job on ${NUM_NODES} nodes:"
for node in "${NODES[@]}"; do
  echo "  $node"
done

# =============================================================
# Submit Jobs (one job per GPN)
# =============================================================

NUM_SUBMIT=0
NUM_EXIST=0
NUM_FAIL=0

for gpn in "${GPN_ARRAY[@]}"; do
  job="NCCL_N${NUM_NODES}_G${gpn}_${NODES[0]}_plus"
  log_prefix="${LOG_DIR}/nccl_N${NUM_NODES}_G${gpn}"

  if [[ -f "${log_prefix}.log" || -f "${log_prefix}.err" ]]; then
    echo "Skip (log exists): ${job}"
    ((NUM_EXIST++))
    continue
  fi

  if [[ "$DEBUG" -eq 1 && "$DRY_RUN" -eq 0 ]]; then
    export NCCL_DEBUG_FILE="${log_prefix}_debug.log"
  fi

  echo "Submit: ${job}  --nodelist=${NODELIST_SLURM}  --gpus-per-node=${gpn}"

  # Build NCCL test command using common library
  nccl_cmd=$(build_nccl_cmd "$NCCL_TEST" "$ITERS_COUNT" "$WARMUP_ITERS" "$STEP_FACTOR" \
             "$MINIMUM_TRANSFER_SIZE" "$MAXIMUM_TRANSFER_SIZE" "${RUN_BIN[@]}")

  sbatch_args=(
    -J "${job}"
    -p "${PARTITION}"
    -N "${NUM_NODES}"
    --nodelist="${NODELIST_SLURM}"
    --gpus-per-node="${gpn}"
    --ntasks-per-node="${gpn}"
    --cpus-per-task="${CPUS_PER_TASK}"
    -o "${log_prefix}.log"
    -e "${log_prefix}.err"
    --time "${JOB_TIME_LIMIT}"
    --wrap "${nccl_cmd}"
  )

  if ! submit_sbatch_job "$DRY_RUN" "${sbatch_args[@]}"; then
    echo "Submit FAILED: ${job}" >&2
    ((NUM_FAIL++))
    continue
  fi

  ((NUM_SUBMIT++))
done

# =============================================================
# Summary
# =============================================================

if [[ "$DRY_RUN" -eq 1 ]]; then
  echo "DRY RUN complete. Would have submitted ${NUM_SUBMIT} jobs. (${NUM_EXIST} skipped due to existing logs.)"
else
  echo "Submitted ${NUM_SUBMIT} jobs. (${NUM_EXIST} skipped, ${NUM_FAIL} failed submissions.)"
fi

print_submission_summary "Submission Summary" \
  "Nodes" "${NUM_NODES}" \
  "GPN values" "${#GPN_ARRAY[@]}" \
  "Submitted" "${NUM_SUBMIT}" \
  "Skipped" "${NUM_EXIST}" \
  "Failed" "${NUM_FAIL}" \
  "DRY RUN" "${DRY_RUN}" \
  "NCCL DEBUG" "${DEBUG}"
