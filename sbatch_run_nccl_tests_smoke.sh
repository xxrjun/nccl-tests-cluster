#!/bin/bash

set -uo pipefail

# Quick sanity NCCL test across two nodes with minimal payloads.

# Source common library
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/lib/nccl_common.sh"

usage() {
  cat <<'USAGE'
Submit a lightweight smoke test (all_reduce_perf + sendrecv_perf) on two nodes.

Usage:
  sbatch_run_nccl_tests_smoke.sh -p <PARTITION> [options]

Options:
  -p, --partition <PART>     Slurm partition name (required).
  -c, --cluster <NAME>       Cluster name for organizing logs. Default: cluster00
  -n, --nodelist "<string>"  Compressed nodelist; if omitted, first two nodes in the partition are used.
  -r, --run-id <ID>          Run ID. Default: YYYYMMDD-HHMMSS
  -l, --log-dir <DIR>        Custom log directory. Default: benchmarks/<CLUSTER>/nccl-benchmark-results/smoke/runs/<RUN_ID>/logs
      --gpn <N>              GPUs per node. Default: 1
      --dry-run              Show commands without executing them
      --debug                Enable NCCL debug mode (may affect performance)
  -h, --help                 Show this help

Environment:
  NCCL_HOME, NCCL_TEST, LD_LIBRARY_PATH same as other scripts.
  You can override MINIMUM_TRANSFER_SIZE, MAXIMUM_TRANSFER_SIZE, STEP_FACTOR, ITERS_COUNT, WARMUP_ITERS.
USAGE
}

# =============================================================
# Parse Arguments
# =============================================================

while [[ $# -gt 0 ]]; do
  case "$1" in
    -p|--partition)
      [[ $# -ge 2 ]] || { echo "Missing value for $1" >&2; usage; exit 2; }
      PARTITION="$2"; shift 2;;
    -c|--cluster)
      [[ $# -ge 2 ]] || { echo "Missing value for $1" >&2; usage; exit 2; }
      CLUSTER_NAME="$2"; shift 2;;
    -n|--nodelist)
      [[ $# -ge 2 ]] || { echo "Missing value for $1" >&2; usage; exit 2; }
      NODELIST="$2"; shift 2;;
    -r|--run-id)
      [[ $# -ge 2 ]] || { echo "Missing value for $1" >&2; usage; exit 2; }
      RUN_ID="$2"; shift 2;;
    -l|--log-dir)
      [[ $# -ge 2 ]] || { echo "Missing value for $1" >&2; usage; exit 2; }
      LOG_DIR="$2"; LOG_DIR_SET=1; shift 2;;
    --gpn)
      [[ $# -ge 2 ]] || { echo "Missing value for $1" >&2; usage; exit 2; }
      GPN="$2"; shift 2;;
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
CLUSTER_NAME=${CLUSTER_NAME:-cluster00}
NODELIST=${NODELIST:-}
GPN=${GPN:-1}
CPUS_PER_TASK=${CPUS_PER_TASK:-2}
RUN_ID=${RUN_ID:-$(date +%Y%m%d-%H%M%S)}
RESULTS_ROOT_BASE=${RESULTS_ROOT_BASE:-"benchmarks/$CLUSTER_NAME/nccl-benchmark-results"}
RESULTS_ROOT=${RESULTS_ROOT:-"$RESULTS_ROOT_BASE/smoke"}
RUN_DIR=${RUN_DIR:-"$RESULTS_ROOT/runs/$RUN_ID"}
LOG_DIR_SET=${LOG_DIR_SET:-0}
DEBUG=${DEBUG:-0}
DRY_RUN=${DRY_RUN:-0}

if [[ -z "${PARTITION}" ]]; then
  echo "Error: --partition is required." >&2
  usage
  exit 2
fi

# Smoke test defaults (faster than full tests)
MAXIMUM_TRANSFER_SIZE=${MAXIMUM_TRANSFER_SIZE:-512M}
MINIMUM_TRANSFER_SIZE=${MINIMUM_TRANSFER_SIZE:-32M}
STEP_FACTOR=${STEP_FACTOR:-2}
ITERS_COUNT=${ITERS_COUNT:-5}
WARMUP_ITERS=${WARMUP_ITERS:-2}
JOB_TIME_LIMIT=${JOB_TIME_LIMIT:-"00:05:00"}

# Smoke test binaries
RUN_BIN=(all_reduce_perf sendrecv_perf)

# =============================================================
# Setup Environment and Directories
# =============================================================

LOG_DIR=${LOG_DIR:-"$RUN_DIR/logs"}

setup_nccl_env "$DEBUG" || exit 1
setup_directories "$RESULTS_ROOT" "$RUN_DIR" "$LOG_DIR" "$DRY_RUN" "$LOG_DIR_SET"

# =============================================================
# Resolve Node List
# =============================================================

declare -a NODES
if [[ -n "${NODELIST}" ]]; then
  mapfile -t NODES < <(expand_nodelist "${NODELIST}")
else
  mapfile -t NODES < <(get_nodes_from_partition "${PARTITION}" | head -n 2)
fi

if [[ ${#NODES[@]} -lt 2 ]]; then
  echo "Error: need at least 2 nodes, found ${#NODES[@]}." >&2
  exit 2
fi
NODES=("${NODES[@]:0:2}")
NODELIST_SLURM=$(IFS=,; echo "${NODES[*]}")

echo "Submitting smoke test on nodes: ${NODES[*]} (GPN=${GPN})"

# =============================================================
# Submit Job
# =============================================================

job="NCCL_SMOKE_N2_G${GPN}_${NODES[0]}_${NODES[1]}"
log_prefix="${LOG_DIR}/nccl_smoke_N2_G${GPN}_${NODES[0]}_${NODES[1]}"

if [[ -f "${log_prefix}.log" || -f "${log_prefix}.err" ]]; then
  echo "Skip (log exists): ${job}"
  exit 0
fi

if [[ "$DEBUG" -eq 1 && "$DRY_RUN" -eq 0 ]]; then
  export NCCL_DEBUG_FILE="${log_prefix}_debug.log"
fi

# Build NCCL test command using common library
nccl_cmd=$(build_nccl_cmd "$NCCL_TEST" "$ITERS_COUNT" "$WARMUP_ITERS" "$STEP_FACTOR" \
           "$MINIMUM_TRANSFER_SIZE" "$MAXIMUM_TRANSFER_SIZE" "${RUN_BIN[@]}")

sbatch_args=(
  -J "${job}"
  -p "${PARTITION}"
  -N 2
  --nodelist="${NODELIST_SLURM}"
  --gpus-per-node="${GPN}"
  --ntasks-per-node="${GPN}"
  --cpus-per-task="${CPUS_PER_TASK}"
  -o "${log_prefix}.log"
  -e "${log_prefix}.err"
  --time "${JOB_TIME_LIMIT}"
  --wrap "${nccl_cmd}"
)

submit_sbatch_job "$DRY_RUN" "${sbatch_args[@]}"

echo "Submitted smoke test job: ${job}"
