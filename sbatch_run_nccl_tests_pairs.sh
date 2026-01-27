#!/bin/bash
set -uo pipefail

# Source common library
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/lib/nccl_common.sh"

# =============================================================
# Parse Arguments
# ============================================================

usage() {
  cat <<'USAGE'
Enumerate all node pairs (or a user-specified compressed nodelist)
and submit NCCL tests. Default GPUs per node: {1,2,4,8}.

Usage:
  sbatch_run_nccl_tests_pairs.sh [options]

Options:
  -p, --partition <PART>     Slurm partition name.
  -c, --cluster <NAME>       Cluster name for organizing logs. Default: cluster00
  -n, --nodelist "<string>"  Compressed nodelist to limit pairs, e.g. "cnode-[009,011-013]". If not set, all nodes in the partition are used.
  -r, --run-id <ID>          Run ID for timestamped results. Default: YYYYMMDD-HHMMSS
  -l, --log-dir <DIR>        Directory for logs. Default: benchmarks/<CLUSTER>/nccl-benchmark-results/pairwise/runs/<RUN_ID>/without-debug/logs (or with-debug)
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

USAGE
}

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
CLUSTER_NAME=${CLUSTER_NAME:-cluster00}
NODELIST=${NODELIST:-}
GPN_LIST=${GPN_LIST:-"1 2 4 8"}
GPN_LIST=${GPN_LIST//,/ }
CPUS_PER_TASK=${CPUS_PER_TASK:-2}
RUN_ID=${RUN_ID:-$(date +%Y%m%d-%H%M%S)}
RESULTS_ROOT_BASE=${RESULTS_ROOT_BASE:-"benchmarks/$CLUSTER_NAME/nccl-benchmark-results"}
RESULTS_ROOT=${RESULTS_ROOT:-"$RESULTS_ROOT_BASE/pairwise"}
RUN_DIR=${RUN_DIR:-"$RESULTS_ROOT/runs/$RUN_ID"}
LOG_DIR_SET=${LOG_DIR_SET:-0}

DEBUG=${DEBUG:-0}
DRY_RUN=${DRY_RUN:-0}

if [[ -z "${PARTITION}" ]]; then
  echo "Error: --partition is required." >&2
  usage
  exit 2
fi

# Pairwise test defaults
MAXIMUM_TRANSFER_SIZE=${MAXIMUM_TRANSFER_SIZE:-32G}
MINIMUM_TRANSFER_SIZE=${MINIMUM_TRANSFER_SIZE:-4G}
STEP_FACTOR=${STEP_FACTOR:-2}
ITERS_COUNT=${ITERS_COUNT:-20}
WARMUP_ITERS=${WARMUP_ITERS:-5}
JOB_TIME_LIMIT=${JOB_TIME_LIMIT:-"00:50:00"}

# Default binaries - kept minimal for pairwise to reduce test time
DEFAULT_RUN_BIN=(
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
# Generate Node Pairs
# =============================================================

declare -a NODES
if [[ -n "${NODELIST}" ]]; then
  mapfile -t NODES < <(expand_nodelist "${NODELIST}")
else
  mapfile -t NODES < <(get_nodes_from_partition "${PARTITION}")
fi

# Generate unique unordered pairs (i<j)
pairs=()
for ((i=0; i<${#NODES[@]}-1; i++)); do
  for ((j=i+1; j<${#NODES[@]}; j++)); do
    pairs+=("${NODES[$i]},${NODES[$j]}")
  done
done

echo "Submitting ${#pairs[@]} pairs..."
for pair in "${pairs[@]}"; do
  echo "  $pair"
done

# Count GPN items
num_gpn=0
for _g in ${GPN_LIST}; do ((num_gpn++)); done

NUM_TOTAL=$(( ${#pairs[@]} * num_gpn ))
NUM_SUBMIT=0
NUM_EXIST=0

# =============================================================
# Submit Jobs
# =============================================================

for pair in "${pairs[@]}"; do
  A=${pair%,*}
  B=${pair#*,}

  for gpn in ${GPN_LIST}; do
    job="NCCL_N2_G${gpn}_${A}_${B}"
    log="${LOG_DIR}/nccl_N2_G${gpn}_${A}_${B}"

    if [[ -f "${log}.log" || -f "${log}.err" ]]; then
      echo "Skip (log exists): ${job}"
      ((NUM_EXIST++))
      continue
    fi

    if [[ "$DEBUG" -eq 1 && "$DRY_RUN" -eq 0 ]]; then
      export NCCL_DEBUG_FILE="${log}_debug.log"
    fi

    echo "Submit: ${job}  --nodelist=${A},${B}  --gpus-per-node=${gpn}"

    # Build NCCL test command using common library
    nccl_cmd=$(build_nccl_cmd "$NCCL_TEST" "$ITERS_COUNT" "$WARMUP_ITERS" "$STEP_FACTOR" \
               "$MINIMUM_TRANSFER_SIZE" "$MAXIMUM_TRANSFER_SIZE" "${RUN_BIN[@]}")

    sbatch_args=(
      -J "${job}"
      -p "${PARTITION}"
      -N 2
      --nodelist="${A},${B}"
      --gpus-per-node="${gpn}"
      --ntasks-per-node="${gpn}"
      --cpus-per-task="${CPUS_PER_TASK}"
      -o "${log}.log"
      -e "${log}.err"
      --time "${JOB_TIME_LIMIT}"
      --wrap "${nccl_cmd}"
    )

    submit_sbatch_job "$DRY_RUN" "${sbatch_args[@]}"
    ((NUM_SUBMIT++))
  done
done

# =============================================================
# Summary
# =============================================================

if [[ "$DRY_RUN" -eq 1 ]]; then
  echo "DRY RUN complete. Would have submitted ${NUM_SUBMIT} jobs. (${NUM_EXIST} skipped due to existing logs.)"
else
  echo "Submitted ${NUM_SUBMIT} jobs. (${NUM_EXIST} skipped due to existing logs.)"
fi

print_submission_summary "Submission Summary" \
  "Total pairs" "${#pairs[@]}" \
  "Jobs per pair" "${num_gpn}" \
  "Total jobs" "${NUM_TOTAL}" \
  "Submitted" "${NUM_SUBMIT}" \
  "Skipped" "${NUM_EXIST}" \
  "DRY RUN" "${DRY_RUN}" \
  "NCCL DEBUG" "${DEBUG}"
printf "%-15s %s\n" "DRY RUN:"        "${DRY_RUN}"
printf "%-15s %s\n" "NCCL DEBUG:"     "${DEBUG}"
echo "=========================================="
