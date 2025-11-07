#!/bin/bash

# =============================================================
# Parse Arguments
# ============================================================

usage() {
  cat <<'USAGE'
Enumerate all idle-node pairs (or a user-specified compressed nodelist)
and submit NCCL tests for GPUs per node in {1,2,3,4,5,6,7,8} by default.

Usage:
  sbatch_run_nccl_tests_pairs.sh [options]

Options:
  -p, --partition <PART>     Slurm partition name.
  -n, --nodelist "<string>"  Compressed nodelist to limit pairs, e.g. "cnode-[009,011-013]". If not set, all nodes in the partition are used.
  -l, --log-dir <DIR>        Directory for logs. Default: benchmarks/cluster01/nccl-tests-pairs/without-debug/logs
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
    -n|--nodelist)
      [[ $# -ge 2 ]] || { echo "Missing value for $1" >&2; usage; exit 2; }
      NODELIST="$2"; shift 2;;
    -l|--log-dir)
      [[ $# -ge 2 ]] || { echo "Missing value for $1" >&2; usage; exit 2; }
      LOG_DIR="$2"; shift 2;;
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
# SLURM and Test Settings
# =============================================================

PARTITION=${PARTITION:-}
NODELIST=${NODELIST:-}
GPN_LIST=${GPN_LIST:-"1 2 4 8"}
CPUS_PER_TASK=${CPUS_PER_TASK:-2}

DEBUG=${DEBUG:-0} # WARN: may affect performance results
DRY_RUN=${DRY_RUN:-0}

CLUSTER_NAME=${CLUSTER_NAME:-}

if [[ $DEBUG -eq 1 ]]; then
  LOG_DIR=${LOG_DIR:-"benchmarks/$CLUSTER_NAME/nccl-tests-pairs/with-debug/logs"}
else
  LOG_DIR=${LOG_DIR:-"benchmarks/$CLUSTER_NAME/nccl-tests-pairs/without-debug/logs"}
fi
mkdir -p "${LOG_DIR}"

# =============================================================
# NCCL Tests Settings
# nccl env vars docs: https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/env.html
# =============================================================
# export NCCL_SOCKET_IFNAME= # specifies which IP interfaces to use for communication.
# export NCCL_IB_HCA=

# Optional: set your nccl / nccl-tests path via env before running
export NCCL_HOME="${NCCL_HOME:-$HOME/nccl-tests-cluster/nccl/build}"
export NCCL_TEST="${NCCL_TEST:-$HOME/nccl-tests-cluster/nccl/nccl-tests}"
export LD_LIBRARY_PATH=$NCCL_HOME/lib:$LD_LIBRARY_PATH

MAXIMUM_TRANSFER_SIZE=16G
MINIMUM_TRANSFER_SIZE=32M
STEP_FACTOR=2
ITERS_COUNT=20
WARMUP_ITERS=5

RUN_BIN=(
    alltoall_perf
    sendrecv_perf

    # all_reduce_perf
    # all_gather_perf
    # reduce_scatter_perf
    # broadcast_perf
    # reduce_perf
)


if [[ ! -d "${NCCL_TEST}/build" ]]; then
    echo "Error: NCCL_TEST path not found: ${NCCL_TEST}/build" >&2
    exit 1
fi

if [[ "$DEBUG" -eq 1 ]]; then
  echo "NCCL DEBUG: Enabled"

  # WARN: NCCL debug env vars may affect performance results
  export NCCL_DEBUG=INFO 
  export NCCL_DEBUG_SUBSYS=ALL,^CALL,^PROXY
 
  # question about adaptive routing: https://github.com/NVIDIA/nccl/issues/1687
  # export NCCL_SOCKET_IFNAME= # specifies which IP interfaces to use for communication.
  # export NCCL_IB_ADAPTIVE_ROUTING=1 # default 1 on IB networks, 0 on RoCE
else
  echo "NCCL DEBUG: Disabled"
fi

# =============================================================
# Generate Node Pairs
# =============================================================

# Resolve node list
declare -a NODES
if [[ -n "${NODELIST}" ]]; then
  # Accept compressed format like "cnode-[009,011-013]"
  mapfile -t NODES < <(scontrol show hostnames "${NODELIST}")
else
  # All nodes in the partition (uncomment the -t idle variant to restrict to idle)
  # mapfile -t NODES < <(sinfo -p "${PARTITION}" -t idle -h -N -o %N | sort -V)
  mapfile -t NODES < <(sinfo -p "${PARTITION}" -h -N -o %N | sort -V)
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

    if [[ "$DEBUG" -eq 1 ]]; then
      export NCCL_DEBUG_FILE="${log}_debug.log"
    fi

    echo "Submit: ${job}  --nodelist=${A},${B}  --gpus-per-node=${gpn}"


    # Build NCCL test commands
    nccl_cmd=""
    for bin in "${RUN_BIN[@]}"; do
      # nccl_cmd+="srun --mpi=pmix --nodes=2 --ntasks-per-node=${gpn} --ntasks $(( gpn * 2 )) ${NCCL_TEST}/build/${bin}"
      nccl_cmd+="srun ${NCCL_TEST}/build/${bin}"
      nccl_cmd+=" --iters ${ITERS_COUNT}"
      nccl_cmd+=" --warmup_iters ${WARMUP_ITERS}"
      nccl_cmd+=" -f ${STEP_FACTOR}"
      nccl_cmd+=" --datatype double"
      nccl_cmd+=" --minbytes ${MINIMUM_TRANSFER_SIZE}"
      nccl_cmd+=" --maxbytes ${MAXIMUM_TRANSFER_SIZE}"
      nccl_cmd+="; "
    done

    if [[ "$DRY_RUN" -eq 1 ]]; then
      echo "=========================================="
      echo "DRY RUN: Would execute the following sbatch command:"
      echo ""
      echo "sbatch \\"
      echo "  -J \"${job}\" \\"
      echo "  -p \"${PARTITION}\" \\"
      echo "  -N 2 \\"
      echo "  --nodelist=\"${A},${B}\" \\"
      echo "  --gpus-per-node=\"${gpn}\" \\"
      echo "  --ntasks-per-node=\"${gpn}\" \\"
      echo "  -o \"${log}.log\" \\"
      echo "  -e \"${log}.err\" \\"
      echo "  --wrap=\\"
      echo ""
      for bin in "${RUN_BIN[@]}"; do
        echo "srun ${NCCL_TEST}/build/${bin} \\"
        echo "  --iters ${ITERS_COUNT} \\"
        echo "  --warmup_iters ${WARMUP_ITERS} \\"
        echo "  -f ${STEP_FACTOR} \\"
        echo "  --datatype double \\"
        echo "  --minbytes ${MINIMUM_TRANSFER_SIZE} \\"
        echo "  --maxbytes ${MAXIMUM_TRANSFER_SIZE}"
        echo ""
      done
      echo "=========================================="
      echo ""
    else
      sbatch \
        -J "${job}" \
        -p "${PARTITION}" \
        -N 2 \
        --nodelist="${A},${B}" \
        --gpus-per-node="${gpn}" \
        --ntasks-per-node="${gpn}" \
        --cpus-per-task="${CPUS_PER_TASK:-2}" \
        -o "${log}.log" \
        -e "${log}.err" \
        --wrap="${nccl_cmd}"
    fi

    ((NUM_SUBMIT++))
  done
done

if [[ "$DRY_RUN" -eq 1 ]]; then
  echo "DRY RUN complete. Would have submitted ${NUM_SUBMIT} jobs. (${NUM_EXIST} skipped due to existing logs.)"
else
  echo "Submitted ${NUM_SUBMIT} jobs. (${NUM_EXIST} skipped due to existing logs.)"
fi
echo "Total pairs: ${#pairs[@]}. Total jobs: ${NUM_TOTAL}."


echo "=========================================="
echo "Submission Summary"
echo "=========================================="
printf "%-15s %s\n" "Total pairs:"    "${#pairs[@]}"
printf "%-15s %s\n" "Jobs per pair:"  "${num_gpn}"
printf "%-15s %s\n" "Total jobs:"     "${NUM_TOTAL}"
printf "%-15s %s\n" "Submitted:"      "${NUM_SUBMIT}"
printf "%-15s %s\n" "Skipped:"        "${NUM_EXIST}"
printf "%-15s %s\n" "DRY RUN:"        "${DRY_RUN}"
printf "%-15s %s\n" "NCCL DEBUG:"     "${DEBUG}"
echo "=========================================="