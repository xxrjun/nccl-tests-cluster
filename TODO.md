# Todo

## Core

- [ ] feat: suggestions selection based on historical data and available hardware

  - user inputs how many nodes / gpus they want to use
  - automatically detect available hardware (e.g., nodes, GPUs per node)
  - suggest optimal test pairs and configurations

- [x] feat: multi-node nccl-tests script `sbatch_run_nccl_tests_multi.sh`

  - organize logs by cluster name
  - parse logs to csv/markdown table (no topology visualization)

- [x] feat: single node nccl-tests script `sbatch_run_nccl_tests_single.sh`

  - [x] organize logs by cluster name
  - [x] parse logs to csv/markdown table
  - [x] plots (via `plot_nccl_bandwidth.py`)
    - x-axis: message size
    - y-axis: bandwidth
    - each node has single plot, also combined plot for all nodes with different lines

- [x] feat: plotting across multiple messages sizes

  - implemented in `plot_nccl_bandwidth.py`
  - supports both single-node and pairwise test results

## Enhancements

- [x] Design a better file structure for benchmarks results

  - timestamped folders for different runs with resume capability, or better version control.
  - At the moment, the archive folder is managed manually, which is not very convenient.

- [x] refactor: benchmarks file structure

  ```bash
  benchmarks/
  {cluster_name}/
    nccl-benchmark-results/
      single-node/
        runs/<RUN_ID>/{with-debug,without-debug}/logs
        latest -> runs/<RUN_ID>
      pairwise/
        runs/<RUN_ID>/{with-debug,without-debug}/logs
        latest -> runs/<RUN_ID>
      multi-node/
        runs/<RUN_ID>/{with-debug,without-debug}/logs
        latest -> runs/<RUN_ID>
      smoke/
        runs/<RUN_ID>/logs
        latest -> runs/<RUN_ID>
  ```

- [x] feat: summarize failed tests or missing data into `failures.txt` using `summarize_nccl_logs.py`

  - list unique failed node pairs

- [ ] feat: better visualization for larger clusters (e.g., N>17)

  - [x] heatmap
  - [ ] interactive 3D visualization

- [ ] docs: add more usage examples and troubleshooting tips

- [ ] feat: add torch nccl benchmark script

  - similar to nccl-tests but using PyTorch's distributed package, which is easier to setup.

- [ ] feat: specify NICs to use for tests (e.g., for nodes with multiple NICs)

- [x] feat: slurm timeout control

- [x] feat: add number in the heatmap cells to indicate bandwidth values

- [x] refactor: extract shared shell helpers for sbatch scripts to reduce duplication

  - created `lib/nccl_common.sh` with shared functions for NCCL environment setup,
    directory management, command building, and job submission
  - integrated into all sbatch scripts: `sbatch_run_nccl_tests_{single,pairs,multi,smoke}.sh`

## Documentation

- [ ] docs: how to evaluate whether the benchmark results are reasonable

  - what should be the expected performance ranges for different interconnects (e.g., NVLink, PCIe, InfiniBand)
  - what to look for in the performance numbers
  - common anomalies and their possible causes
  - next steps if anomalies are detected

- [ ] docs: add related sources

  - [Networking Benchmarks | Stas00](https://github.com/stas00/ml-engineering/tree/master/network/benchmarks)
  - [coreweave/nccl-tests](https://github.com/coreweave/nccl-tests) - NVIDIA NCCL Tests for Distributed Training

## Discussion

- Survey existing tools for cluster topology visualization, performance monitoring, and fault tolerance.
- How can we assess whether these figures are reasonable and consistent with expectations? Next step if we find anomalies?
