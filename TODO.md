# Todo

## Core

- [ ] feat: suggestions selection based on historical data and available hardware

  - user inputs how many nodes / gpus they want to use
  - automatically detect available hardware (e.g., nodes, GPUs per node)
  - suggest optimal test pairs and configurations

- [ ] feat: multi-node nccl-tests script `sbatch_run_nccl_tests_multi.sh`

  - organize logs by cluster name
  - parse logs to csv/markdown table (no topology visualization)

- [ ] feat: single node nccl-tests script `sbatch_run_nccl_tests_single.sh`

  - organize logs by cluster name
  - parse logs to csv/markdown table (no topology visualization)

## Enhancements

- [x] refactor: benchmarks file structure

  ```bash
  benchmarks/
  {cluster_name}/                    # e.g., cluster01: 8 nodes × 8 H100 GPUs each
    nccl-benchmark-results/
      single-node/                   # Single node test results
        with-debug/
          logs/
          summary.csv
          summary.md
        without-debug/
          (same as above)
      multi-node/                   # Multi-node test results
        (same as above)
      pairwise/                     # Pairwise test results
        with-debug/
          logs/
          topology/
          summary.csv
          summary.md
        without-debug/
          (same as above)
    # ... others documents/scripts of this cluster
  {cluster_name2}/
    ...
  ```

- [x] feat: summarize failed tests or missing data into `failures.txt` using `summarize_nccl_logs.py`

  - list unique failed node pairs

- [ ] feat: better visualization for larger clusters (e.g., N>17)

- [ ] docs: add more usage examples and troubleshooting tips

## Documentation

- [ ] docs: how to evaluate whether the benchmark results are reasonable

  - what should be the expected performance ranges for different interconnects (e.g., NVLink, PCIe, InfiniBand)
  - what to look for in the performance numbers
  - common anomalies and their possible causes
  - next steps if anomalies are detected

## Discussion

- Survey existing tools for cluster topology visualization, performance monitoring, and fault tolerance.
- How can we assess whether these figures are reasonable and consistent with expectations? Next step if we find anomalies?
