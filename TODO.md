# Todo

- [ ] feat: suggestions selection based on historical data and available hardware
  - user inputs how many nodes / gpus they want to use
  - automatically detect available hardware (e.g., nodes, GPUs per node)
  - suggest optimal test pairs and configurations
- [x] feat: summarize failed tests or missing data into `failures.txt` using `summarize_nccl_logs.py`
  - list unique failed node pairs
- [ ] feat: better visualization for larger clusters (e.g., N>17)
- [ ] docs: add more usage examples and troubleshooting tips
