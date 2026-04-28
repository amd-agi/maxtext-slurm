# Metric Reference

## Metric Reference

Complete catalog of metrics available in the TSDB. All metrics have a `host` label.

### GPU metrics (`hw_gpu_*`) — from `gpu_metrics_plugin.sh`

| Metric | Labels | Type | Description |
|--------|--------|------|-------------|
| `hw_gpu_temperature_celsius` | `gpu`, `host` | gauge | Junction temperature |
| `hw_gpu_power_watts` | `gpu`, `host` | gauge | Power draw |
| `hw_gpu_clock_mhz` | `gpu`, `host`, `type` | gauge | Clock speed (`sclk`=core, `mclk`=memory) |
| `hw_gpu_vram_used_bytes` | `gpu`, `host` | gauge | VRAM used |
| `hw_gpu_vram_total_bytes` | `gpu`, `host` | gauge | VRAM total |
| `hw_gpu_ras_umc_{ue,ce}_total` | `gpu`, `host` | counter | HBM ECC errors |
| `hw_gpu_ras_xgmi_{ue,ce}_total` | `gpu`, `host` | counter | XGMI/WAFL link errors |
| `hw_gpu_ras_gfx_{ue,ce}_total` | `gpu`, `host` | counter | Compute engine errors |
| `hw_gpu_ras_mmhub_{ue,ce}_total` | `gpu`, `host` | counter | Memory hub errors |
| `hw_gpu_ras_sdma_{ue,ce}_total` | `gpu`, `host` | counter | SDMA engine errors |
| `hw_gpu_pcie_correctable_total` | `gpu`, `host` | counter | PCIe correctable errors |
| `hw_gpu_pcie_nonfatal_total` | `gpu`, `host` | counter | PCIe non-fatal errors |
| `hw_gpu_pcie_fatal_total` | `gpu`, `host` | counter | PCIe fatal errors |

### Host/network metrics (`hw_*`) — from `host_metrics_plugin.sh`

| Metric | Labels | Type | Description |
|--------|--------|------|-------------|
| `hw_net_{rx,tx}_bytes_total` | `device`, `host` | counter | Network bytes |
| `hw_net_{rx,tx}_errors_total` | `device`, `host` | counter | NIC errors |
| `hw_net_{rx,tx}_drop_total` | `device`, `host` | counter | NIC drops |
| `hw_tcp_retransmits_total` | `host` | counter | TCP retransmits |
| `hw_tcp_listen_overflows_total` | `host` | counter | Listen queue overflows |
| `hw_tcp_listen_drops_total` | `host` | counter | Listen queue drops |
| `hw_tcp_estab_resets_total` | `host` | counter | Established conn resets |
| `hw_tcp_abort_on_timeout_total` | `host` | counter | Conns aborted after timeout |
| `hw_rdma_{rx,tx}_bytes_total` | `device`, `port`, `host` | counter | RDMA bytes |
| `hw_rdma_{rx,tx}_pkts_total` | `device`, `port`, `host` | counter | RDMA packets |
| `hw_rdma_tx_retx_{bytes,pkts}_total` | `device`, `port`, `host` | counter | RDMA retransmits |
| `hw_rdma_tx_ack_timeout_total` | `device`, `port`, `host` | counter | RDMA ACK timeouts |
| `hw_rdma_{rx,tx}_cnp_pkts_total` | `device`, `port`, `host` | counter | Congestion notifications |
| `hw_rdma_req_rx_cqe_err_total` | `device`, `port`, `host` | counter | CQE errors |
| `hw_rdma_req_tx_retry_excd_err_total` | `device`, `port`, `host` | counter | Retry exhaustion |
| `hw_rdma_port_state` | `device`, `port`, `host` | gauge | 1=ACTIVE, 0=not |
| `hw_procs_running` | `host` | gauge | Runnable processes |
| `hw_procs_blocked` | `host` | gauge | Blocked on I/O |
| `hw_context_switches_total` | `host` | counter | Context switches |
| `hw_oom_kills_total` | `host` | counter | OOM killer invocations |
| `hw_mem_dirty_bytes` | `host` | gauge | Dirty pages |
| `hw_mem_writeback_bytes` | `host` | gauge | Writeback pages |
| `hw_io_pressure_some_pct` | `host` | gauge | I/O pressure (some, 10s) |
| `hw_io_pressure_full_pct` | `host` | gauge | I/O pressure (full, 10s) |
| `hw_io_pressure_{some,full}_avg300_pct` | `host` | gauge | I/O pressure (300s avg) |
| `hw_io_pressure_full_total_us` | `host` | counter | Cumulative I/O stall time |
| `hw_dmesg_gpu_errors_total` | `host` | counter | GPU/driver errors in dmesg |
| `hw_gpu_user_processes` | `host` | gauge | Processes with /dev/kfd open |
| `hw_scrape_duration_seconds` | `host` | gauge | Plugin scrape time (emitted by `metrics_exporter.sh`). **Key health indicator:** normal is ~0.1s (plugins fast) or ~12s (GPU plugin timing out). A flatline means the collection loop is stuck and all `hw_*` data on that host is ghost data |

### Ray metrics (`ray_node_*`) — from Ray built-in exporter

| Metric | Labels | Type | Description |
|--------|--------|------|-------------|
| `ray_node_gpus_utilization` | `host` (or `instance`) | gauge | Per-GPU utilization (%) |
| `ray_node_mem_used` | `host` | gauge | Host memory used (bytes) |
| `ray_node_mem_total` | `host` | gauge | Host memory total (bytes) |
| `ray_node_cpu_utilization` | `host` | gauge | CPU utilization (%) |
| `ray_node_disk_*` | `host` | various | Disk I/O metrics |

### Training metrics (`tb_*`) — from `tb_metrics_plugin.sh`

**Caveat:** These metrics are bridged from TensorBoard event files by a best-effort plugin. Gaps are possible (see "Recovering from `tb_*` gaps"). The raw event file at `<job_dir>/tensorboard/events.out.tfevents.*` is the authoritative source for training scalars.

| Metric | Labels | Type | Description |
|--------|--------|------|-------------|
| `tb_step` | `host` | gauge | Current training step |
| `tb_learning_loss` | `host` | gauge | Training loss |
| `tb_learning_grad_norm` | `host` | gauge | Gradient norm (post-clipping) |
| `tb_learning_raw_grad_norm` | `host` | gauge | Raw gradient norm (pre-clipping) |
| `tb_learning_param_norm` | `host` | gauge | Parameter norm |
| `tb_learning_current_learning_rate` | `host` | gauge | Learning rate |
| `tb_learning_total_weights` | `host` | gauge | Total model parameters |
| `tb_perf_step_time_seconds` | `host` | gauge | Wall-clock time per step |
| `tb_perf_per_device_tflops` | `host` | gauge | Per-device TFLOP/s |
| `tb_perf_per_device_tokens_per_sec` | `host` | gauge | Per-device tokens/sec |
| `tb_learning_moe_lb_loss` | `host` | gauge | MoE load balance loss (only present for MoE models) |
| `tb_metrics_plugin_staleness_fill` | `host` | gauge | 0=real data, 1=synthetic fill |
