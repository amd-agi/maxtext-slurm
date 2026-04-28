# Known-Harmless Log Entries

## Known-harmless log entries

These patterns appear in normal, healthy jobs. Do **not** classify them as failures or mention them in the triage report:

| Pattern | Why it's harmless |
|---------|-------------------|
| `Failed call to cuInit: UNKNOWN ERROR (303)`, `INTERNAL: CUDA error` | JAX/XLA probes for CUDA on AMD GPU nodes. The probe fails (expected) and falls back to ROCm. Appears in every job. |
| `NCCL WARN MSCCL++: Feature not enabled` | RCCL init notice — MSCCL++ is a compile-time feature not enabled in the current build. Appears on every RCCL job. |
| `Token indices sequence length is longer than the specified maximum sequence length` | HuggingFace tokenizer truncation warning. The model handles this internally; not an error. |
| `OCI runtime exec failed` + `[exec] docker exec failed ... falling back to host-level kill` + `[cgroup] Sent SIGKILL to 0/0 processes` | Preflight cleanup killing stale containers from a previous job. The `0/0 processes` confirms there was nothing left to kill. |
| `OCI runtime exec failed: exec failed: unable to start container process: error executing setns process: exit status 1: unknown` (standalone, during teardown) | Container namespace teardown race — the container exited before Docker could exec into it for cleanup. Common during job cancellation or when containers shut down quickly. No data loss or training impact. |
| `Cannot read CPU core N` (topology.cc) | XLA/ROCm topology probe on cores outside the container's cgroup. Harmless. |
| `No hardware is found. Using default TPU version: jellyfish` | XLA probes for TPU on a GPU node. Expected, falls back to GPU. |
| `No device identifiers found` (trace.cc) | XLA tracing probe. Harmless. |
| `Enabling PjRt/TPU event dependency logging` | XLA internal logging init. Harmless on GPU nodes. |
| `Fiber init: default domain = futex` (init-domain.cc) | Internal threading init. Harmless. |
| `Error response from daemon: cannot remove container ... could not kill container: tried to kill container, but did not receive an exit event` | Docker container slow to exit during teardown (e.g., stuck in RCCL busy-wait). Harmless — cleanup completes eventually. |
| `srun: error: <host>: task N: Exited with exit code 143` + `srun: Terminating StepId=` | Normal Slurm cascade after `scancel`. Exit 143 = SIGTERM. All nodes exiting with 143 confirms a clean cancellation. |
| `NODE_EXIT host=<hostname> exit=143` (all nodes) | Clean SIGTERM on every node — expected from `scancel`. Not an error. |
