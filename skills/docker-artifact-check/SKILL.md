---
name: docker-artifact-check
description: Audit AMD ROCm training Docker containers for installed software versions, git hashes, branches, source code, and repo links. Use when the user asks to analyze a container environment, check software versions, find git hashes, or inventory installed AMD/ROCm/JAX/MaxText artifacts.
---

# AMD Training Docker Artifact Check

Inventory all key software in an AMD ROCm training container: versions, git hashes, branches, source code presence, and upstream repos.

## Components to Check

| Component | Pip Package(s) | Typical Source Path | Upstream Repo |
|---|---|---|---|
| JAX | `jax` | `/opt/jax/` | `jax-ml/jax` |
| jaxlib (contains XLA) | `jaxlib` | (built from `/opt/xla/`) | `ROCm/xla` |
| ROCm-JAX plugin | `jax-rocm7-plugin`, `jax-rocm7-pjrt` | `/opt/rocm-jax/` | `ROCm/rocm-jax` |
| ROCm libraries | system debs | `/workspace/rocm-libraries/` | `ROCm/rocm-libraries` |
| ROCm systems | system debs | `/workspace/rocm-systems/` | `ROCm/rocm-systems` |
| MaxText | `maxtext` | `/workspace/maxtext/` | `ROCm/maxtext` |
| RCCL | system deb + custom build | `/workspace/rccl/` | `ROCm/rccl` |
| AMD-ANP | N/A | `/workspace/amd-anp/` | `ROCm/amd-anp` |
| maxtext-slurm | N/A | `/maxtext-slurm/` | `AMD-AGI/maxtext-slurm` |

## Step-by-Step Workflow

### Step 0: Check for build manifest

If the container has a build-time manifest, read it and skip to the Output Template — no probing needed.

```bash
if [[ -f /etc/build-manifest.json ]]; then
  jq . /etc/build-manifest.json
  # Done. Use the manifest to fill the Output Template directly.
  # Only continue with Steps 1-8 if the manifest is missing or incomplete.
fi
```

### Step 0.5: Execution context

You must be running commands **inside the target container**. Common ways to get a shell:

```bash
# Running Slurm job — exec into the job's container on a compute node
srun --overlap --jobid=<JOBID> --pty bash

# Standalone container from an image
docker run --rm -it <IMAGE> bash

# Existing container
docker exec -it <CONTAINER_ID> bash
```

If you are an AI agent inside a container (e.g., via `.host-cmd`), the commands below run directly. If you are on the host, enter the container first.

### Step 1: Python packages — versions and git hashes

```bash
# JAX + jaxlib versions
python3 -c "import jax; print(jax.__version__, jax.__file__)"
python3 -c "import jaxlib; print(jaxlib.__version__, jaxlib.__file__)"

# All relevant pip packages
pip list 2>/dev/null | grep -iE "jax|rocm|xla|maxtext|flax|optax|transformer.engine|xprof"

# Detailed pip metadata
pip show jax jaxlib jax-rocm7-plugin jax-rocm7-pjrt maxtext transformer-engine xprof 2>/dev/null
```

### Step 2: Embedded git hashes from version files

JAX and jaxlib embed `_git_hash` in their `version.py`. Use Python to resolve paths dynamically (avoids hardcoding the Python version):

```bash
# JAX git hash
python3 -c "from jax.version import _git_hash; print('jax _git_hash:', _git_hash)"

# jaxlib git hash
python3 -c "from jaxlib.version import _git_hash; print('jaxlib _git_hash:', _git_hash)"
```

The ROCm plugin records build-time commits from three repos in an auto-generated file:

```bash
python3 -c "from jax_rocm7_plugin.commit_info import commit_info; import json; print(json.dumps(commit_info, indent=2))"
# Returns dict with keys: "ROCm/xla", "ROCm/rocm-jax", "jax"

# Fallback if the above fails (alternative location):
python3 -c "from jax_plugins.xla_rocm7.commit_info import commit_info; import json; print(json.dumps(commit_info, indent=2))"
```

Note: The plugin `commit_info.py` "jax" hash may differ from the installed JAX wheel's `_git_hash`. The plugin hash is the jax commit used at plugin build time; the wheel hash is the jax release commit.

### Step 3: Git repos present in container

Scan for `.git` directories at known source paths:

```bash
for d in /opt/jax /opt/xla /opt/rocm-jax /workspace/maxtext /workspace/rccl \
         /workspace/amd-anp /workspace/rocm-libraries /workspace/rocm-systems \
         /maxtext-slurm; do
  if [ -d "$d/.git" ]; then
    echo "=== $d ==="
    git -C "$d" log --oneline -1
    git -C "$d" rev-parse HEAD
    git -C "$d" symbolic-ref --short HEAD 2>/dev/null || echo "(detached)"
    git -C "$d" describe --tags --always 2>/dev/null
    git -C "$d" remote -v | head -2
  else
    echo "=== $d === NOT PRESENT"
  fi
done
```

### Step 4: ROCm system stack

```bash
# ROCm version
cat /opt/rocm*/.info/version 2>/dev/null

# HIP version
/opt/rocm/bin/hipcc --version 2>&1 | head -3

# rocm-smi
/opt/rocm/bin/rocm-smi --version 2>&1

# rocprofiler-sdk (v3)
/opt/rocm/bin/rocprofv3 --version 2>&1 | head -5

# rocprofiler v2 (legacy)
/opt/rocm/bin/rocprof --version 2>&1 | head -5
```

### Step 5: ROCm library and system packages (debs)

```bash
# ROCm math/DNN libraries (from rocm-libraries monorepo)
dpkg -l 2>/dev/null | grep -iE "rocblas|rocfft|rocsolver|rocsparse|rocrand|rocprim|rocthrust|hipblas|hipfft|hipsolver|hipsparse|hipsparselt|miopen|comgr"

# ROCm system packages (from rocm-systems monorepo)
dpkg -l 2>/dev/null | grep -iE "rocprof|roctracer|rccl|hip-runtime|hsa-rocr|amd-smi|hipcc"
```

### Step 6: Additional infrastructure

```bash
# OpenMPI
mpirun --version 2>/dev/null

# UCX
ls /workspace/ucx-*/

# Python
python3 --version

# Venv location
echo $VIRTUAL_ENV; ls /opt/venv/ 2>/dev/null
```

### Step 7: Runtime-critical environment variables

These env vars materially change which library loads and how the GPU stack behaves.
Two containers with identical packages but different env vars can perform very differently.

```bash
# Library resolution order (determines which librccl.so, libhipblaslt.so, etc. wins)
echo "LD_LIBRARY_PATH=$LD_LIBRARY_PATH"

# XLA compiler flags
echo "XLA_FLAGS=$XLA_FLAGS"

# RCCL / NCCL tuning
env | grep -iE "^NCCL_|^RCCL_" | sort

# ROCm / HIP / HSA flags
env | grep -iE "^ROCM_|^HIP_|^HSA_|^GPU_MAX_HW_QUEUES" | sort

# Transformer Engine flags
env | grep -iE "^NVTE_" | sort

# JAX memory and client config
env | grep -iE "^XLA_PYTHON_CLIENT|^JAX_" | sort
```

### Step 8: Custom-built libraries

Check for libraries built from source alongside system installs:

```bash
# Custom RCCL build (vs system /opt/rocm/lib/librccl.so)
find /workspace/rccl -name "librccl*.so*" -type f 2>/dev/null

# Custom hipBLASLt (check if version differs from standard ROCm)
dpkg -l | grep hipblaslt

# AMD-ANP plugin
ls /opt/rocm/lib/librccl-anp.so 2>/dev/null
```

## Output Template

Present results in this format:

```
## Container Environment Summary

**Docker Image**: [image name from container_env.sh or user]
**Base OS**: [distro + version]
**Python**: [version] at [venv path]
**ROCm**: [version] at /opt/rocm-X.Y.Z/

### Python Packages

| Package | Version | Git Hash | Source in Container? | Path |
|---|---|---|---|---|

### ROCm System Packages

| Package | Version (dpkg) | Notes |
|---|---|---|

### Git Repos in Container

| Path | Repo | Git Hash | Branch/Tag |
|---|---|---|---|

### Runtime-Critical Environment Variables

| Variable | Value |
|---|---|

### Source Paths NOT Present (need cloning)

| Expected Path | Repo URL | Known Hash |
|---|---|---|
```

## Monorepo Mapping

When the user needs source for a specific ROCm library, know which monorepo contains it:

**[ROCm/rocm-libraries](https://github.com/ROCm/rocm-libraries)** (`/workspace/rocm-libraries/`):
`projects/`: rocblas, rocfft, rocsolver, rocsparse, rocrand, rocprim, rocthrust, hipblas, hipblaslt, hipcub, hipfft, hiprand, hipsolver, hipsparse, hipsparselt, hiptensor, miopen, composablekernel, rocwmma
`shared/`: tensile, rocroller, origami, mxdatagenerator

**[ROCm/rocm-systems](https://github.com/ROCm/rocm-systems)** (`/workspace/rocm-systems/`):
`projects/`: rocprofiler, rocprofiler-sdk, rocprofiler-register, rocprofiler-compute, rocprofiler-systems, roctracer, rccl, clr, hip, hipother, hip-tests, rocr-runtime (rocrruntime), rocminfo, rocm-core, rocmsmilib, amdsmi, aqlprofile, rdc, rocshmem

## Notes

- The `jax-rocm7-*` package names encode the ROCm major version (7). Future containers with ROCm 8 would use `jax-rocm8-*`.
- `jaxlib` version containing `+selfbuilt` means it was compiled from source but the source tree was not retained.
- The venv path may vary; check `$VIRTUAL_ENV` or look under `/opt/venv/`.
- `site-packages` path depends on Python version (e.g., `python3.12`). Adjust grep paths accordingly.
- hipBLASLt is often custom-built (different version hash from standard ROCm release).
