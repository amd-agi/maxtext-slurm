# Build & Verify the RCCL-upgraded image

How to build the RCCL-upgraded training image from [`Dockerfile.rccl`](Dockerfile.rccl) and how to prove the RCCL upgrade actually took effect. This doubles as a **tutorial** (reusable commands + expected output) and a **report** (the real output captured on a full run, 2026-06-02).

> TL;DR — `docker build -f docker/Dockerfile.rccl` produces `rocm/jax-training:maxtext-v26.2-rccl-pr2063`, an exact copy of the base `rocm/jax-training:maxtext-v26.2` with RCCL rebuilt to include [ROCm/rccl PR #2063](https://github.com/ROCm/rccl/pull/2063) ("Use one side stream per process"). **Verdict of the 2026-06-02 run: PASS** — built from scratch in ~10.9 min, rccl-tests `all_reduce_perf` on 8x MI350X had `#wrong=0`, and JAX loads the new library automatically.

---

## 1. What this is and why

- The base image `rocm/jax-training:maxtext-v26.2` ships **RCCL 2.27.7** (built from the `rocm-rel-7.1` branch).
- That 2.27.7 is **missing** PR #2063, which makes RCCL use **one side stream per process** instead of creating/destroying thousands of streams that interfere with HIP graph capture (the "empty stream" problem).
- The Dockerfile rebuilds RCCL at the **exact PR #2063 merge commit `185e78a8`**, against the in-image ROCm 7.1.1, for `gfx950` only, installed over `/opt/rocm` keeping the `librccl.so.1` soname so jaxlib picks it up automatically.

**Important — the version number does not change.** PR #2063 landed on `develop` while it was still `2.27.7` (before the 2.28 bump), so both the old and new libraries report `2.27.7`. You tell them apart by the **embedded git hash**: the upgraded one reports `2.27.7-HEAD:185e78a`. Do not expect a higher version number.

---

## 2. Prerequisites

Build host (where you run `docker build`):
- Docker (BuildKit). No GPU required to build — compilation is pure CPU (hipcc -> gfx950 code objects).
- The base image `rocm/jax-training:maxtext-v26.2` available locally or pullable from `docker.io` (~74 GB on first pull).
- Network access to `github.com` during the build (the Dockerfile `git fetch`es RCCL by SHA).
- x86-64 host (the in-image ROCm toolchain is x86-64).
- Disk: base image (~74 GB) + a small extra layer + build scratch.

Run host (where you actually run training/tests):
- **AMD `gfx950` GPUs (MI350X / MI355X).** The image's RCCL contains gfx950 kernels only. For MI300X/MI325X (`gfx942`), rebuild with `--build-arg GFX=gfx942`.

---

## 3. Quick start

```bash
# Build (empty context -- the Dockerfile COPIES nothing). Defaults already pin
# RCCL_COMMIT=185e78a8 (PR #2063) and GFX=gfx950, so no --build-arg needed.
DOCKER_BUILDKIT=1 docker build -f docker/Dockerfile.rccl \
  -t rocm/jax-training:maxtext-v26.2-rccl-pr2063 "$(mktemp -d)"

# Use it in maxtext-slurm by overriding DOCKER_IMAGE (see container_env.sh):
DOCKER_IMAGE=rocm/jax-training:maxtext-v26.2-rccl-pr2063 ./run_local.sh <model> -- ...
```

That's it. The rest of this doc explains each step and how to verify.

---

## 4. Step 1 - Build from the Dockerfile

```bash
DOCKER_BUILDKIT=1 docker build --no-cache -f docker/Dockerfile.rccl \
  -t rocm/jax-training:maxtext-v26.2-rccl-pr2063 "$(mktemp -d)"
```

- `--no-cache` forces a genuine recompile (otherwise BuildKit reuses the cached RUN layer and finishes instantly). Drop it for everyday rebuilds.
- The `RUN` clones RCCL by SHA, configures with `GPU_TARGETS=gfx950`, `make -j$(nproc)`, `make install`, then removes the old packaged `librccl.so.1.0.70101` and runs `ldconfig`. It ends with two **build-time self-checks** that fail the build if the upgrade did not take:
  - `test "$(readlink /opt/rocm/lib/librccl.so.1)" = "librccl.so.1.0"`
  - `strings /opt/rocm/lib/librccl.so.1 | grep -q <short-sha>`

What you should see at the end (actual, 2026-06-02 run):

```text
#5 650.8 -- Installing: /opt/rocm/bin/rcclras
#5 650.8 + find /opt/rocm/lib -maxdepth 1 -name librccl.so.1.* ! -name librccl.so.1.0 -delete
#5 650.8 + ldconfig
#5 650.9 + readlink /opt/rocm/lib/librccl.so.1
#5 650.9 + test librccl.so.1.0 = librccl.so.1.0
#5 650.9 + strings /opt/rocm/lib/librccl.so.1
#5 650.9 + grep -q 185e78a
#5 650.9 + rm -rf /tmp/rccl
#5 DONE 650.9s
#6 exporting to image
#6 writing image sha256:39e8323550ce471194d9d1321832f4dbcb9a1a736f8ba559db7852a451447118 done
#6 naming to docker.io/rocm/jax-training:maxtext-v26.2-rccl-pr2063 done
BUILD_EXIT=0
```

- Build time: **~10.9 min** (652 s) on this host (256 vCPU; linking the single .so dominates the tail).
- Result: image id `39e8323550ce`, size ~74.3 GB (only ~12 MB more than the base; the rest is shared layers).

---

## 5. Step 2 - Inspect the image (no GPU needed)

Purpose: confirm the soname points at our build and the binary carries the PR commit.

```bash
TAG=rocm/jax-training:maxtext-v26.2-rccl-pr2063
docker image inspect "$TAG" --format 'labels={{json .Config.Labels}}'
docker run --rm "$TAG" bash -lc '
  ls -la /opt/rocm/lib/librccl.so*
  readlink /opt/rocm/lib/librccl.so.1
  strings /opt/rocm/lib/librccl.so.1 | grep -m5 -iE "RCCL version :|rcclGitHash|HEAD:185e78a"
'
```

Actual output (2026-06-02):

```text
labels={... "rccl.base":"rocm/jax-training:maxtext-v26.2",
            "rccl.commit":"185e78a8f0d57294f3ec6e51882c32ab2298ea43",
            "rccl.upgrade":"2.27.7+pr2063" ...}

lrwxrwxrwx ... /opt/rocm/lib/librccl.so   -> librccl.so.1
lrwxrwxrwx ... /opt/rocm/lib/librccl.so.1 -> librccl.so.1.0
-rw-r--r-- ... 12066400  /opt/rocm/lib/librccl.so.1.0
readlink librccl.so.1: librccl.so.1.0
RCCL version 2.27.7 compiled with ROCm "7.1.1.0-38-26aae437f6"
HEAD:185e78a
RCCL version : 2.27.7
contains 185e78a? FOUND
```

Pass criteria: `librccl.so.1 -> librccl.so.1.0`, and `185e78a` present.

---

## 6. Step 3 - Before/after comparison vs the base image

Purpose: show the base image does NOT have the fix and the upgraded one does. Same probe, both images:

```bash
for img in rocm/jax-training:maxtext-v26.2 rocm/jax-training:maxtext-v26.2-rccl-pr2063; do
  echo "== $img =="
  docker run --rm "$img" bash -lc '
    readlink /opt/rocm/lib/librccl.so.1
    (strings /opt/rocm/lib/librccl.so.1 | grep -m1 185e78a && echo "PR#2063: present") || echo "PR#2063: ABSENT"'
done
```

Actual result:

| Aspect | Base `...maxtext-v26.2` | Upgraded `...-rccl-pr2063` |
| --- | --- | --- |
| `librccl.so.1` -> | `librccl.so.1.0.70101` (873 MB, ROCm-packaged) | `librccl.so.1.0` (12 MB, our build) |
| RCCL version string | `2.27.7` | `2.27.7` (same number) |
| git hash `185e78a` | **ABSENT** | **present** (`2.27.7-HEAD:185e78a`) |
| PR #2063 (one side stream) | no | yes |

The size difference (873 MB vs 12 MB) is expected: the packaged base lib is unstripped and built for many gfx arches; ours is a stripped Release for `gfx950` only.

---

## 7. Step 4 - rccl-tests on 8 GPUs (functional + bandwidth)

Purpose: prove the rebuilt gfx950 kernels actually run correctly on the GPUs, and that the test binary loads our library.

```bash
docker run --rm \
  --device=/dev/kfd --device=/dev/dri \
  --group-add video --group-add render \
  --security-opt seccomp=unconfined --shm-size=16g \
  rocm/jax-training:maxtext-v26.2-rccl-pr2063 bash -lc '
    git clone -q --depth 1 https://github.com/ROCm/rccl-tests /tmp/rccl-tests
    cd /tmp/rccl-tests && make MPI=0 HIP_HOME=/opt/rocm NCCL_HOME=/opt/rocm -j16
    ldd build/all_reduce_perf | grep -i rccl
    ./build/all_reduce_perf -b 8 -e 256M -f 2 -g 8'
```

Actual output (trimmed; full sweep 8 B - 256 MB, all `#wrong=0`):

```text
librccl.so.1 => /opt/rocm-7.1.1/lib/librccl.so.1
RCCL version : 2.27.7-HEAD:185e78a
Librccl path : /opt/rocm-7.1.1/lib/librccl.so.1
#       size  ...     busbw  #wrong  ...  busbw  #wrong
        8     ...      0.00       0  ...   0.00       0
   ...
    33554432  ...    296.44       0  ...  299.03       0
    67108864  ...    331.73       0  ...  332.39       0
   134217728  ...    350.99       0  ...  351.21       0
   268435456  ...    360.91       0  ...  361.00       0
# Out of bounds values : 0 OK
# Avg bus bandwidth    : 78.2811
```

Pass criteria: header shows `2.27.7-HEAD:185e78a` and `Librccl path` = `/opt/rocm-7.1.1/lib/librccl.so.1`; every `#wrong` is `0`; `Out of bounds values : 0 OK`. Peak busbw ~361 GB/s (single-process, intra-node XGMI) is in the sane range for MI350X.

`Librccl path` (printed by rccl-tests via `dladdr`) is the single fastest way to confirm at runtime which library was loaded.

---

## 8. Step 5 - JAX end-to-end load proof

Purpose: confirm jaxlib (what MaxText uses) loads the new RCCL via the soname and a real collective is correct.

```bash
docker run --rm \
  --device=/dev/kfd --device=/dev/dri \
  --group-add video --group-add render \
  --security-opt seccomp=unconfined --shm-size=16g \
  rocm/jax-training:maxtext-v26.2-rccl-pr2063 bash -lc '
    NCCL_DEBUG=VERSION python3 -c "
import jax, jax.numpy as jnp
print(\"device_count:\", jax.device_count(), jax.devices()[0].device_kind)
f = jax.pmap(lambda x: jax.lax.psum(x, \"i\"), axis_name=\"i\")
print(\"psum:\", [float(v) for v in f(jnp.arange(8.).reshape(8,1)).ravel()])
"'
```

Actual output:

```text
device_count: 8 | AMD Radeon Graphics
psum(0..7) per-device = [28.0, 28.0, 28.0, 28.0, 28.0, 28.0, 28.0, 28.0]
RCCL version : 2.27.7-HEAD:185e78a
Librccl path : /opt/rocm-7.1.1/lib/librccl.so.1
```

Pass criteria: 8 devices, the all-reduce result is `28` on every device (sum of 0..7), and `NCCL_DEBUG=VERSION` reports `2.27.7-HEAD:185e78a` from `/opt/rocm-7.1.1/lib/librccl.so.1`.

---

## 9. Using it in maxtext-slurm

Override `DOCKER_IMAGE` (defined in [`container_env.sh`](../container_env.sh); ingested by [`_container.sh`](../_container.sh)):

```bash
DOCKER_IMAGE=rocm/jax-training:maxtext-v26.2-rccl-pr2063 ./run_local.sh llama3_1_8b -- ...
```

To confirm the new RCCL is loaded in a real run, set `NCCL_DEBUG=VERSION` (one line) or `NCCL_DEBUG=INFO` (verbose) and look for `2.27.7-HEAD:185e78a` and the `librccl.so.1` path in the logs.

Multi-node note: the image must exist on every node. Since this is a local-only image, for multi-node you would first `docker push` it to a registry or `docker save` a `.tar` onto shared storage (then `_container.sh` can `docker load` it). See `<DIST>` in [`../RCCL_UPGRADE.md`](../RCCL_UPGRADE.md).

---

## 10. Troubleshooting / gotchas

- **`ldconfig` keeps the old library.** The standalone build installs `librccl.so.1.0`, but the base ships `librccl.so.1.0.70101`. Both have soname `librccl.so.1`, and `ldconfig` version-sorts `1.0.70101` *ahead* of `1.0`, so the soname keeps pointing at the old 873 MB lib. The Dockerfile fixes this by deleting the old file before `ldconfig` (and the self-check `readlink ... = librccl.so.1.0` would fail the build if it regressed).
- **`AMDGPU_TARGETS` is ignored.** RCCL's CMake reads `GPU_TARGETS`. Passing `AMDGPU_TARGETS` only emits a "Manually-specified variables were not used" warning and you may accidentally build all default arches. Use `-DGPU_TARGETS=gfx950`.
- **Wrong GPU arch at runtime** (`no kernel image is available for execution`): the image is `gfx950`-only. Rebuild with `--build-arg GFX=gfx942` for MI300X/MI325X.
- **`git fetch` fails in build**: the build host needs `github.com` access (proxy/airgap will break it). The Dockerfile uses fetch-by-SHA against `https://github.com/ROCm/rccl` (the repo is marked deprecated/moved to ROCm/rocm-systems, but the SHA is still fetchable).
- **Version number looks unchanged**: expected - it stays `2.27.7`; verify by the `-HEAD:185e78a` suffix / git hash, not the number.

---

## 11. This run - environment and result

- Host: `build-host`, 8x AMD Instinct MI350X (`gfx950` / CDNA4), 256 vCPU, ~3 TB RAM, Docker 29.4.3.
- In-image: ROCm 7.1.1, HIP 7.1, AMD clang 20; RCCL rebuilt at commit `185e78a8`.
- Image: `rocm/jax-training:maxtext-v26.2-rccl-pr2063` (id `39e8323550ce`, ~74.3 GB).

Result summary:

| Check | Result |
| --- | --- |
| Build from Dockerfile (`--no-cache`) | PASS (~10.9 min, `BUILD_EXIT=0`, in-build self-checks passed) |
| Image inspection (soname + git hash + labels) | PASS (`librccl.so.1 -> librccl.so.1.0`, `185e78a` FOUND) |
| Before/after vs base | PASS (base lacks `185e78a`; upgraded has it) |
| rccl-tests `all_reduce_perf -g 8` | PASS (`#wrong=0`, `0 OK`, `2.27.7-HEAD:185e78a`, ~361 GB/s) |
| JAX `pmap(psum)` load proof | PASS (8 dev, result `28`, loads our `librccl.so.1`) |

## 12. Out of scope (not verified here)

- Multi-node RCCL (this was single node, 1x8 GPUs).
- A full MaxText training job and a behavioral confirmation that the empty-stream/graph-capture issue is actually resolved under load (would use `NCCL_DEBUG=INFO` on a real run).
- IB/RoCE networking path (intra-node XGMI only here; unrelated to this RCCL change - see the NIC/ionic caveat in [`../RCCL_UPGRADE.md`](../RCCL_UPGRADE.md)).

## 13. References

- Dockerfile: [`docker/Dockerfile.rccl`](Dockerfile.rccl)
- Full task log / decisions: [`../RCCL_UPGRADE.md`](../RCCL_UPGRADE.md)
- PR: [ROCm/rccl#2063 "Use one side stream per process"](https://github.com/ROCm/rccl/pull/2063) (merge commit `185e78a8`)
- rccl-tests: [https://github.com/ROCm/rccl-tests](https://github.com/ROCm/rccl-tests)
