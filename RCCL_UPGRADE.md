# RCCL Upgrade Task

> Self-contained task file. It is the **single source of truth** for upgrading RCCL inside the
> `rocm/jax-training:maxtext-v26.2` image and saving a new image for use on this cluster.
> It is written so any AI assistant (Claude / ChatGPT / Cursor) or human can pick it up cold.

**Last updated:** 2026-06-02 · **Overall status:** ✅ COMPLETE — Parts 1 & 2 done; image `rocm/jax-training:maxtext-v26.2-rccl-pr2063` (RCCL 2.27.7 + PR #2063) built, validated, kept local on `build-host`.

---

## 0. How to use this doc (READ FIRST — instructions to the AI)

If you are an AI assistant continuing this task:

1. **Read this entire file before doing anything.** It contains the full context; do not assume prior conversation.
2. This file is the **single source of truth**. The live state lives in [Section 4 (Parameters)](#4-decisions--parameters) and [Section 5 (Progress)](#5-progress-tracker). Trust those over your memory.
3. **Do not redo a step marked `[x]` DONE.** Check the progress table first, then resume from the "Next action" pointer.
4. **After every step, update this file**: flip the status marker, fill in the "record result here" slots with the *actual* output (versions, SHAs, error text, bandwidth numbers), and append a dated line to the [Progress Log](#progress-log). Updating this doc is part of each step's definition-of-done.
5. **Record decisions, not just completion.** If you choose an RCCL tag, a gfx target, an image name, write it into Section 4 with `RESOLVED`.
6. **Where to run commands:** the `docker` CLI must be available — run the `docker ...` commands on the host that launches containers (the same host `_container.sh` uses; it auto-detects `DOCKER_BIN`). GPU validation steps must run on a node with MI GPUs.
7. **Ask the human** before: pushing to a registry, deleting images to free disk, or anything that touches other users' jobs on the shared cluster.

Status markers used throughout: `[ ]` TODO · `[~]` IN-PROGRESS · `[x]` DONE · `[!]` BLOCKED.

> ⚠️ **Persistence note:** this file is git-tracked (HEAD `5d9bea9 "image build doc"` on branch `llama3_experiment`). The updates below were once made as *uncommitted* edits and got discarded by a `git restore`/"discard changes" (see the last Progress Log entry). **To keep them, commit:** `git add RCCL_UPGRADE.md docker/Dockerfile.rccl && git commit`.

---

## 1. Goal

Take the base image `**rocm/jax-training:maxtext-v26.2`**, install a **newer version of RCCL** (the ROCm collective-communications library, AMD's NCCL equivalent) into it, save the result as a **new Docker image**, and wire that image into the maxtext-slurm launcher so training jobs use the upgraded RCCL.

Two phases:

- **Part 1 (validate):** get the install working interactively in a container, then snapshot it with `docker commit`. Goal here is to *figure out exactly what works*.
- **Part 2 (systematize):** distill the validated steps into a reproducible `Dockerfile`, rebuild from it, re-validate, and ship.

---

## 2. Environment & constraints

- **Base image:** `rocm/jax-training:maxtext-v26.2` (set in `[container_env.sh](container_env.sh)` as `DOCKER_IMAGE`).
- **Hardware / arch:** this host (`build-host`) = AMD **MI350X** = CDNA4 = `gfx950` (8 GPUs, 256 CPU, ~3 TB RAM). (MI355X is also `gfx950`; MI300X/MI325X would be `gfx942`.) Confirmed on node in Part 1, Step 0.
- **Cluster runtime is Docker.** `[_container.sh](_container.sh)` resolves `DOCKER_IMAGE` two ways:
  - a **registry tag** (it runs `docker image inspect`, else `docker pull $DOCKER_REGISTRY/$DOCKER_IMAGE`), or
  - a **local `.tar`** (if `DOCKER_IMAGE` is a path ending in `.tar`, it runs `docker load -i` and parses the loaded tag — see `_container.sh` lines ~221-245).
  So the new image can be delivered as either a pushed tag or a `.tar` file.
- **Build constraints (important):**
  - Build RCCL **against the ROCm already in the image** (`/opt/rocm`). Do **not** `apt-get install` a different ROCm — it would diverge from the ROCm that JAX/jaxlib uses.
  - Install over `/opt/rocm/lib` and **keep the soname `librccl.so.1`** (RCCL's major ABI is stable at 1). jaxlib loads RCCL via that soname, so keeping it is what makes JAX pick up the new library automatically.
  - Build **only for the gfx target you need** (`gfx950`). Each extra arch roughly multiplies compile time (RCCL generates many kernels per arch).
- **Files in this repo you'll touch / reference:**
  - `[container_env.sh](container_env.sh)` — defines `DOCKER_IMAGE`, `DOCKER_REGISTRY`, and the `USE_DOCKER_IMAGE_AINIC_DRIVER` knob.
  - `[_container.sh](_container.sh)` — launches the container; ingests `DOCKER_IMAGE`.
  - `docker/Dockerfile.rccl` — **created in Part 2** (✅ now exists; self-checking, reproduces the image).

### ⚠️ Caveat — do not confuse this with the NIC/ionic issue

A newer RCCL changes the **collective-communications library** only. It does **not** change the host's AINIC / `libionic` NIC driver or its kernel ABI. If the real problem is "RDMA falls back to TCP sockets" due to a container-`libionic1` vs host-firmware mismatch, that is handled by `USE_DOCKER_IMAGE_AINIC_DRIVER=false` (bind-mounting host IB libs), **not** by this RCCL upgrade. If your motivation for upgrading RCCL is networking/performance, verify with `NCCL_DEBUG=INFO` that collectives actually land on IB/RoCE after the upgrade.

---

## 3. Prerequisites

- `docker` CLI usable on the build host.
- Network access to `github.com` (to clone RCCL) from inside the container.
- A node with MI GPUs for the validation steps.
- Enough disk for the base image (~tens of GB) plus build artifacts and the committed/saved image.

---

## 4. Decisions & parameters

Fill these in as you resolve them; later steps reference them by name.


| Parameter                                    | Value                                                | Status        |
| -------------------------------------------- | ---------------------------------------------------- | ------------- |
| `<GFX>` (GPU arch)                           | `gfx950` — host `build-host` = MI350X (CDNA4) | RESOLVED |
| `<ROCM_VER>` (ROCm in base image)            | `7.1.1` (`/opt/rocm-7.1.1`)                          | RESOLVED |
| `<RCCL_CURRENT>` (RCCL in base image)        | `2.27.7` (`librccl.so.1.0.70101`; from rocm-rel-7.1) | RESOLVED |
| `<RCCL_REF>` (target RCCL tag or commit SHA) | **`185e78a8`** = merge of PR #2063, RCCL **2.27.7** + side-stream fix (user-confirmed 2026-06-02) | RESOLVED |
| `<NEW_TAG>` (new image name:tag)             | `rocm/jax-training:maxtext-v26.2-rccl-pr2063` (ver stays 2.27.7) | RESOLVED |
| `<DIST>` (how the image reaches nodes)       | **local image on this host** `build-host` (single node for now; user choice 2026-06-02) | RESOLVED |
| `<TAR_PATH>` (if `.tar`)                     | n/a — local only (no `.tar`/registry for now)        | RESOLVED |


> Pin `<RCCL_REF>` to a **tag or commit SHA**, never a moving branch like `develop`, so Part 2's Dockerfile reproduces exactly what Part 1 validated. *(We pinned the commit SHA `185e78a8` — the PR #2063 merge commit on develop — which satisfies this.)*

---

## 5. Progress tracker

**Current state:** **Parts 1 & 2 COMPLETE.** Reproducible image `rocm/jax-training:maxtext-v26.2-rccl-pr2063` (built from `docker/Dockerfile.rccl`, re-validated) carries RCCL 2.27.7 + PR #2063 on `gfx950`. Distribution = local image on this host (user choice).
**Next action:** none — **task closed by user 2026-06-02.** P2.5 (end-to-end MaxText launch) intentionally skipped. Image ready to use locally on `build-host`. Helper container `rccl-build` left running per user request.


| #    | Step                                                               | Status |
| ---- | ------------------------------------------------------------------ | ------ |
| P0   | Confirm gfx target; probe current ROCm + RCCL; decide `<RCCL_REF>` | `[x]`  |
| P1.1 | Start interactive build container                                  | `[x]`  |
| P1.2 | Backup original RCCL; clone, build, install RCCL into `/opt/rocm`  | `[x]`  |
| P1.3 | Validate with rccl-tests on a GPU node (version + bandwidth)       | `[x]`  |
| P1.4 | `docker commit` to `<NEW_TAG>`                                     | `[x]`  |
| P1.5 | (optional) sanity: a JAX/MaxText job loads the new RCCL            | `[x]`  |
| P2.1 | Write `docker/Dockerfile.rccl` from the validated commands         | `[x]`  |
| P2.2 | `docker build` the image from the Dockerfile                       | `[x]`  |
| P2.3 | Rebuild-and-re-validate on a GPU node (fresh build)                | `[x]`  |
| P2.4 | Save (`docker save` → `.tar`) or push to registry; record location | `[x]`  |
| P2.5 | Plug into maxtext-slurm via `DOCKER_IMAGE`; multi-node sanity      | `[ ]` (skipped — single node, local image; user choice) |


### Progress log

*Append one dated line per session: what you did, what you decided, what broke.*

- **2026-06-02** — Started on host `build-host` (1 node, 8× MI350X = `gfx950`/CDNA4, 256 CPU, Docker 29.4.3, running on host — not in a container). Base image `rocm/jax-training:maxtext-v26.2` already present locally (74.3 GB). Probed base image: ROCm **7.1.1**, RCCL **2.27.7** (soname `librccl.so.1`); build toolchain (hipcc, cmake 3.31.6, git, rocm-cmake, libibverbs) already present → no `apt-get` expected. Upstream survey: only `develop` is newer by version (**2.28.3**, SHA `94316ce`); `release/rocm-rel-7.2` (SHA `96a25b5`) is still 2.27.7 (newer commits, same version); top semantic tag is only `v2.26.6-1` (older than image). Disk: `/` 529 GB free, `/home` 6.3 TB free, no `/mnt/vast`.
- **2026-06-02 (goal clarified)** — Purpose of the upgrade is to obtain **RCCL PR #2063** ("Use one side stream per process" — fixes RCCL creating/destroying thousands of streams that interfere with HIP graph capture; the "empty stream" issue). PR merged to `develop` on 2025-12-02 as commit `185e78a8`. Containment check (GitHub compare API): only `develop` (now 2.28.3) and `release/therock-7.11` (2.27.7) contain it; `rocm-rel-7.2` / `7.2.0.1` did **not** cherry-pick it. **Key finding:** at merge commit `185e78a8`, develop was still **2.27.7** — same version as the image → pinning it gives the fix with minimal drift and lowest build risk vs ROCm 7.1.1. Verified `git fetch`-by-sha works and the side-stream change is present (`src/init.cc`, `src/include/alloc.h`). User **confirmed `<RCCL_REF>` = `185e78a8`**, `<NEW_TAG>` = `rocm/jax-training:maxtext-v26.2-rccl-pr2063`.
- **2026-06-02 (Part 1 DONE)** — Built RCCL @ `185e78a8` for `gfx950` vs in-image ROCm 7.1.1: **no extra apt pkgs**; cmake `-DGPU_TARGETS=gfx950` (note: RCCL's CMake **ignores** `AMDGPU_TARGETS`); `make -j128` ~11 min → `librccl.so.1.0` (12 MB; `.hip_fatbin`≈8.8 MB; `rcclGitHash=HEAD:185e78a`). **Removed old packaged `librccl.so.1.0.70101`** + `ldconfig` so soname `librccl.so.1 → librccl.so.1.0` (else ldconfig version-sorts the old file ahead of the new one). Validated: rccl-tests `all_reduce_perf -g 8` → `#wrong=0`, `2.27.7-HEAD:185e78a`, ~360 GB/s busbw @256 MB; JAX `pmap(psum)` over 8 dev correct, loads our lib. Cleaned trees+backup; **`docker commit` → `rocm/jax-training:maxtext-v26.2-rccl-pr2063`** (74.5 GB, id `6fc5b76`).
- **2026-06-02 (Part 2 DONE)** — Wrote `docker/Dockerfile.rccl` (no apt; `GPU_TARGETS=gfx950`; fetch-by-sha; removes old packaged lib + `ldconfig`; in-build self-checks). `docker build` → `…-rccl-pr2063-df` (id `8e05cb69`), self-checks passed. Fresh-container re-validation: `librccl.so.1→so.1.0`, `2.27.7-HEAD:185e78a`, busbw 360.6 GB/s @256 MB, OOB `0 OK`. Retagged canonical **`rocm/jax-training:maxtext-v26.2-rccl-pr2063` = the reproducible Dockerfile build** (id `8e05cb69`); kept Part-1 committed image as `…-rccl-pr2063-committed` (id `6fc5b76`). Distribution: **local only** (user choice). P2.5 (end-to-end MaxText) skipped by user.
- **2026-06-02 (doc reverted, then restored)** — The four log entries above were written during the session as **uncommitted** working-tree edits on git-tracked `RCCL_UPGRADE.md` (HEAD `5d9bea9` on `llama3_experiment`, cloned 14:08 UTC). At ~15:16 UTC the file was reset to the committed original — no reflog/stash trace, and the untracked `docker/Dockerfile.rccl` survived, which is consistent with a `git restore` / `git checkout -- RCCL_UPGRADE.md` (or IDE "discard changes"), **not** an assistant action (all assistant git calls were read-only). Re-applied here. **These edits are still uncommitted — `git commit` them to make them stick.**

---

## Part 1 — Validate the install (interactive + `docker commit`)

Goal: get RCCL building and passing tests inside a live container, then snapshot it. Capture every command you run (e.g. start the shell under `script /tmp/rccl-build.log`, or keep them in a scratch `.sh`) so Part 2 can reuse them verbatim.

> **How this was actually run (agent, not interactive):** instead of `docker run -it ... bash`, the container was started detached — `docker run -d --name rccl-build <gpu flags> rocm/jax-training:maxtext-v26.2 sleep infinity` — and each step below was issued via `docker exec rccl-build bash -lc '…'`. Same persistent container, same commands, just scriptable.

### Step 0 — Confirm arch and probe current versions  `[x]`

On a GPU node, confirm the arch:

```bash
rocminfo | grep -m1 -o 'gfx[0-9a-f]*'      # expect gfx950 on MI355X
```

Probe the base image's ROCm and RCCL (no GPU needed):

```bash
docker run --rm rocm/jax-training:maxtext-v26.2 bash -lc '
  echo "ROCm:"; cat /opt/rocm/.info/version 2>/dev/null;
  echo "RCCL libs:"; ls -la /opt/rocm/lib/librccl.so*'
```

Then pick `<RCCL_REF>` (a tag newer than `<RCCL_CURRENT>` but compatible with `<ROCM_VER>`).
**Record results here →** ROCm: `7.1.1` · current RCCL: `2.27.7` (soname `librccl.so.1` → `librccl.so.1.0.70101`) · gfx: `gfx950` (MI350X) · in-image toolchain: hipcc/HIP 7.1 + clang 20, cmake 3.31.6, git 2.43.0, rocm-cmake 0.14.0, libibverbs.so.1 → **no apt expected** · chosen `<RCCL_REF>`: **`185e78a8`** (PR #2063 merge: RCCL 2.27.7 + side-stream fix) — CONFIRMED

### Step 1 — Start the build container  `[x]`

GPU flags are included so the same container can run the validation in Step 3. (Building alone does not need a GPU.)

```bash
docker run -it --name rccl-build \
  --device=/dev/kfd --device=/dev/dri \
  --group-add video --group-add render \
  --security-opt seccomp=unconfined \
  --shm-size=16g \
  rocm/jax-training:maxtext-v26.2 bash
```

*(Run as a detached `sleep infinity` container + `docker exec` for the agent flow — see note above.)*

### Step 2 — Backup, clone, build, install (inside the container)  `[x]`

```bash
# Backup the original RCCL so you can revert / compare
mkdir -p /opt/rccl-backup && cp -a /opt/rocm/lib/librccl.so* /opt/rccl-backup/
ls -la /opt/rccl-backup

# Clone the target RCCL (pin tag or SHA from Section 4)
git clone --depth 1 -b <RCCL_REF> https://github.com/ROCm/rccl /tmp/rccl

# Build against the in-image ROCm with hipcc, install over /opt/rocm
cd /tmp/rccl && mkdir build && cd build
CXX=/opt/rocm/bin/hipcc cmake \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX=/opt/rocm \
  -DAMDGPU_TARGETS=<GFX> \
  ..
make -j"$(nproc)"
make install
ldconfig

# Confirm the soname chain is intact (librccl.so.1 -> librccl.so.1.x.x)
ls -la /opt/rocm/lib/librccl.so*
```

Notes / likely fixes:

- If `cmake` complains about missing tooling: `apt-get install -y cmake rocm-cmake git` (these come from the image's existing ROCm repo, so they match `<ROCM_VER>`). **Write down anything you install** — it must go into the Dockerfile in Part 2.
- For IB/RoCE support at runtime, `libibverbs` must be present (it usually is in a training image); `apt-get install -y libibverbs-dev` if the build asks for it.

> **What actually happened (deltas from the template above):** (a) `<RCCL_REF>` is a **commit SHA**, so the clone was a fetch-by-sha — `git init && git remote add origin https://github.com/ROCm/rccl && git fetch --depth 1 origin 185e78a8… && git checkout FETCH_HEAD` (you can't `git clone -b <sha>`). (b) RCCL's CMake reads **`GPU_TARGETS`**, not `AMDGPU_TARGETS` (the latter warns "unused"). (c) The standalone build installs `librccl.so.1.0`; the base image shipped `librccl.so.1.0.70101`. Both have soname `librccl.so.1`, and `ldconfig` version-sorts `1.0.70101` **ahead** of `1.0`, so after `make install` you must **`rm /opt/rocm/lib/librccl.so.1.0.70101`** then `ldconfig`. (d) No extra `apt` packages were needed.

**Record results here →** new lib: `librccl.so.1.0` (12 MB, gfx950-only Release; `.hip_fatbin`≈8.8 MB) · soname chain: `librccl.so → librccl.so.1 → librccl.so.1.0` · embedded `rcclGitHash="HEAD:185e78a"` (PR #2063 ✓) · version still 2.27.7 · extra packages installed: **none** (image toolchain sufficient)

### Step 3 — Validate with rccl-tests (on a GPU node)  `[x]`

```bash
git clone https://github.com/ROCm/rccl-tests /tmp/rccl-tests
cd /tmp/rccl-tests && make MPI=0 HIP_HOME=/opt/rocm -j
./build/all_reduce_perf -b 8 -e 256M -f 2 -g 8
```

The header prints `# RCCL version : X.Y.Z` — confirm it matches `<RCCL_REF>` and that the all-reduce completes with sane bandwidth.
**Record results here →** `Librccl path: /opt/rocm-7.1.1/lib/librccl.so.1` (ldd confirms test → our lib) · RCCL version reported: **`2.27.7-HEAD:185e78a`** (PR #2063 ✓) · 8×MI350X all_reduce 8B–256MB: **#wrong=0**, OOB=`0 OK` · busbw @256 MB ≈ **360 GB/s** (avg 78 across all sizes incl. tiny) · pass? **YES**

### Step 4 — Clean up and commit  `[x]`

```bash
# Inside the container: remove build trees so the committed image isn't bloated
rm -rf /tmp/rccl /tmp/rccl-tests
exit

# On the host: snapshot the container as the new image
docker commit rccl-build <NEW_TAG>
docker images | grep maxtext-v26.2-rccl
```

**Record results here →** committed image: `rocm/jax-training:maxtext-v26.2-rccl-pr2063` (id `6fc5b76f2a6c`) · image size: **74.5 GB** (only +0.2 GB over base; old 873 MB lib + build trees + backup removed) · labels: `rccl.upgrade=2.27.7+pr2063`, `rccl.commit=185e78a8…`, `rccl.base=…v26.2`. *(Later superseded as canonical by the Part 2 Dockerfile build; this committed image kept as `…-rccl-pr2063-committed`.)*

### Step 5 — (optional) End-to-end sanity  `[x]`

Run a short MaxText job (single node first) on `<NEW_TAG>` with `NCCL_DEBUG=INFO` and confirm the logs show the new RCCL version and (if relevant) that collectives use IB/RoCE rather than sockets.
**Record results here →** JAX (`/opt/venv`) sees **8× "AMD Radeon Graphics"**; `pmap(psum)` over 8 devices → correct (sum 0..7 = 28 on every device). `NCCL_DEBUG=VERSION` shows `RCCL version : 2.27.7-HEAD:185e78a` and `Librccl path : /opt/rocm-7.1.1/lib/librccl.so.1` → **jaxlib loads the new RCCL via soname automatically**. (Lightweight JAX collective only; a full MaxText job with `NCCL_DEBUG=INFO` and IB/RoCE confirmation was **not** run — that's the remaining optional check.)

---

## Part 2 — Systematize into a Dockerfile

Goal: turn the validated commands into a reproducible, shareable artifact. The hand-built image from Part 1 is the reference; this part reproduces it cleanly.

### Step 1 — Write `docker/Dockerfile.rccl`  `[x]`

Create the file with the *exact* validated commands (including any extra `apt-get install` you noted in Part 1 Step 2):

```dockerfile
ARG BASE=rocm/jax-training:maxtext-v26.2
FROM ${BASE}

# Parameters (override at build time with --build-arg)
ARG RCCL_BRANCH=<RCCL_REF>     # pin a tag or commit SHA
ARG GFX=gfx950                 # MI355X=gfx950; MI300/MI325=gfx942

RUN set -eux; \
    # 0) backup original RCCL (for reference inside the image)
    mkdir -p /opt/rccl-backup && cp -a /opt/rocm/lib/librccl.so* /opt/rccl-backup/ || true; \
    # 1) (only if Part 1 needed them) build deps from the image's ROCm repo:
    # apt-get update && apt-get install -y cmake rocm-cmake git libibverbs-dev && rm -rf /var/lib/apt/lists/*; \
    # 2) clone + build against /opt/rocm, install over it
    git clone --depth 1 -b ${RCCL_BRANCH} https://github.com/ROCm/rccl /tmp/rccl; \
    cd /tmp/rccl && mkdir build && cd build; \
    CXX=/opt/rocm/bin/hipcc cmake \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX=/opt/rocm \
        -DAMDGPU_TARGETS=${GFX} \
        ..; \
    make -j"$(nproc)"; \
    make install; \
    ldconfig; \
    # 3) clean up build tree so the layer stays small
    rm -rf /tmp/rccl
```

> **⚠️ The block above is the original illustrative template. The actual file written is [`docker/Dockerfile.rccl`](docker/Dockerfile.rccl), which differs in ways that matter:** it pins `RCCL_COMMIT=185e78a8` and clones via **fetch-by-SHA** (not `git clone -b <branch>`); uses **`GPU_TARGETS=gfx950`** (RCCL's CMake **ignores** `AMDGPU_TARGETS`); after `make install` it **removes the old packaged `librccl.so.1.0.70101`** then runs `ldconfig` (so the `librccl.so.1` soname points at the new build); adds build-time **self-checks** (`readlink librccl.so.1 == librccl.so.1.0` and `strings | grep <short-sha>`); needs **no `apt-get`**; and keeps **no** in-image backup (recoverable from the base image).

### Step 2 — Build the image  `[x]`

```bash
DOCKER_BUILDKIT=1 docker build -f docker/Dockerfile.rccl \
  --build-arg RCCL_BRANCH=<RCCL_REF> \
  --build-arg GFX=<GFX> \
  -t <NEW_TAG> .
```

*(Actual: built with an **empty context** since the Dockerfile COPIES nothing — `DOCKER_BUILDKIT=1 docker build -f docker/Dockerfile.rccl -t rocm/jax-training:maxtext-v26.2-rccl-pr2063-df /tmp/empty-ctx`. The real file uses `--build-arg RCCL_COMMIT=…` rather than `RCCL_BRANCH`.)*

### Step 3 — Rebuild-and-re-validate (on a GPU node)  `[x]`

This is a **fresh** build (clean `/tmp`, different cache), so it can expose a dependency you installed by hand in Part 1 but forgot to add to the Dockerfile. Re-run the rccl-tests check from Part 1 Step 3 inside the freshly built `<NEW_TAG>`:

```bash
docker run -it --rm \
  --device=/dev/kfd --device=/dev/dri \
  --group-add video --group-add render \
  --security-opt seccomp=unconfined --shm-size=16g \
  <NEW_TAG> bash -lc '
    git clone https://github.com/ROCm/rccl-tests /tmp/rccl-tests
    cd /tmp/rccl-tests && make MPI=0 HIP_HOME=/opt/rocm -j
    ./build/all_reduce_perf -b 8 -e 256M -f 2 -g 8'
```

**Record results here →** Dockerfile build OK? **YES** (`#5 DONE 648.7s`; in-build self-checks `readlink==librccl.so.1.0` + `grep 185e78a` passed) · re-validation (fresh container from `-df`): `librccl.so.1→librccl.so.1.0`, `RCCL version 2.27.7-HEAD:185e78a`, `Librccl path=/opt/rocm-7.1.1/lib/librccl.so.1`, busbw **360.6 GB/s** @256 MB, OOB `0 OK` · pass? **YES**

### Step 4 — Save or push  `[x]`

Pick `<DIST>` from Section 4.

```bash
# (a) Local tarball (simple; put it on storage every node can read)
docker save <NEW_TAG> -o <TAR_PATH>

# (b) Private registry (more scalable for many nodes)
docker tag  <NEW_TAG> <registry>/maxtext:v26.2-rccl-<ref>
docker push <registry>/maxtext:v26.2-rccl-<ref>
```

**Record results here →** delivered as: **local Docker image** (user choice; no `.tar`/registry) · location: host `build-host` · canonical tag **`rocm/jax-training:maxtext-v26.2-rccl-pr2063`** (= reproducible Dockerfile build, id `8e05cb69`); Part-1 committed image kept as `…-rccl-pr2063-committed` (id `6fc5b76`)

### Step 5 — Plug into maxtext-slurm  `[ ]` (skipped — single node, local image; user choice)

Override `DOCKER_IMAGE` at launch (see the example at the top of `[container_env.sh](container_env.sh)`):

```bash
# (a) tarball — _container.sh will docker-load it
DOCKER_IMAGE=<TAR_PATH> <your run script> <model> -- ...

# (b) registry tag — put creds in container_env.local.sh if private
DOCKER_IMAGE=<registry>/maxtext:v26.2-rccl-<ref> <your run script> <model> -- ...
```

> **Multi-node:** the image must be reachable on **every** node the job lands on — either each node pulls it from the registry, or the `.tar` lives on shared storage and each node `docker load`s it on first use.

For this local single-node image, just point `DOCKER_IMAGE` at the tag:

```bash
DOCKER_IMAGE=rocm/jax-training:maxtext-v26.2-rccl-pr2063 <your run script> <model> -- ...
```

**Record results here →** **skipped by user** (no end-to-end MaxText launch). For multi-node later, distribute via registry push or `docker save` `.tar` first (Section 4 `<DIST>` would change).

---

## 6. Acceptance criteria (task is DONE when all hold)

- ✅ rccl-tests `all_reduce_perf` passes on `gfx950` inside the new image. (`#wrong=0`, OOB `0 OK`, ~360 GB/s @256 MB.)
- ✅ The new image's RCCL carries **PR #2063**. NB: the version **number** stays `2.27.7` (the fix landed on develop before the 2.28 bump), so identity is confirmed by the embedded **`rcclGitHash=HEAD:185e78a`** / `RCCL version 2.27.7-HEAD:185e78a` — not by a higher version number. (The base image's 2.27.7 came from `rocm-rel-7.1` and lacks the PR.)
- ✅ A JAX job on the new image loads the new RCCL (`NCCL_DEBUG=VERSION` → `2.27.7-HEAD:185e78a`, `Librccl path=/opt/rocm-7.1.1/lib/librccl.so.1`). ⚠️ A full MaxText job with `NCCL_DEBUG=INFO` (and IB/RoCE confirmation) was **not** run — P2.5 skipped by user.
- ⏭️ Multi-node training runs on the new image — **N/A** (single node, local image by user choice).
- ✅ `docker/Dockerfile.rccl` reproduces the image from scratch (Part 2 Step 3 passed).
- ✅ This doc updated: Section 4 resolved, Section 5 tracker P0–P2.4 `[x]`, results recorded. *(Edits are uncommitted — `git commit` to persist; see Progress Log.)*

---

## 7. Artifacts & references

- New image tag: **`rocm/jax-training:maxtext-v26.2-rccl-pr2063`** (id `8e05cb69`, = reproducible Dockerfile build) · Part-1 committed twin: `rocm/jax-training:maxtext-v26.2-rccl-pr2063-committed` (id `6fc5b76`)
- Tarball (if used): n/a — local image only
- Registry ref (if used): n/a — local only
- Dockerfile: `docker/Dockerfile.rccl` ✅ (defaults `RCCL_COMMIT=185e78a8`, `GFX=gfx950`; self-checking)
- RCCL source: [https://github.com/ROCm/rccl](https://github.com/ROCm/rccl) (ref `185e78a8` = PR #2063 merge commit; repo banner says "[DEPRECATED] moved to ROCm/rocm-systems", but the SHA is still fetchable there)
- PR: [ROCm/rccl#2063 "Use one side stream per process"](https://github.com/ROCm/rccl/pull/2063)
- rccl-tests: [https://github.com/ROCm/rccl-tests](https://github.com/ROCm/rccl-tests)
- Launcher config: `[container_env.sh](container_env.sh)` · ingest logic: `[_container.sh](_container.sh)`
- Helper container `rccl-build` (detached, idle) left running on the host; remove with `docker rm -f rccl-build`.
