// ANP plugin standalone reproducer
//
// Single-file C++ benchmark that exercises RCCL cross-node collectives at
// a size sweep that spans the LL / LL128 / Simple protocol thresholds.
// Runs without MaxText, JAX, or mpirun. Uses file-based uniqueId bootstrap.
//
// Expected result (2N x 8 MI355X, ionic RoCE):
//   * Small sizes (<=~64 KB): ANP == noANP (LL / LL128 path, no CTS offload)
//   * Large sizes (>=~1 MB):  ANP is 1.5-2.3x slower than noANP (Simple
//                             protocol, engages ionic CTS-offload + GDA recv)
//
// Compile (inside the training container):
//   hipcc -std=c++17 -O2 -o anp_repro anp_repro.cc -lrccl
//
// Runtime env:
//   GLOBAL_RANK, WORLD_SIZE, LOCAL_RANK, UID_FILE
//   REPRO_SIZES (optional; bytes, comma-separated)
//   REPRO_ITERS (optional; default 30)
//   REPRO_OPS   (optional; subset of {ag,rs,ar}; default all)

#include <rccl/rccl.h>
#include <hip/hip_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#define HIPCHECK(expr) do {                                                    \
    hipError_t _e = (expr);                                                    \
    if (_e != hipSuccess) {                                                    \
      fprintf(stderr, "[rank %d] HIP error %s at %s:%d: %s\n",                \
              g_rank, #expr, __FILE__, __LINE__, hipGetErrorString(_e));      \
      std::exit(1);                                                            \
    }                                                                          \
  } while (0)

#define NCCLCHECK(expr) do {                                                   \
    ncclResult_t _e = (expr);                                                  \
    if (_e != ncclSuccess) {                                                   \
      fprintf(stderr, "[rank %d] NCCL error %s at %s:%d: %s\n",               \
              g_rank, #expr, __FILE__, __LINE__, ncclGetErrorString(_e));     \
      std::exit(1);                                                            \
    }                                                                          \
  } while (0)

static int g_rank = -1;
static int g_world = -1;

// ---------- Env helpers ------------------------------------------------
static std::string env_str(const char* k, const char* def = "") {
  const char* v = std::getenv(k);
  return v ? std::string(v) : std::string(def);
}

static int env_int(const char* k, int def) {
  const char* v = std::getenv(k);
  return v ? std::atoi(v) : def;
}

static std::vector<size_t> default_sweep() {
  // Covers LL, LL128, and Simple protocol regimes. Deliberately includes the
  // ~1 GB point because the profile from MaxText FSDP showed the ANP
  // regression concentrated there.
  return {
      1024,              // 1 KB    — LL
      16 * 1024,         // 16 KB   — LL / LL128 boundary
      65 * 1024,         // 65 KB   — LL128
      256 * 1024,        // 256 KB  — LL128 / Simple boundary
      1 * 1024 * 1024,   // 1 MB    — Simple
      16 * 1024 * 1024,  // 16 MB
      128 * 1024 * 1024, // 128 MB
      512ULL * 1024 * 1024,  // 512 MB
      1ULL * 1024 * 1024 * 1024,  // 1 GB
      2ULL * 1024 * 1024 * 1024,  // 2 GB
  };
}

static std::vector<size_t> parse_sizes(const std::string& s) {
  if (s.empty()) return default_sweep();
  std::vector<size_t> out;
  std::stringstream ss(s);
  std::string tok;
  while (std::getline(ss, tok, ',')) {
    if (!tok.empty()) out.push_back(std::strtoull(tok.c_str(), nullptr, 10));
  }
  return out;
}

// ---------- File-based ncclUniqueId bootstrap --------------------------
static void write_uid(const std::string& path, const ncclUniqueId& id) {
  std::string tmp = path + ".tmp";
  std::ofstream f(tmp, std::ios::binary);
  if (!f) { fprintf(stderr, "cannot write %s\n", tmp.c_str()); std::exit(2); }
  f.write(reinterpret_cast<const char*>(&id), sizeof(id));
  f.close();
  std::rename(tmp.c_str(), path.c_str());
}

static void read_uid(const std::string& path, ncclUniqueId& id) {
  for (int attempt = 0; attempt < 600; ++attempt) {
    std::ifstream f(path, std::ios::binary);
    if (f && f.read(reinterpret_cast<char*>(&id), sizeof(id)) && f.gcount() == sizeof(id)) {
      return;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }
  fprintf(stderr, "[rank %d] timed out reading %s\n", g_rank, path.c_str());
  std::exit(3);
}

// ---------- Benchmarking kernels ---------------------------------------
struct BenchResult {
  size_t bytes;               // per-rank count (sendbuff bytes for AG/RS/AR)
  double per_op_ms;           // median per-invocation, across iters
  double best_per_op_ms;      // min, for eyeballing tail
  double bw_algbw_gbps;       // "algorithmic" BW = bytes/time (as rccl-tests)
};

static double now_sec() {
  return std::chrono::duration<double>(
      std::chrono::steady_clock::now().time_since_epoch()).count();
}

static double median(std::vector<double> v) {
  if (v.empty()) return 0.0;
  std::sort(v.begin(), v.end());
  size_t n = v.size();
  return (n % 2) ? v[n/2] : 0.5 * (v[n/2 - 1] + v[n/2]);
}

// size_bytes is the TOTAL payload the op transfers ("count" in rccl-tests
// terminology, scaled to the ncclDataType). For AG and RS we match the
// convention that size_bytes is the per-rank buffer size (so full buffer =
// size_bytes * world for AG send / RS recv).
static BenchResult bench_op(
    ncclComm_t comm, hipStream_t stream,
    const char* op, size_t size_bytes,
    int iters, int warmup)
{
  // Cap the per-rank alloc so we don't blow out HBM. Each rank needs send+recv
  // of up to size_bytes and size_bytes*world for the gather/scatter cases.
  size_t per_rank = size_bytes;
  size_t full     = size_bytes * g_world;

  // Truncate to at least 1 element of int8
  if (per_rank == 0) per_rank = 1;
  if (full == 0)     full = 1;

  char *send = nullptr, *recv = nullptr;
  HIPCHECK(hipMalloc(&send, full));
  HIPCHECK(hipMalloc(&recv, full));
  HIPCHECK(hipMemsetAsync(send, 0, full, stream));
  HIPCHECK(hipStreamSynchronize(stream));

  auto launch = [&](void) {
    if      (std::string(op) == "ag")
      NCCLCHECK(ncclAllGather(send, recv, per_rank, ncclInt8, comm, stream));
    else if (std::string(op) == "rs")
      NCCLCHECK(ncclReduceScatter(send, recv, per_rank, ncclInt8, ncclSum, comm, stream));
    else /* ar */
      NCCLCHECK(ncclAllReduce(send, recv, per_rank, ncclInt8, ncclSum, comm, stream));
  };

  // Warmup
  for (int i = 0; i < warmup; ++i) launch();
  HIPCHECK(hipStreamSynchronize(stream));

  // Time: 3 outer trials, each does `iters` ops. Take median per-op of all trials.
  std::vector<double> per_op_times;
  per_op_times.reserve(3);
  double best = 1e18;
  for (int trial = 0; trial < 3; ++trial) {
    HIPCHECK(hipStreamSynchronize(stream));
    double t0 = now_sec();
    for (int i = 0; i < iters; ++i) launch();
    HIPCHECK(hipStreamSynchronize(stream));
    double t1 = now_sec();
    double per_op = (t1 - t0) / iters;
    per_op_times.push_back(per_op);
    best = std::min(best, per_op);
  }

  HIPCHECK(hipFree(send));
  HIPCHECK(hipFree(recv));

  BenchResult r{};
  r.bytes = per_rank;
  r.per_op_ms = median(per_op_times) * 1000.0;
  r.best_per_op_ms = best * 1000.0;
  // Algorithmic bandwidth convention from rccl-tests:
  //   AG: payload transferred into a rank = (world-1) * per_rank
  //   RS: payload transferred out of a rank = (world-1) * per_rank
  //   AR: payload on the wire per rank     = 2 * (world-1)/world * per_rank
  double gb = 0;
  if (std::string(op) == "ag" || std::string(op) == "rs") {
    gb = double(per_rank) * (g_world - 1);
  } else {
    gb = 2.0 * double(per_rank) * (g_world - 1) / g_world;
  }
  r.bw_algbw_gbps = gb / (best * 1e9);
  return r;
}

// ---------- Main -------------------------------------------------------
int main(int argc, char** argv) {
  g_rank  = env_int("GLOBAL_RANK", -1);
  g_world = env_int("WORLD_SIZE", -1);
  int local_rank = env_int("LOCAL_RANK", -1);
  std::string uid_file = env_str("UID_FILE", "/tmp/anp_repro_uid");

  if (g_rank < 0 || g_world < 0 || local_rank < 0) {
    fprintf(stderr, "Missing env: need GLOBAL_RANK, WORLD_SIZE, LOCAL_RANK. Got "
                    "GLOBAL_RANK=%d WORLD_SIZE=%d LOCAL_RANK=%d\n",
            g_rank, g_world, local_rank);
    return 4;
  }

  // Bind to the local GPU for this rank.
  HIPCHECK(hipSetDevice(local_rank));

  // Derive NODE_RANK / LOCAL_WORLD from globals. We assume a uniform mesh:
  //   LOCAL_WORLD = GPUs per node (typically 8).
  //   NODE_RANK   = g_rank / LOCAL_WORLD  (i.e. which node this rank lives on).
  //
  // Given 2 nodes x 8 GPUs:
  //   comm_all : 16-rank (all GPUs)   — "global" ring for reference
  //   comm_ici :  8-rank, same node    — intra-node NVLink comm
  //   comm_dcn :  2-rank, across nodes — single cross-node pair per GPU,
  //                                       matches JAX/XLA FSDP DCN=2 pattern
  int local_world = env_int("LOCAL_WORLD", 8);
  if (local_world <= 0 || g_world % local_world != 0) {
    fprintf(stderr, "[rank %d] LOCAL_WORLD=%d doesn't divide WORLD_SIZE=%d\n",
            g_rank, local_world, g_world);
    return 5;
  }
  int node_rank    = g_rank / local_world;     // 0 or 1
  int rank_in_node = g_rank % local_world;     // 0..7
  int nnodes       = g_world / local_world;    // 2

  // ---- Bootstrap three uniqueIds via shared files ----
  // comm_all uses a single UID written by g_rank=0.
  // comm_ici uses one UID per node, written by the node's rank_in_node=0.
  // comm_dcn uses one UID per rank_in_node slot, written by the node 0 rank.
  ncclUniqueId id_all, id_ici, id_dcn;
  std::string uid_all  = uid_file;                                       // existing
  std::string uid_ici  = uid_file + ".ici.node" + std::to_string(node_rank);
  std::string uid_dcn  = uid_file + ".dcn.slot" + std::to_string(rank_in_node);

  if (g_rank == 0) { NCCLCHECK(ncclGetUniqueId(&id_all)); write_uid(uid_all, id_all); } else { read_uid(uid_all, id_all); }
  if (rank_in_node == 0) { NCCLCHECK(ncclGetUniqueId(&id_ici)); write_uid(uid_ici, id_ici); } else { read_uid(uid_ici, id_ici); }
  if (node_rank == 0)    { NCCLCHECK(ncclGetUniqueId(&id_dcn)); write_uid(uid_dcn, id_dcn); } else { read_uid(uid_dcn, id_dcn); }

  ncclComm_t comm_all, comm_ici, comm_dcn;
  NCCLCHECK(ncclCommInitRank(&comm_all, g_world,     id_all, g_rank));
  NCCLCHECK(ncclCommInitRank(&comm_ici, local_world, id_ici, rank_in_node));
  NCCLCHECK(ncclCommInitRank(&comm_dcn, nnodes,      id_dcn, node_rank));

  hipStream_t stream;
  HIPCHECK(hipStreamCreate(&stream));

  // Barrier + UID cleanup via a tiny AllReduce on comm_all.
  {
    int *one_dev = nullptr;
    HIPCHECK(hipMalloc(&one_dev, sizeof(int)));
    int one = 1;
    HIPCHECK(hipMemcpyAsync(one_dev, &one, sizeof(int), hipMemcpyHostToDevice, stream));
    NCCLCHECK(ncclAllReduce(one_dev, one_dev, 1, ncclInt32, ncclSum, comm_all, stream));
    HIPCHECK(hipStreamSynchronize(stream));
    HIPCHECK(hipFree(one_dev));
    if (g_rank == 0)        std::remove(uid_all.c_str());
    if (rank_in_node == 0)  std::remove(uid_ici.c_str());
    if (node_rank == 0)     std::remove(uid_dcn.c_str());
  }

  // Parse sweep plan.
  std::vector<size_t> sizes = parse_sizes(env_str("REPRO_SIZES"));
  int iters  = env_int("REPRO_ITERS",  30);
  int warmup = env_int("REPRO_WARMUP", 5);
  std::string ops_env = env_str("REPRO_OPS", "ag,rs,ar");
  std::vector<std::string> ops;
  {
    std::stringstream ss(ops_env);
    std::string t;
    while (std::getline(ss, t, ',')) if (!t.empty()) ops.push_back(t);
  }

  if (g_rank == 0) {
    const char* plugin = std::getenv("NCCL_NET_PLUGIN");
    printf("#\n");
    printf("# ANP_REPRO  world=%d iters=%d warmup=%d\n", g_world, iters, warmup);
    printf("# NCCL_NET_PLUGIN=%s\n", plugin ? plugin : "(unset; using RCCL built-in IB)");
    printf("# scope='dcn' -> 2-rank cross-node pair (matches XLA DCN-FSDP-2)\n");
    printf("# scope='ici' -> 8-rank intra-node\n");
    printf("# scope='all' -> 16-rank global ring\n");
    printf("#\n");
    printf("# %-4s %-4s %14s %14s %10s %10s %10s\n",
           "scope", "op", "per_rank_bytes", "total_bytes", "per_op_ms", "best_ms", "algbw_GBs");
    fflush(stdout);
  }

  // Which comms to run on — default exercises all three, but the DCN one is
  // the apples-to-apples analogue of MaxText's per-layer FSDP cross-node op.
  std::string scopes_env = env_str("REPRO_SCOPES", "dcn,ici,all");
  std::vector<std::string> scopes;
  { std::stringstream ss(scopes_env); std::string t;
    while (std::getline(ss, t, ',')) if (!t.empty()) scopes.push_back(t); }

  auto run_sweep = [&](const std::string& scope_name, ncclComm_t c, int comm_world) {
    // Warm this comm.
    int *junk = nullptr;
    HIPCHECK(hipMalloc(&junk, 4));
    HIPCHECK(hipMemsetAsync(junk, 0, 4, stream));
    NCCLCHECK(ncclAllReduce(junk, junk, 1, ncclInt32, ncclSum, c, stream));
    HIPCHECK(hipStreamSynchronize(stream));
    HIPCHECK(hipFree(junk));

    int saved_world = g_world; g_world = comm_world;
    for (const auto& op : ops) {
      for (size_t sz : sizes) {
        size_t full = sz * comm_world;
        if (full > (4ULL * 1024 * 1024 * 1024)) continue;
        BenchResult r = bench_op(c, stream, op.c_str(), sz, iters, warmup);
        if (g_rank == 0) {
          printf("  %-4s %-4s %14zu %14zu %10.3f %10.3f %10.2f\n",
                 scope_name.c_str(), op.c_str(),
                 r.bytes, r.bytes * comm_world,
                 r.per_op_ms, r.best_per_op_ms, r.bw_algbw_gbps);
          fflush(stdout);
        }
      }
    }
    g_world = saved_world;
  };

  for (const auto& scope : scopes) {
    if      (scope == "dcn") run_sweep("dcn", comm_dcn, nnodes);
    else if (scope == "ici") run_sweep("ici", comm_ici, local_world);
    else if (scope == "all") run_sweep("all", comm_all, g_world);
    else if (g_rank == 0) fprintf(stderr, "Unknown scope '%s' (expected dcn/ici/all)\n", scope.c_str());
  }

  // Final barrier before teardown.
  {
    int *junk = nullptr;
    HIPCHECK(hipMalloc(&junk, 4));
    NCCLCHECK(ncclAllReduce(junk, junk, 1, ncclInt32, ncclSum, comm_all, stream));
    HIPCHECK(hipStreamSynchronize(stream));
    HIPCHECK(hipFree(junk));
  }

  NCCLCHECK(ncclCommDestroy(comm_dcn));
  NCCLCHECK(ncclCommDestroy(comm_ici));
  NCCLCHECK(ncclCommDestroy(comm_all));
  HIPCHECK(hipStreamDestroy(stream));
  return 0;
}
