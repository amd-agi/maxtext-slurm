// Minimal rocSHMEM device-side barrier_all repro.
//
// Intended launch shape:
//   - one process per local GPU on each node
//   - same LOCAL_RANK across nodes forms one rocSHMEM world
//   - NODE_RANK is used as rocSHMEM PE rank, NNODES as rocSHMEM world size
//
// The only GPU-side operation under test is rocshmem_barrier_all().

#include <hip/hip_runtime.h>
#include <rocshmem/rocshmem.hpp>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <unistd.h>

#define CHECK_HIP(cmd)                                                                            \
  do {                                                                                            \
    hipError_t _err = (cmd);                                                                      \
    if (_err != hipSuccess) {                                                                     \
      std::fprintf(stderr, "[RS-BARRIER-H] HIP error %s:%d: %s (%d)\n", __FILE__, __LINE__,       \
                   hipGetErrorString(_err), static_cast<int>(_err));                              \
      std::fflush(stderr);                                                                        \
      std::exit(2);                                                                               \
    }                                                                                             \
  } while (0)

#define CHECK_RS(cmd)                                                                             \
  do {                                                                                            \
    int _err = (cmd);                                                                             \
    if (_err != 0) {                                                                              \
      std::fprintf(stderr, "[RS-BARRIER-H] rocSHMEM error %s:%d: rc=%d\n", __FILE__, __LINE__,    \
                   _err);                                                                         \
      std::fflush(stderr);                                                                        \
      std::exit(3);                                                                               \
    }                                                                                             \
  } while (0)

static int getenv_int(const char *name, int default_value) {
  const char *raw = std::getenv(name);
  if (raw == nullptr || raw[0] == '\0') {
    return default_value;
  }
  return std::atoi(raw);
}

static std::string getenv_str(const char *name, const char *default_value) {
  const char *raw = std::getenv(name);
  return (raw == nullptr || raw[0] == '\0') ? std::string(default_value) : std::string(raw);
}

static std::string hex_encode(const uint8_t *data, size_t len) {
  static constexpr char kHex[] = "0123456789abcdef";
  std::string out;
  out.reserve(len * 2);
  for (size_t i = 0; i < len; ++i) {
    out.push_back(kHex[data[i] >> 4]);
    out.push_back(kHex[data[i] & 0x0f]);
  }
  return out;
}

static uint8_t hex_digit(char c) {
  if (c >= '0' && c <= '9') return static_cast<uint8_t>(c - '0');
  if (c >= 'a' && c <= 'f') return static_cast<uint8_t>(10 + c - 'a');
  if (c >= 'A' && c <= 'F') return static_cast<uint8_t>(10 + c - 'A');
  throw std::runtime_error("invalid hex digit");
}

static std::vector<uint8_t> hex_decode(const std::string &hex) {
  if (hex.size() % 2 != 0) {
    throw std::runtime_error("odd hex length");
  }
  std::vector<uint8_t> out(hex.size() / 2);
  for (size_t i = 0; i < out.size(); ++i) {
    out[i] = static_cast<uint8_t>((hex_digit(hex[2 * i]) << 4) | hex_digit(hex[2 * i + 1]));
  }
  return out;
}

static void write_text_atomic(const std::string &path, const std::string &text) {
  const std::string tmp = path + ".tmp." + std::to_string(::getpid());
  {
    std::ofstream f(tmp, std::ios::out | std::ios::trunc);
    if (!f) {
      throw std::runtime_error("failed to open temp file for write: " + tmp);
    }
    f << text << "\n";
  }
  if (std::rename(tmp.c_str(), path.c_str()) != 0) {
    throw std::runtime_error("failed to rename temp file to " + path);
  }
}

static std::string read_text_wait(const std::string &path, int timeout_s) {
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(timeout_s);
  while (std::chrono::steady_clock::now() < deadline) {
    std::ifstream f(path);
    if (f) {
      std::stringstream ss;
      ss << f.rdbuf();
      std::string text = ss.str();
      while (!text.empty() && (text.back() == '\n' || text.back() == '\r' || text.back() == ' ')) {
        text.pop_back();
      }
      if (!text.empty()) {
        return text;
      }
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }
  throw std::runtime_error("timed out waiting for " + path);
}

__global__ void barrier_all_kernel(int global_rank, int local_rank, int my_pe, int n_pes, int iters,
                                   int *status) {
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    rocshmem::rocshmem_ctx_t ctx = rocshmem::ROCSHMEM_CTX_DEFAULT;
    printf("[RS-BARRIER-K g=%d l=%d pe=%d/%d] ENTER ctx=%p team=%p status=%p iters=%d\n",
           global_rank, local_rank, my_pe, n_pes, ctx.ctx_opaque, ctx.team_opaque, status, iters);
    for (int i = 0; i < iters; ++i) {
      printf("[RS-BARRIER-K g=%d l=%d pe=%d/%d] iter=%d PRE  rocshmem_barrier_all\n",
             global_rank, local_rank, my_pe, n_pes, i);
      rocshmem::rocshmem_barrier_all();
      printf("[RS-BARRIER-K g=%d l=%d pe=%d/%d] iter=%d POST rocshmem_barrier_all\n",
             global_rank, local_rank, my_pe, n_pes, i);
    }
    status[0] = 0x51A7;
    printf("[RS-BARRIER-K g=%d l=%d pe=%d/%d] EXIT status=0x%x\n", global_rank, local_rank,
           my_pe, n_pes, status[0]);
  }
}

int main() {
  const int local_rank = getenv_int("LOCAL_RANK", 0);
  const int global_rank = getenv_int("GLOBAL_RANK", -1);
  const int node_rank = getenv_int("NODE_RANK", 0);
  const int nnodes = getenv_int("NNODES", 1);
  const int iters = getenv_int("ROCSHMEM_BARRIER_REPRO_ITERS", 1);
  const int uid_timeout_s = getenv_int("ROCSHMEM_BARRIER_REPRO_UID_TIMEOUT", 120);
  const bool host_barrier = getenv_int("ROCSHMEM_BARRIER_REPRO_HOST_BARRIER", 0) != 0;
  const std::string exchange_dir = getenv_str("ROCSHMEM_BARRIER_REPRO_DIR", "/tmp");

  CHECK_HIP(hipSetDevice(local_rank));

  std::fprintf(stderr,
               "[RS-BARRIER-H g=%d l=%d node=%d/%d] START device=%d dir=%s backend=%s provider=%s\n",
               global_rank, local_rank, node_rank, nnodes, local_rank, exchange_dir.c_str(),
               getenv_str("ROCSHMEM_BACKEND", "<unset>").c_str(),
               getenv_str("ROCSHMEM_GDA_PROVIDER", "<unset>").c_str());
  std::fflush(stderr);

  rocshmem::rocshmem_uniqueid_t uid;
  std::memset(&uid, 0, sizeof(uid));
  const std::string uid_path = exchange_dir + "/uid_local_rank_" + std::to_string(local_rank) + ".hex";

  if (node_rank == 0) {
    CHECK_RS(rocshmem::rocshmem_get_uniqueid(&uid));
    write_text_atomic(uid_path, hex_encode(uid.data(), uid.size()));
    std::fprintf(stderr, "[RS-BARRIER-H g=%d l=%d] wrote root unique id: %s\n", global_rank,
                 local_rank, uid_path.c_str());
  } else {
    const std::string uid_hex = read_text_wait(uid_path, uid_timeout_s);
    const auto uid_bytes = hex_decode(uid_hex);
    if (uid_bytes.size() != uid.size()) {
      throw std::runtime_error("unexpected uid byte count in " + uid_path);
    }
    std::memcpy(uid.data(), uid_bytes.data(), uid.size());
    std::fprintf(stderr, "[RS-BARRIER-H g=%d l=%d] read root unique id: %s\n", global_rank,
                 local_rank, uid_path.c_str());
  }
  std::fflush(stderr);

  rocshmem::rocshmem_init_attr_t attr;
  std::memset(&attr, 0, sizeof(attr));
  CHECK_RS(rocshmem::rocshmem_set_attr_uniqueid_args(node_rank, nnodes, &uid, &attr));
  CHECK_RS(rocshmem::rocshmem_init_attr(rocshmem::ROCSHMEM_INIT_WITH_UNIQUEID, &attr));

  const int my_pe = rocshmem::rocshmem_my_pe();
  const int n_pes = rocshmem::rocshmem_n_pes();
  void *device_ctx = rocshmem::rocshmem_get_device_ctx();
  std::fprintf(stderr, "[RS-BARRIER-H g=%d l=%d] init OK my_pe=%d n_pes=%d device_ctx=%p\n",
               global_rank, local_rank, my_pe, n_pes, device_ctx);
  std::fflush(stderr);

  if (my_pe != node_rank || n_pes != nnodes) {
    std::fprintf(stderr,
                 "[RS-BARRIER-H g=%d l=%d] unexpected PE metadata: got my_pe=%d n_pes=%d, "
                 "expected my_pe=%d n_pes=%d\n",
                 global_rank, local_rank, my_pe, n_pes, node_rank, nnodes);
    std::fflush(stderr);
    return 4;
  }

  if (host_barrier) {
    std::fprintf(stderr, "[RS-BARRIER-H g=%d l=%d] PRE  host rocshmem_barrier_all\n", global_rank,
                 local_rank);
    std::fflush(stderr);
    rocshmem::rocshmem_barrier_all();
    std::fprintf(stderr, "[RS-BARRIER-H g=%d l=%d] POST host rocshmem_barrier_all\n", global_rank,
                 local_rank);
    std::fflush(stderr);
  }

  int *status = nullptr;
  CHECK_HIP(hipMalloc(&status, sizeof(int)));
  CHECK_HIP(hipMemset(status, 0, sizeof(int)));

  std::fprintf(stderr, "[RS-BARRIER-H g=%d l=%d] launching device barrier kernel\n", global_rank,
               local_rank);
  std::fflush(stderr);
  barrier_all_kernel<<<1, 128>>>(global_rank, local_rank, my_pe, n_pes, iters, status);
  hipError_t launch_err = hipGetLastError();
  std::fprintf(stderr, "[RS-BARRIER-H g=%d l=%d] launch returned %d (%s)\n", global_rank,
               local_rank, static_cast<int>(launch_err), hipGetErrorString(launch_err));
  std::fflush(stderr);
  if (launch_err != hipSuccess) {
    return 5;
  }

  hipError_t sync_err = hipDeviceSynchronize();
  std::fprintf(stderr, "[RS-BARRIER-H g=%d l=%d] hipDeviceSynchronize returned %d (%s)\n",
               global_rank, local_rank, static_cast<int>(sync_err), hipGetErrorString(sync_err));
  std::fflush(stderr);
  if (sync_err != hipSuccess) {
    return 6;
  }

  int host_status = 0;
  CHECK_HIP(hipMemcpy(&host_status, status, sizeof(int), hipMemcpyDeviceToHost));
  std::fprintf(stderr, "[RS-BARRIER-H g=%d l=%d] status=0x%x\n", global_rank, local_rank,
               host_status);
  std::fflush(stderr);

  CHECK_HIP(hipFree(status));
  rocshmem::rocshmem_finalize();
  std::fprintf(stderr, "[RS-BARRIER-H g=%d l=%d] PASS\n", global_rank, local_rank);
  std::fflush(stderr);
  return host_status == 0x51A7 ? 0 : 7;
}
