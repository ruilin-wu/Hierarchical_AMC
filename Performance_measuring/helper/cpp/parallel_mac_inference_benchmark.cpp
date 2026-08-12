// OpenMP classifier-MAC benchmark for the accelerated Table-I comparison.
//
// Parallelism is only across scalar MAC operations.  Auto-vectorization, FMA
// contraction, BLAS, MKL, oneDNN, and LibTorch are not used.

#include <chrono>
#include <cstdint>
#include <vector>

#include <omp.h>

namespace {

volatile float parallel_sink = 0.0f;

inline void one_mac(float& value, float multiplier, float addend) {
  value = value * multiplier + addend;
}

float run_scalar_chunk(std::uint64_t operations, std::uint64_t seed) {
  float a0 = 0.101f + static_cast<float>((seed >> 0) & 255U) * 1.0e-7f;
  float a1 = 0.203f + static_cast<float>((seed >> 8) & 255U) * 1.0e-7f;
  float a2 = 0.307f + static_cast<float>((seed >> 16) & 255U) * 1.0e-7f;
  float a3 = 0.409f + static_cast<float>((seed >> 24) & 255U) * 1.0e-7f;
  float a4 = 0.503f + static_cast<float>((seed >> 32) & 255U) * 1.0e-7f;
  float a5 = 0.607f + static_cast<float>((seed >> 40) & 255U) * 1.0e-7f;
  float a6 = 0.709f + static_cast<float>((seed >> 48) & 255U) * 1.0e-7f;
  float a7 = 0.811f + static_cast<float>((seed >> 56) & 255U) * 1.0e-7f;

  const std::uint64_t blocks = operations / 8U;
  for (std::uint64_t i = 0; i < blocks; ++i) {
    one_mac(a0, 0.99999994f, 1.01e-7f);
    one_mac(a1, 0.99999988f, 1.03e-7f);
    one_mac(a2, 0.99999982f, 1.07e-7f);
    one_mac(a3, 0.99999976f, 1.09e-7f);
    one_mac(a4, 0.99999970f, 1.13e-7f);
    one_mac(a5, 0.99999964f, 1.17e-7f);
    one_mac(a6, 0.99999958f, 1.19e-7f);
    one_mac(a7, 0.99999952f, 1.23e-7f);
  }
  switch (operations & 7U) {
    case 7: one_mac(a6, 0.99999958f, 1.19e-7f); [[fallthrough]];
    case 6: one_mac(a5, 0.99999964f, 1.17e-7f); [[fallthrough]];
    case 5: one_mac(a4, 0.99999970f, 1.13e-7f); [[fallthrough]];
    case 4: one_mac(a3, 0.99999976f, 1.09e-7f); [[fallthrough]];
    case 3: one_mac(a2, 0.99999982f, 1.07e-7f); [[fallthrough]];
    case 2: one_mac(a1, 0.99999988f, 1.03e-7f); [[fallthrough]];
    case 1: one_mac(a0, 0.99999994f, 1.01e-7f); [[fallthrough]];
    default: break;
  }
  return a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7;
}

}  // namespace

extern "C" double parallel_mac_wall_seconds(std::uint64_t operations_per_inference,
                                             std::uint64_t logical_inferences,
                                             std::uint64_t seed,
                                             int requested_threads,
                                             float* checksum) {
  const int threads = requested_threads > 0 ? requested_threads : 1;
  omp_set_dynamic(0);
  std::vector<float> partial(static_cast<std::size_t>(threads), 0.0f);
  float total = 0.0f;

  const auto start = std::chrono::steady_clock::now();
  for (std::uint64_t inference = 0; inference < logical_inferences; ++inference) {
#pragma omp parallel num_threads(threads)
    {
      const int tid = omp_get_thread_num();
      const int team = omp_get_num_threads();
      const std::uint64_t base = operations_per_inference / static_cast<std::uint64_t>(team);
      const std::uint64_t remainder = operations_per_inference % static_cast<std::uint64_t>(team);
      const std::uint64_t local_ops = base + (static_cast<std::uint64_t>(tid) < remainder ? 1U : 0U);
      partial[static_cast<std::size_t>(tid)] = run_scalar_chunk(
          local_ops, seed + inference * 131U + static_cast<std::uint64_t>(tid));
    }
    for (int tid = 0; tid < threads; ++tid) total += partial[static_cast<std::size_t>(tid)];
  }
  const auto stop = std::chrono::steady_clock::now();

  parallel_sink = total;
  if (checksum != nullptr) *checksum = total;
  return std::chrono::duration<double>(stop - start).count();
}

extern "C" float parallel_mac_sink() {
  return parallel_sink;
}
