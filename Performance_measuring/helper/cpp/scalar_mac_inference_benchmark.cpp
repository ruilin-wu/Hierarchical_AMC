// Controlled single-thread scalar-MAC benchmark for Table-I methods 1--5.
//
// This deliberately does not use LibTorch, oneDNN, BLAS, OpenMP, SIMD, or
// batched kernels.  Each loop update is one scalar multiply followed by one
// scalar add, corresponding to one MAC in the operation-count notebook.

#include <cstdint>
#include <ctime>

namespace {

volatile float benchmark_sink = 0.0f;

inline void one_mac(float& value, float multiplier, float addend) {
  value = value * multiplier + addend;
}

}  // namespace

extern "C" double scalar_mac_seconds(std::uint64_t operations,
                                     std::uint64_t seed,
                                     float* checksum) {
  // Eight independent dependency chains reduce pipeline-stall bias while all
  // instructions remain scalar.  Constants keep the recurrences finite.
  float a0 = 0.101f + static_cast<float>((seed >> 0) & 255U) * 1.0e-7f;
  float a1 = 0.203f + static_cast<float>((seed >> 8) & 255U) * 1.0e-7f;
  float a2 = 0.307f + static_cast<float>((seed >> 16) & 255U) * 1.0e-7f;
  float a3 = 0.409f + static_cast<float>((seed >> 24) & 255U) * 1.0e-7f;
  float a4 = 0.503f + static_cast<float>((seed >> 32) & 255U) * 1.0e-7f;
  float a5 = 0.607f + static_cast<float>((seed >> 40) & 255U) * 1.0e-7f;
  float a6 = 0.709f + static_cast<float>((seed >> 48) & 255U) * 1.0e-7f;
  float a7 = 0.811f + static_cast<float>((seed >> 56) & 255U) * 1.0e-7f;

  // Thread CPU time excludes scheduler pre-emption by unrelated processes.
  // This is the appropriate clock for comparing operation-count scaling on a
  // shared workstation; the notebook separately reports end-to-end wall time.
  timespec start{};
  timespec stop{};
  clock_gettime(CLOCK_THREAD_CPUTIME_ID, &start);
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

  clock_gettime(CLOCK_THREAD_CPUTIME_ID, &stop);
  const float result = a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7;
  benchmark_sink = result;
  if (checksum != nullptr) *checksum = result;
  return static_cast<double>(stop.tv_sec - start.tv_sec) +
         1.0e-9 * static_cast<double>(stop.tv_nsec - start.tv_nsec);
}

extern "C" float scalar_mac_sink() {
  return benchmark_sink;
}
