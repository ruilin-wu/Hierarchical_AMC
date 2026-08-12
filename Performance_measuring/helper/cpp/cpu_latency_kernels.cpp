// CPU kernels shared by latency-comparison notebooks 1--5.
//
// Python only orchestrates the stages and renders tables.  Every timed signal
// processing kernel in this file executes through ATen's C++ CPU backend.

#include <torch/extension.h>

#include <ATen/Parallel.h>
#include <ATen/ops/abs.h>
#include <ATen/ops/arange.h>
#include <ATen/ops/complex.h>
#include <ATen/ops/constant_pad_nd.h>
#include <ATen/ops/conv1d.h>
#include <ATen/ops/fft_fft.h>
#include <ATen/ops/fft_ifft.h>
#include <ATen/ops/mean.h>
#include <ATen/ops/roll.h>
#include <ATen/ops/stack.h>
#include <ATen/ops/topk.h>
#include <c10/util/complex.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <complex>
#include <cstdint>
#include <fstream>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace py = pybind11;
using Tensor = at::Tensor;
using Complex = c10::complex<float>;

namespace {

constexpr int64_t kSignalLength = 32768;
constexpr float kPi = 3.14159265358979323846f;

Tensor fftshift(const Tensor& x, int64_t dim) {
  return at::roll(x, {x.size(dim) / 2}, {dim});
}

Tensor ifftshift(const Tensor& x, int64_t dim) {
  return at::roll(x, {-(x.size(dim) / 2)}, {dim});
}

Tensor fft_centered(const Tensor& x, int64_t dim) {
  return fftshift(at::fft_fft(x, c10::nullopt, dim, c10::nullopt), dim);
}

Tensor ifft_centered(const Tensor& x, int64_t dim) {
  return at::fft_ifft(ifftshift(x, dim), c10::nullopt, dim, c10::nullopt);
}

Tensor hann_window(int64_t n, at::ScalarType dtype = at::kFloat) {
  auto opts = at::TensorOptions().dtype(dtype).device(at::kCPU);
  if (n <= 1) return at::ones({n}, opts);
  Tensor idx = at::arange(n, opts);
  return 0.5 - 0.5 * at::cos((2.0 * M_PI / static_cast<double>(n - 1)) * idx);
}

Tensor read_tim_impl(const std::string& path, int64_t samples) {
  std::ifstream stream(path, std::ios::binary);
  if (!stream) throw std::runtime_error("cannot open .tim file: " + path);
  stream.seekg(0, std::ios::end);
  const std::streamoff bytes = stream.tellg();
  stream.seekg(0, std::ios::beg);
  const int64_t floats = static_cast<int64_t>(bytes / sizeof(float));
  const int64_t need = 2 + 2 * samples;
  if (floats < need) {
    throw std::runtime_error(path + ": fewer samples than requested");
  }
  std::vector<float> raw(static_cast<size_t>(need));
  stream.read(reinterpret_cast<char*>(raw.data()), need * sizeof(float));
  if (!stream) throw std::runtime_error("short read from .tim file: " + path);
  std::vector<Complex> iq(static_cast<size_t>(samples));
  for (int64_t i = 0; i < samples; ++i) {
    iq[static_cast<size_t>(i)] = Complex(raw[2 + 2 * i], raw[3 + 2 * i]);
  }
  return at::from_blob(iq.data(), {samples}, at::TensorOptions().dtype(at::kComplexFloat)).clone();
}

Tensor boi_shift_impl(const Tensor& input, bool unit_power) {
  Tensor x = input.to(at::kCPU, at::kComplexFloat).contiguous().flatten();
  const int64_t n = x.numel();
  Tensor spectrum = fft_centered(x, 0);
  Tensor power = at::abs(spectrum).square() / static_cast<double>(n * n);

  int64_t width = std::max<int64_t>(1, static_cast<int64_t>(std::floor(n * 0.002)));
  if ((width & 1) == 0) ++width;
  Tensor kernel = at::ones({1, 1, width}, power.options()) / static_cast<double>(width);
  Tensor smooth = at::conv1d(power.view({1, 1, n}), kernel, c10::nullopt,
                             {1}, {width / 2}, {1}, 1).flatten().contiguous();
  auto sm = smooth.accessor<float, 1>();
  const int64_t half = width / 2;
  float min_power = std::numeric_limits<float>::infinity();
  int64_t peak = half;
  float peak_power = -1.0f;
  for (int64_t i = half; i < n - half; ++i) {
    const float v = sm[i];
    min_power = std::min(min_power, v);
    if (v > peak_power) { peak_power = v; peak = i; }
  }
  const float detection = min_power * std::pow(10.0f, 5.0f / 10.0f);
  int64_t left = peak;
  int64_t right = peak;
  while (left > 0 && sm[left - 1] >= detection) --left;
  while (right + 1 < n && sm[right + 1] >= detection) ++right;

  // The source BOI routine uses a 1.5 dB boundary threshold and limits the
  // refinement to eight smoothing widths.
  const float boundary = min_power * std::pow(10.0f, 1.5f / 10.0f);
  const int64_t max_expand = 8 * width;
  int64_t count = 0;
  while (left > 0 && count++ < max_expand && sm[left - 1] >= boundary) --left;
  count = 0;
  while (right + 1 < n && count++ < max_expand && sm[right + 1] >= boundary) ++right;

  // Adaptive guard.  Keeping the filtered spectrum in the centered layout
  // lets one roll perform BPF and CFO centering together.
  const int64_t span = std::max<int64_t>(8, right - left + 1);
  const int64_t guard = std::max<int64_t>(2, static_cast<int64_t>(std::ceil(0.20 * span)));
  left = std::max<int64_t>(0, left - guard);
  right = std::min<int64_t>(n - 1, right + guard);
  const int64_t center = (left + right) / 2;
  Tensor masked = at::zeros_like(spectrum);
  masked.narrow(0, left, right - left + 1).copy_(spectrum.narrow(0, left, right - left + 1));
  masked = at::roll(masked, {n / 2 - center}, {0});
  Tensor output = ifft_centered(masked, 0).to(at::kComplexFloat);
  if (unit_power) {
    Tensor rms = at::sqrt(at::mean(at::abs(output).square())).clamp_min(1.0e-12);
    output = output / rms;
  }
  return output.contiguous();
}

Tensor cheb64() {
  static const std::array<float, 64> values = {
    5.2122791557e-03f, 7.8619391339e-03f, 1.3513280645e-02f, 2.1555806250e-02f,
    3.2568317431e-02f, 4.7174045502e-02f, 6.6024238277e-02f, 8.9777690996e-02f,
    1.1907665533e-01f, 1.5451980153e-01f, 1.9663313729e-01f, 2.4583998873e-01f,
    3.0243131010e-01f, 3.6653769679e-01f, 4.3810452179e-01f, 5.1687159065e-01f,
    6.0235861160e-01f, 6.9385760579e-01f, 7.9043314275e-01f, 8.9093098630e-01f,
    9.9399538959e-01f, 1.0980948999e+00f, 1.2015561428e+00f, 1.3026046718e+00f,
    1.3994116109e+00f, 1.4901445084e+00f, 1.5730205719e+00f, 1.6463602853e+00f,
    1.7086393325e+00f, 1.7585367630e+00f, 1.7949774531e+00f, 1.8171671185e+00f,
    1.8246184272e+00f, 1.8171671185e+00f, 1.7949774531e+00f, 1.7585367630e+00f,
    1.7086393325e+00f, 1.6463602853e+00f, 1.5730205719e+00f, 1.4901445084e+00f,
    1.3994116109e+00f, 1.3026046718e+00f, 1.2015561428e+00f, 1.0980948999e+00f,
    9.9399538959e-01f, 8.9093098630e-01f, 7.9043314275e-01f, 6.9385760579e-01f,
    6.0235861160e-01f, 5.1687159065e-01f, 4.3810452179e-01f, 3.6653769679e-01f,
    3.0243131010e-01f, 2.4583998873e-01f, 1.9663313729e-01f, 1.5451980153e-01f,
    1.1907665533e-01f, 8.9777690996e-02f, 6.6024238277e-02f, 4.7174045502e-02f,
    3.2568317431e-02f, 2.1555806250e-02f, 1.3513280645e-02f, 7.8619391339e-03f
  };
  return at::from_blob(const_cast<float*>(values.data()), {64}, at::TensorOptions().dtype(at::kFloat)).clone();
}

Tensor periodic_psd_lookup(const Tensor& query, const Tensor& psd) {
  // PSD bins are fftshifted uniform samples on [-0.5, 0.5).  Linear
  // interpolation matches the periodic source implementation closely while
  // avoiding materialising three copies of the PSD vector.
  const int64_t m = psd.numel();
  Tensor wrapped = at::remainder(query + 0.5, 1.0) - 0.5;
  Tensor pos = (wrapped + 0.5) * static_cast<double>(m);
  Tensor i0 = at::floor(pos).to(at::kLong).remainder(m);
  Tensor i1 = (i0 + 1).remainder(m);
  Tensor frac = pos - at::floor(pos);
  return psd.index({i0}) * (1.0 - frac) + psd.index({i1}) * frac;
}

Tensor ssca_checksum_impl(const Tensor& input, bool conjugate) {
  constexpr int64_t n = 32768;
  constexpr int64_t np = 64;
  constexpr int64_t needed = n + np;
  Tensor x0 = input.to(at::kCPU, at::kComplexFloat).contiguous().flatten();
  Tensor x = at::zeros({needed}, x0.options());
  const int64_t copy_n = std::min<int64_t>(x0.numel(), needed);
  x.narrow(0, 0, copy_n).copy_(x0.narrow(0, 0, copy_n));

  Tensor windows = x.unfold(0, np, 1).narrow(0, 0, n) * cheb64().view({1, np});
  Tensor xf = fftshift(at::fft_fft(windows, c10::nullopt, 1, c10::nullopt), 1);
  Tensor nidx = at::arange(n, at::TensorOptions().dtype(at::kFloat)).view({n, 1});
  Tensor j = at::arange(np, at::TensorOptions().dtype(at::kFloat));
  Tensor ju = at::remainder(j + np / 2, np).view({1, np});
  Tensor angle = (-2.0 * M_PI / static_cast<double>(np)) * nidx * ju;
  Tensor phase = at::complex(at::cos(angle), at::sin(angle));
  Tensor xt = xf * phase;
  Tensor center = x.narrow(0, np / 2, n).view({n, 1});
  Tensor cdp = conjugate ? xt * center : xt * at::conj(center);
  Tensor scf = fftshift(at::fft_fft(cdp, c10::nullopt, 0, c10::nullopt), 0) /
               static_cast<double>(n);

  // Welch PSD: nfft=512, hop=256, Hann window, exactly as the reference.
  constexpr int64_t pfft = 512;
  constexpr int64_t hop = 256;
  Tensor segments = x.narrow(0, 0, n).unfold(0, pfft, hop) * hann_window(pfft).view({1, pfft});
  Tensor ps = fftshift(at::fft_fft(segments, c10::nullopt, 1, c10::nullopt), 1);
  Tensor psd = at::mean(at::abs(ps).square(), 0) /
               at::sum(hann_window(pfft).square()).clamp_min(1.0e-15);
  psd = psd.clamp_min(1.0e-15);

  Tensor q = (at::arange(n, at::TensorOptions().dtype(at::kFloat)) - n / 2) /
             static_cast<double>(n);
  Tensor k = (at::arange(np, at::TensorOptions().dtype(at::kFloat)) - np / 2) /
             static_cast<double>(np);
  Tensor alpha = k.view({1, np}) + q.view({n, 1});
  Tensor freq = 0.5 * (k.view({1, np}) - q.view({n, 1}));
  Tensor p1 = periodic_psd_lookup(freq + 0.5 * alpha, psd);
  Tensor second_query = conjugate ? (0.5 * alpha - freq) : (freq - 0.5 * alpha);
  Tensor p2 = periodic_psd_lookup(second_query, psd);
  const double floor = std::max(1.0e-15, 1.0e-3 * psd.max().item<double>());
  Tensor valid = (p1 >= floor) & (p2 >= floor);
  Tensor denom = at::sqrt((p1 * p2).clamp_min(1.0e-15));
  Tensor coh = at::where(valid, scf / denom, at::zeros_like(scf));
  Tensor mag = at::abs(coh);
  coh = at::where(mag > 1.0, coh / mag.clamp_min(1.0e-15), coh);
  // A scalar keeps the optimiser from eliding work and avoids transferring a
  // 32768x64 complex result through the Python boundary.
  return at::stack({at::abs(scf).mean(), at::abs(coh).mean()});
}

Tensor fft_x4_checksum_impl(const Tensor& input) {
  Tensor x = input.to(at::kCPU, at::kComplexFloat).contiguous().flatten();
  Tensor x2 = x * x;
  Tensor x4 = x2 * x2;
  Tensor spec = fft_centered(x4 * hann_window(x.numel()), 0);
  Tensor mag = at::abs(spec);
  Tensor curve = 20.0 * at::log10(mag / mag.max().clamp_min(1.0e-12) + 1.0e-12);
  return curve.mean();
}

Tensor warp_td(const Tensor& z, double exponent) {
  Tensor mag = at::abs(z);
  Tensor scale = at::pow(mag.max().clamp_min(1.0e-12), exponent);
  return at::stack({at::real(z) / scale, at::imag(z) / scale}, 0).unsqueeze(0).to(at::kFloat);
}

Tensor warp_fd(const Tensor& z) {
  Tensor mag = at::abs(at::fft_fft(z, c10::nullopt, 0, c10::nullopt));
  Tensor scale = at::sqrt(mag.max().clamp_min(1.0e-12));
  return (mag / scale).view({1, 1, z.numel()}).to(at::kFloat);
}

std::vector<Tensor> snoap_features_impl(const Tensor& input) {
  Tensor usp = input.to(at::kCPU, at::kComplexFloat).contiguous().flatten();
  Tensor utp = usp / at::sqrt(at::mean(at::abs(usp).square())).clamp_min(1.0e-12);
  std::vector<Tensor> out;
  out.reserve(10);
  for (const Tensor& z : {utp, usp}) {
    Tensor z2 = z * z;
    Tensor z4 = z2 * z2;
    Tensor z6 = z4 * z2;
    out.push_back(warp_td(z2, 0.5));
    out.push_back(warp_fd(z2));
    out.push_back(warp_td(z4, 0.5));
    out.push_back(warp_fd(z4));
    out.push_back(warp_td(z6, 2.0 / 3.0));
  }
  return out;
}

std::tuple<Tensor, Tensor> lr55_features_impl(const Tensor& input) {
  Tensor x = input.to(at::kCPU, at::kComplexFloat).contiguous().flatten();
  Tensor x2 = x * x;
  Tensor x4 = x2 * x2;
  Tensor abs2 = at::abs(x).square().to(at::kComplexFloat);
  std::array<Tensor, 3> mags = {
    at::abs(fft_centered(x2, 0)),
    at::abs(fft_centered(x4, 0)),
    at::abs(fft_centered(abs2, 0)),
  };
  std::vector<Tensor> parts;
  parts.reserve(5);
  std::vector<Tensor> peak_indices;
  for (Tensor mag : mags) {
    mag = mag / mag.max().clamp_min(1.0e-12);
    auto top = at::topk(mag, 11, -1, true, true);
    parts.push_back(std::get<0>(top).to(at::kFloat));
    peak_indices.push_back(std::get<1>(top).narrow(0, 0, 1).to(at::kFloat));
  }
  // 22 stable distribution/shape statistics complete the LR55 vector.
  Tensor base = at::abs(x);
  std::vector<Tensor> stats;
  stats.reserve(22);
  for (int p = 1; p <= 11; ++p) {
    Tensor v = at::pow(base, static_cast<double>(p) / 4.0);
    stats.push_back(v.mean().to(at::kFloat));
    stats.push_back(v.std(false).to(at::kFloat));
  }
  parts.push_back(at::stack(stats));
  Tensor features = at::cat(parts).contiguous();
  Tensor aux = at::cat(peak_indices).contiguous();
  return {features, aux};
}

Tensor estimate_rs_cfo_impl(const Tensor& aux) {
  Tensor a = aux.to(at::kCPU, at::kFloat).contiguous();
  const float i2 = a[0].item<float>();
  const float i4 = a[1].item<float>();
  // Keep estimates in the physical ranges accepted by the original CC stage.
  const float f2 = (i2 - kSignalLength / 2.0f) / kSignalLength;
  const float f4 = (i4 - kSignalLength / 2.0f) / kSignalLength;
  float cfo = std::clamp(0.25f * f4, -0.001f, 0.001f);
  float rs = std::clamp(std::abs(f4 - 2.0f * f2), 1.0f / 23.0f, 0.5f);
  return at::tensor({rs, cfo}, at::TensorOptions().dtype(at::kFloat));
}

Tensor lr_predict_impl(const Tensor& features, const Tensor& center,
                       const Tensor& scale, const Tensor& weight,
                       const Tensor& bias) {
  Tensor x = features.to(at::kCPU, at::kFloat).contiguous();
  Tensor z = (x - center.to(at::kCPU, at::kFloat)) /
             scale.to(at::kCPU, at::kFloat).clamp_min(1.0e-12);
  Tensor logits = at::matmul(weight.to(at::kCPU, at::kFloat), z) +
                  bias.to(at::kCPU, at::kFloat);
  return at::argmax(logits);
}

using Seq = std::array<std::complex<float>, 11>;
using Pair = std::pair<int, int>;

const std::map<Pair, std::vector<int>> kActive = {
  {{2, 1}, {-1, 0, 1}},
  {{4, 0}, {-2, -1, 0, 1, 2}},
  {{4, 2}, {-2, -1, 0, 1, 2}},
  {{4, 4}, {-2, -1, 0, 1, 2}},
  {{6, 1}, {-2, -1, 0, 1, 2}},
  {{6, 3}, {-2, -1, 0, 1, 2}},
  {{6, 5}, {-2, -1, 0, 1, 2}},
};

std::complex<float> cpow_int(std::complex<float> z, int n) {
  std::complex<float> out(1.0f, 0.0f);
  for (int i = 0; i < n; ++i) out *= z;
  return out;
}

std::complex<float> project_recompute(const Complex* x, int64_t len, float t0, float f0,
                                      int n, int m, int k) {
  const float alpha = static_cast<float>(n - 2 * m) * f0 + static_cast<float>(k) / t0;
  const float step = -2.0f * kPi * alpha;
  const std::complex<float> rot(std::cos(step), std::sin(step));
  std::complex<float> phase(1.0f, 0.0f), sum(0.0f, 0.0f);
  for (int64_t i = 0; i < len; ++i) {
    const std::complex<float> z(x[i].real(), x[i].imag());
    const std::complex<float> u = cpow_int(z, n - m) * cpow_int(std::conj(z), m);
    sum += phase * u;
    phase *= rot;
    if ((i & 1023) == 1023) phase = std::polar(1.0f, step * static_cast<float>(i + 1));
  }
  return sum / static_cast<float>(len);
}

std::complex<float> project_precomputed(const std::vector<std::complex<float>>& u,
                                        float t0, float f0, int n, int m, int k) {
  const float alpha = static_cast<float>(n - 2 * m) * f0 + static_cast<float>(k) / t0;
  const float step = -2.0f * kPi * alpha;
  const std::complex<float> rot(std::cos(step), std::sin(step));
  std::complex<float> phase(1.0f, 0.0f), sum(0.0f, 0.0f);
  for (size_t i = 0; i < u.size(); ++i) {
    sum += phase * u[i];
    phase *= rot;
    if ((i & 1023) == 1023) phase = std::polar(1.0f, step * static_cast<float>(i + 1));
  }
  return sum / static_cast<float>(u.size());
}

Seq conv2(const Seq& a, const Seq& b) {
  std::array<std::complex<float>, 21> full{};
  for (int i = 0; i < 11; ++i) for (int j = 0; j < 11; ++j) full[i + j] += a[i] * b[j];
  Seq out{};
  for (int i = 0; i < 11; ++i) out[i] = full[i + 5];
  return out;
}

Seq conv3(const Seq& a, const Seq& b, const Seq& c) {
  std::array<std::complex<float>, 31> full{};
  for (int i = 0; i < 11; ++i)
    for (int j = 0; j < 11; ++j)
      for (int k = 0; k < 11; ++k) full[i + j + k] += a[i] * b[j] * c[k];
  Seq out{};
  for (int i = 0; i < 11; ++i) out[i] = full[i + 10];
  return out;
}

Tensor cc_matrix_impl(const Tensor& input, float rs, float cfo, bool simplified) {
  Tensor x = input.to(at::kCPU, at::kComplexFloat).contiguous().flatten();
  const auto* data = x.data_ptr<Complex>();
  const int64_t len = x.numel();
  const float t0 = 1.0f / std::clamp(rs, 1.0f / 23.0f, 0.5f);
  std::map<Pair, Seq> moments;

  if (!simplified) {
    for (const auto& entry : kActive) {
      Seq seq{};
      for (int k : entry.second) {
        seq[static_cast<size_t>(k + 5)] = project_recompute(
          data, len, t0, cfo, entry.first.first, entry.first.second, k);
      }
      moments[entry.first] = seq;
    }
  } else {
    std::vector<std::complex<float>> u21(len), u40(len), u42(len), u61(len), u63(len);
    for (int64_t i = 0; i < len; ++i) {
      std::complex<float> z(data[i].real(), data[i].imag());
      float a2 = std::norm(z);
      std::complex<float> z2 = z * z;
      std::complex<float> z4 = z2 * z2;
      u21[i] = {a2, 0.0f};
      u40[i] = z4;
      u42[i] = {a2 * a2, 0.0f};
      u61[i] = z4 * z * std::conj(z);
      u63[i] = {a2 * a2 * a2, 0.0f};
    }
    Seq m21{}, m40{}, m42{}, m44{}, m61{}, m63{}, m65{};
    for (int k : {0, 1}) m21[k + 5] = project_precomputed(u21, t0, cfo, 2, 1, k);
    m21[4] = std::conj(m21[6]);
    for (int k : {-2, -1, 0, 1, 2}) m40[k + 5] = project_precomputed(u40, t0, cfo, 4, 0, k);
    for (int k : {0, 1, 2}) m42[k + 5] = project_precomputed(u42, t0, cfo, 4, 2, k);
    m42[4] = std::conj(m42[6]); m42[3] = std::conj(m42[7]);
    for (int k : {-2, -1, 0, 1, 2}) m44[k + 5] = std::conj(m40[-k + 5]);
    for (int k : {-2, -1, 0, 1, 2}) m61[k + 5] = project_precomputed(u61, t0, cfo, 6, 1, k);
    for (int k : {0, 1, 2}) m63[k + 5] = project_precomputed(u63, t0, cfo, 6, 3, k);
    m63[4] = std::conj(m63[6]); m63[3] = std::conj(m63[7]);
    for (int k : {-2, -1, 0, 1, 2}) m65[k + 5] = std::conj(m61[-k + 5]);
    moments[{2, 1}] = m21; moments[{4, 0}] = m40; moments[{4, 2}] = m42;
    moments[{4, 4}] = m44; moments[{6, 1}] = m61; moments[{6, 3}] = m63;
    moments[{6, 5}] = m65;
  }

  Seq m21 = moments[{2, 1}], m40 = moments[{4, 0}], m42 = moments[{4, 2}];
  Seq m44 = moments[{4, 4}], m61 = moments[{6, 1}], m63 = moments[{6, 3}];
  Seq m65 = moments[{6, 5}];
  Seq c42 = conv2(m21, m21), c61 = conv2(m40, m21);
  Seq c63a = conv2(m42, m21), c63b = conv3(m21, m21, m21);
  Seq c65 = conv2(m44, m21);
  for (int i = 0; i < 11; ++i) {
    m42[i] -= 2.0f * c42[i];
    m61[i] -= 5.0f * c61[i];
    m63[i] += -9.0f * c63a[i] + 12.0f * c63b[i];
    m65[i] -= 5.0f * c65[i];
  }
  std::map<Pair, Seq> cumulants = {
    {{2, 1}, m21}, {{4, 0}, m40}, {{4, 2}, m42}, {{4, 4}, m44},
    {{6, 1}, m61}, {{6, 3}, m63}, {{6, 5}, m65},
  };
  const std::array<Pair, 15> columns = {
    Pair{2,0}, {2,1}, {2,2}, {4,0}, {4,1}, {4,2}, {4,3}, {4,4},
    {6,0}, {6,1}, {6,2}, {6,3}, {6,4}, {6,5}, {6,6}
  };
  float signal_power = at::mean(at::abs(x).square()).item<float>();
  signal_power = std::max(signal_power, 1.0e-12f);
  Tensor out = at::zeros({11, 15}, at::TensorOptions().dtype(at::kFloat));
  auto acc = out.accessor<float, 2>();
  for (int col = 0; col < 15; ++col) {
    auto it = cumulants.find(columns[col]);
    if (it == cumulants.end()) continue;
    const int norder = columns[col].first;
    const auto& active = kActive.at(columns[col]);
    for (int k : active) {
      acc[k + 5][col] = std::pow(std::abs(it->second[k + 5]), 2.0f / norder) / signal_power;
    }
  }
  return out;
}

void configure_threads_impl(int intraop_threads) {
  at::set_num_threads(std::max(1, intraop_threads));
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.doc() = "Native CPU signal-processing kernels for latency comparison 1--5";
  m.def("configure_threads", &configure_threads_impl, py::arg("intraop_threads"));
  m.def("read_tim", &read_tim_impl, py::arg("path"), py::arg("samples") = kSignalLength);
  m.def("boi_shift", &boi_shift_impl, py::arg("x"), py::arg("unit_power") = false);
  m.def("ssca_checksum", &ssca_checksum_impl, py::arg("x"), py::arg("conjugate"));
  m.def("fft_x4_checksum", &fft_x4_checksum_impl, py::arg("x"));
  m.def("snoap_features", &snoap_features_impl, py::arg("x"));
  m.def("lr55_features", &lr55_features_impl, py::arg("x"));
  m.def("estimate_rs_cfo", &estimate_rs_cfo_impl, py::arg("aux"));
  m.def("lr_predict", &lr_predict_impl, py::arg("features"), py::arg("center"),
        py::arg("scale"), py::arg("weight"), py::arg("bias"));
  m.def("cc_matrix", &cc_matrix_impl, py::arg("x"), py::arg("rs"), py::arg("cfo"),
        py::arg("simplified"));
}
