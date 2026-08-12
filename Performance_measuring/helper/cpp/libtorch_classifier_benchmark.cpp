#include <ATen/Parallel.h>
#include <torch/script.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <sched.h>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

std::vector<int64_t> parse_shape(const std::string& text) {
    std::vector<int64_t> shape;
    std::stringstream stream(text);
    std::string token;
    while (std::getline(stream, token, ',')) {
        shape.push_back(std::stoll(token));
    }
    if (shape.empty()) {
        throw std::runtime_error("input shape must not be empty");
    }
    return shape;
}

void pin_to_core(int core) {
    if (core < 0) {
        return;
    }
    cpu_set_t affinity;
    CPU_ZERO(&affinity);
    CPU_SET(core, &affinity);
    if (sched_setaffinity(0, sizeof(affinity), &affinity) != 0) {
        throw std::runtime_error("sched_setaffinity failed");
    }
}

double run_ms(
    torch::jit::Module& module,
    const std::vector<torch::jit::IValue>& inputs,
    int repeats,
    torch::Tensor& last_output) {
    const auto start = std::chrono::steady_clock::now();
    for (int repeat = 0; repeat < repeats; ++repeat) {
        last_output = module.forward(inputs).toTensor();
    }
    const auto stop = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::milli>(stop - start).count();
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 9) {
        std::cerr
            << "usage: libtorch_classifier_benchmark MODEL SHAPE THREADS CORE "
               "WARMUPS TARGET_MS MAX_REPEATS TRIALS\n";
        return 2;
    }

    try {
        const std::string model_path = argv[1];
        const std::vector<int64_t> input_shape = parse_shape(argv[2]);
        const int threads = std::stoi(argv[3]);
        const int core = std::stoi(argv[4]);
        const int warmups = std::stoi(argv[5]);
        const double target_ms = std::stod(argv[6]);
        const int max_repeats = std::stoi(argv[7]);
        const int trials = std::stoi(argv[8]);

        pin_to_core(core);
        at::set_num_threads(threads);
        at::set_num_interop_threads(1);

        c10::InferenceMode inference_guard;
        torch::jit::Module module = torch::jit::load(model_path, torch::kCPU);
        module.eval();
        module = torch::jit::freeze(module);
        module = torch::jit::optimize_for_inference(module);

        torch::manual_seed(0x20260807);
        torch::Tensor input = torch::randn(input_shape, torch::TensorOptions().dtype(torch::kFloat32));
        const std::vector<torch::jit::IValue> inputs{input};
        torch::Tensor last_output;

        for (int warmup = 0; warmup < warmups; ++warmup) {
            last_output = module.forward(inputs).toTensor();
        }

        const double pilot_ms = std::max(run_ms(module, inputs, 1, last_output), 1e-6);
        const int repeats = std::max(
            1,
            std::min(max_repeats, static_cast<int>(std::ceil(target_ms / pilot_ms))));

        std::vector<double> trial_per_inference_ms;
        trial_per_inference_ms.reserve(trials);
        for (int trial = 0; trial < trials; ++trial) {
            const double elapsed_ms = run_ms(module, inputs, repeats, last_output);
            trial_per_inference_ms.push_back(elapsed_ms / repeats);
        }

        std::vector<double> sorted = trial_per_inference_ms;
        std::sort(sorted.begin(), sorted.end());
        const double median_ms = sorted[sorted.size() / 2];
        const double mean_ms = std::accumulate(
            trial_per_inference_ms.begin(), trial_per_inference_ms.end(), 0.0) /
            trial_per_inference_ms.size();
        const double checksum = last_output.sum().item<double>();

        std::cout << std::setprecision(12)
                  << "{\"repeats\":" << repeats
                  << ",\"pilot_ms\":" << pilot_ms
                  << ",\"trial_ms\":[";
        for (std::size_t index = 0; index < trial_per_inference_ms.size(); ++index) {
            if (index != 0) {
                std::cout << ',';
            }
            std::cout << trial_per_inference_ms[index];
        }
        std::cout << "]"
                  << ",\"median_ms\":" << median_ms
                  << ",\"mean_ms\":" << mean_ms
                  << ",\"checksum\":" << checksum
                  << "}\n";
        return 0;
    } catch (const c10::Error& error) {
        std::cerr << error.what() << '\n';
        return 1;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
