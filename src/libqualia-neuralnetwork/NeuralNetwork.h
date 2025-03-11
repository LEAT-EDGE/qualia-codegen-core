// Copyright 2021 (c) Pierre-Emmanuel Novac <penovac@unice.fr> Université Côte d'Azur, CNRS, LEAT. All rights reserved.

#ifndef _NEURALNETWORK_H_
#define _NEURALNETWORK_H_



#ifdef __cplusplus
extern "C" {
#endif

#include "model.h"

struct NNResult {
	unsigned int inference_count;
	unsigned int label;
	float dist;
};

extern unsigned int inference_count;
float *serialBufToFloats(char buf[], size_t buflen);
void stringToFloatArray(float floats[], size_t floatslen, char string[], size_t stringlen);
float round_with_mode(float v, round_mode_t round_mode);
struct NNResult neuralNetworkInfer(const float input[]);
void neuralNetworkRun(const float input[], output_t output);

#ifdef __cplusplus
}
#include <array>
#include <algorithm>
#include <cstring>
#include <cstdint>
#include <cstdlib>
#include <cmath>
#include <memory>
#include <type_traits>

#include "Metrics/Metric.h"

template<size_t NMetrics = 0, typename MetricT = float, std::size_t MetricN = 0>
class NeuralNetwork {
protected:
  std::array<Metric<MetricT, MetricN>*, NMetrics> metrics;

  unsigned int inference_count = 0;
public:
  NeuralNetwork() {

  }
  NeuralNetwork(std::array<Metric<MetricT, MetricN>*, NMetrics> metrics) : metrics(metrics) {

  }

  std::array<std::remove_all_extents<output_t>::type, MODEL_OUTPUT_SAMPLES> run(const std::array<float, MODEL_INPUT_DIMS> input) {
    std::array<std::remove_all_extents<output_t>::type, MODEL_OUTPUT_SAMPLES> output;
    output_t c_output;

    neuralNetworkRun(input.data(), c_output);

    for (size_t i = 0; i < MODEL_OUTPUT_SAMPLES; i++) {
      output[i] = c_output[i];
    }

    return output;
  }

  NNResult classify(const std::array<float, MODEL_INPUT_DIMS> input) {
    auto preds = this->run(input);
    auto e = std::max_element(preds.begin(), preds.end());
    auto i = std::distance(preds.begin(), e);

    inference_count++;

    return {inference_count, i, *e};
  }

  virtual std::array<std::remove_all_extents<output_t>::type, MODEL_OUTPUT_SAMPLES> evaluate(
    const std::array<float, MODEL_INPUT_DIMS> input,
    const std::array<float, MODEL_OUTPUT_SAMPLES> targets) {
    auto preds = this->run(input);

    // De-quantize predictions to match targets for metrics computation
    std::array<metric_return_t, MODEL_OUTPUT_SAMPLES> deqpreds{};
    std::transform(preds.begin(),
                   preds.end(),
                   deqpreds.begin(),
                   [](MODEL_OUTPUT_NUMBER_T v) {
                    return static_cast<metric_return_t>(v) / (1 << MODEL_OUTPUT_SCALE_FACTOR);
                   });

    for (auto &metric: this->metrics) {
      metric->update(deqpreds, targets);
    }

    return preds;
  }

  std::array<float, NMetrics> getMetricsResult() {
    std::array<metric_return_t, NMetrics> metrics_result{};

    std::transform(metrics.begin(),
          metrics.end(),
          metrics_result.begin(),
          [](Metric<MetricT, MetricN> *metric) {
            return metric->compute();
    });

    return metrics_result;
  }

};
#endif
#endif
