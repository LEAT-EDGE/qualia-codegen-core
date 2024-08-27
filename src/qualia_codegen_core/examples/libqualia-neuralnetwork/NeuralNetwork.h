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

float *serialBufToFloats(char buf[], size_t buflen);
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
    output_t output;
    neuralNetworkRun(input.data(), output);
    return std::to_array(output);
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

    for (auto &metric: metrics) {
      metric->update(preds, targets);
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
