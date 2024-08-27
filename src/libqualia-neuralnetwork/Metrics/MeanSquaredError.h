#ifndef _MEANSQUAREDERROR_H_
#define _MEANSQUAREDERROR_H_

#include <algorithm>
#include <array>
#include <functional>
#include <numeric>

#include "Metric.h"

template<typename T, std::size_t N>
class MeanSquaredError: public Metric<T, N> {
  /* Multi-variable MSE metrics */
private:
  metric_return_t squared_error = 0.0f;
  std::size_t count = 0;

public:
  const char *name() {
    return "mse";
  }

  void update(std::array<T, N> preds, std::array<T, N> targets) {
    std::array<T, N> diff{};
    std::transform(preds.begin(), preds.end(), targets.begin(), diff.begin(), std::minus<T>());

    metric_return_t error = std::accumulate(diff.begin(), diff.end(), 0.0f);

    this->squared_error += error * error;

    this->count++;
  }

  metric_return_t compute() {
    return this->squared_error / this->count;
  }
};

#endif//_MEANSQUAREDERROR_H_
