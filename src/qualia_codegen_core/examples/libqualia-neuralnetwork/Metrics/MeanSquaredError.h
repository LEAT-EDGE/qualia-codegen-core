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
  double squared_error = 0.0;
  std::size_t count = 0;

public:
  const char *name() {
    return "mse";
  }

  void update(std::array<T, N> preds, std::array<T, N> targets) {
    std::array<T, N> diff{};
    std::transform(preds.begin(), preds.end(), targets.begin(), diff.begin(), std::minus<T>());

    double error = std::accumulate(diff.begin(), diff.end(), 0.0);

    squared_error += error * error;

    this->count++;
  }

  float compute() {
    return this->squared_error / this->count;
  }
};

#endif//_MEANSQUAREDERROR_H_
