#ifndef _MEANABSOLUTEERROR_H_
#define _MEANABSOLUTEERROR_H_

#include <algorithm>
#include <array>
#include <functional>
#include <numeric>

#include "Metric.h"

template<typename T, std::size_t N>
class MeanAbsoluteError: public Metric<T, N> {
  /* Multi-variable MAE metrics */
private:
  metric_return_t absolute_error = 0.0f;
  std::size_t count = 0;

public:
  const char *name() {
    return "mae";
  }

  void update(std::array<T, N> preds, std::array<T, N> targets) {
    std::array<T, N> diff{};
    std::transform(preds.begin(), preds.end(), targets.begin(), diff.begin(), std::minus<T>());

    metric_return_t error = std::accumulate(diff.begin(), diff.end(), 0.0f);

    if (error > 0) {
      this->absolute_error += error;
    } else {
      this->absolute_error -= error;
    }

    this->count++;
  }

  metric_return_t compute() {
    return this->absolute_error / this->count;
  }
};

#endif//_MEANABSOLUTEERROR_H_
