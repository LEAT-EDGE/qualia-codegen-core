#ifndef _MEANABSOLUTEPERCENTAGEERROR_H_
#define _MEANABSOLUTEPERCENTAGEERROR_H_

#include <algorithm>
#include <array>
#include <functional>
#include <numeric>

#include "Metric.h"

template<typename T, std::size_t N>
class MeanAbsolutePercentageError: public Metric<T, N> {
  /* Multi-variable MAPE metrics */
private:
  constexpr static const metric_return_t epsilon = 1.17e-06f; // From torchmetrics

  metric_return_t absolute_error = 0.0f;
  std::size_t count = 0;

public:
  const char *name() {
    return "mape";
  }

  void update(std::array<T, N> preds, std::array<T, N> targets) {
    std::array<T, N> diff{};
    std::transform(preds.begin(), preds.end(), targets.begin(), diff.begin(), std::minus<T>());

    std::transform(diff.begin(), diff.end(), diff.begin(), [](T v) { return std::abs(v); });

    std::array<metric_return_t, N> denominator{};
    std::transform(targets.begin(), targets.end(), denominator.begin(), [this](T v) { return std::max(static_cast<metric_return_t>(std::abs(v)), this->epsilon); });

    std::array<metric_return_t, N> quotient{};
    std::transform(diff.begin(), diff.end(), denominator.begin(), quotient.begin(), std::divides<metric_return_t>());

    metric_return_t error = std::accumulate(quotient.begin(), quotient.end(), 0.0f);

    this->absolute_error += error;

    this->count += N;
  }

  metric_return_t compute() {
    return this->absolute_error / this->count;
  }
};

#endif//_MEANABSOLUTEPERCENTAGEERROR_H_
