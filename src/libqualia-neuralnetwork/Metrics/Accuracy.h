#ifndef _ACCURACY_H_
#define _ACCURACY_H_

#include <algorithm>
#include <array>

#include "Metric.h"

template<typename T, std::size_t N>
class Accuracy: public Metric<T, N> {
  /* Multi-class accuracy metrics, expects targets to be one-hot encoded */
private:
  std::size_t valids = 0;
  std::size_t count = 0;

public:
  const char *name() {
    return "acc";
  }

  void update(std::array<T, N> preds, std::array<T, N> targets) {
    auto preds_e = std::max_element(preds.begin(), preds.end());
    auto preds_i = std::distance(preds.begin(), preds_e);
    auto targets_e = std::max_element(targets.begin(), targets.end());
    auto targets_i = std::distance(targets.begin(), targets_e);

    if (preds_i == targets_i) {
      valids++;
    }

    this->count++;
  }

  metric_return_t compute() {
    return this->valids / static_cast<metric_return_t>(this->count);
  }
};

#endif//_ACCURACY_H_
