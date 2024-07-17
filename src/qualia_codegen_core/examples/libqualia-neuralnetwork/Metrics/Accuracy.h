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
    auto e = std::max_element(preds.begin(), preds.end());
    auto i = std::distance(preds.begin(), e);

    valids += targets.at(i);

    this->count++;
  }

  float compute() {
    return this->valids / this->count;
  }
};

#endif//_ACCURACY_H_
