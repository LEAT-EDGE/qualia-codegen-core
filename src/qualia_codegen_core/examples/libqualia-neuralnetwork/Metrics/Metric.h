#ifndef _METRIC_H_
#define _METRIC_H_

#include <array>

template<typename T, std::size_t N>
class Metric {
public:
  virtual ~Metric() {};

  virtual const char *name() = 0;
  virtual void update(std::array<T, N> preds, std::array<T, N> targets) = 0;
  virtual float compute() = 0;
};

#endif//_METRIC_H_
