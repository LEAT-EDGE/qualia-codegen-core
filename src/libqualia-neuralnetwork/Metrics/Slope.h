#ifndef _SLOPE_H_
#define _SLOPE_H_

#include <algorithm>
#include <array>
#include <cmath>
#include <functional>
#include <numeric>

#include "Metric.h"

template<typename T, std::size_t N>
class Slope: public Metric<T, N> {
  /* Linear regression slope metric */
private:
  std::size_t count = 0;

  std::array<metric_return_t, N> mean_x{};
  std::array<metric_return_t, N> mean_y{};
  std::array<metric_return_t, N> mean_xy{};
  std::array<metric_return_t, N> mean_squared_x{};
  std::array<metric_return_t, N> mean_squared_y{};
  //std::array<metric_return_t, N> var_x{};
  //std::array<metric_return_t, N> var_y{};

public:
  const char *name() {
    return "slope";
  }

  void update(std::array<T, N> preds, std::array<T, N> targets) {
    std::array<metric_return_t, N> new_mean_x{};
    std::array<metric_return_t, N> new_mean_y{};
    std::array<metric_return_t, N> xy;


    std::transform(preds.begin(), preds.end(), targets.begin(), xy.begin(), [](T p, T t) { return static_cast<metric_return_t>(p) * static_cast<metric_return_t>(t); });

    if (this->count == 0) {
      // Initial mean, needs conversion from whatever preds and targets are (float, int) to metric_return_t
      std::transform(preds.begin(), preds.end(), new_mean_x.begin(), [](T v) { return static_cast<metric_return_t>(v); });
      std::transform(targets.begin(), targets.end(), new_mean_y.begin(), [](T v) { return static_cast<metric_return_t>(v); });
      this->mean_xy = xy;

      std::transform(preds.begin(), preds.end(), this->mean_squared_x.begin(), [](T v) { return static_cast<metric_return_t>(v) * static_cast<metric_return_t>(v); });
      std::transform(targets.begin(), targets.end(), this->mean_squared_y.begin(), [](T v) { return static_cast<metric_return_t>(v) * static_cast<metric_return_t>(v); });
      // Variance stays at 0
    } else {
      // Compute new mean
      std::transform(preds.begin(),
                     preds.end(),
                     this->mean_x.begin(),
                     new_mean_x.begin(),
                     [this](T v, metric_return_t cur_mean) { return (cur_mean * this->count + static_cast<metric_return_t>(v)) / (this->count + 1); });
      std::transform(targets.begin(),
                     targets.end(),
                     this->mean_y.begin(),
                     new_mean_y.begin(),
                     [this](T v, metric_return_t cur_mean) { return (cur_mean * this->count + static_cast<metric_return_t>(v)) / (this->count + 1); });

      std::transform(preds.begin(),
                     preds.end(),
                     this->mean_squared_x.begin(),
                     this->mean_squared_x.begin(),
                     [this](T v, metric_return_t cur_mean) { return (cur_mean * this->count + static_cast<metric_return_t>(v) * static_cast<metric_return_t>(v)) / (this->count + 1); });
      std::transform(targets.begin(),
                     targets.end(),
                     this->mean_squared_y.begin(),
                     this->mean_squared_y.begin(),
                     [this](T v, metric_return_t cur_mean) { return (cur_mean * this->count + static_cast<metric_return_t>(v) * static_cast<metric_return_t>(v)) / (this->count + 1); });
      // xy is already metric_return_t, no conversion required
      std::transform(xy.begin(),
                     xy.end(),
                     this->mean_xy.begin(),
                     this->mean_xy.begin(),
                     [this](metric_return_t v, metric_return_t cur_mean) { return (cur_mean * this->count + v) / (this->count + 1); });

      // Update variance
      /*
      std::array<metric_return_t, N> diff_mean_x{};
      std::array<metric_return_t, N> diff_mean_y{};
      std::array<metric_return_t, N> diff_new_mean_x{};
      std::array<metric_return_t, N> diff_new_mean_y{};
      std::array<metric_return_t, N> delta_var_x{};
      std::array<metric_return_t, N> delta_var_y{};

      std::transform(preds.begin(), preds.end(), new_mean_x.begin(), diff_new_mean_x.begin(), std::minus<metric_return_t>());
      std::transform(preds.begin(), preds.end(), this->mean_x.begin(), diff_mean_x.begin(), std::minus<metric_return_t>());
      std::transform(diff_new_mean_x.begin(), diff_new_mean_x.end(), diff_mean_x.begin(), delta_var_x.begin(), std::minus<metric_return_t>());
      std::transform(delta_var_x.begin(), delta_var_x.end(), this->var_x.begin(), this->var_x.begin(), std::plus<metric_return_t>());

      std::transform(targets.begin(), targets.end(), new_mean_y.begin(), diff_new_mean_y.begin(), std::minus<metric_return_t>());
      std::transform(targets.begin(), targets.end(), this->mean_y.begin(), diff_mean_y.begin(), std::minus<metric_return_t>());
      std::transform(diff_new_mean_y.begin(), diff_new_mean_y.end(), diff_mean_y.begin(), delta_var_y.begin(), std::minus<metric_return_t>());
      std::transform(delta_var_y.begin(), delta_var_y.end(), this->var_y.begin(), this->var_y.begin(), std::plus<metric_return_t>());
      */
    }

    this->mean_x = new_mean_x;
    this->mean_y = new_mean_y;
    this->count++;
  }

  metric_return_t compute() {
    std::array<metric_return_t, N> squared_mean_x{};
    std::array<metric_return_t, N> squared_mean_y{};
    std::array<metric_return_t, N> var_y{};
    std::array<metric_return_t, N> covar{};
    std::array<metric_return_t, N> mean_x_mean_y{};
    std::array<metric_return_t, N> slope{};

    std::transform(this->mean_x.begin(), this->mean_x.end(), squared_mean_x.begin(), [](metric_return_t v) { return v * v; });
    std::transform(this->mean_y.begin(), this->mean_y.end(), squared_mean_y.begin(), [](metric_return_t v) { return v * v; });

    std::transform(this->mean_squared_y.begin(), this->mean_squared_y.end(), squared_mean_y.begin(), var_y.begin(), std::minus<metric_return_t>());

    std::transform(this->mean_x.begin(), this->mean_x.end(), this->mean_y.begin(), mean_x_mean_y.begin(), std::multiplies<metric_return_t>());
    std::transform(this->mean_xy.begin(), this->mean_xy.end(), mean_x_mean_y.begin(), covar.begin(), std::minus<metric_return_t>());

    std::transform(covar.begin(), covar.end(), var_y.begin(), slope.begin(), std::divides<metric_return_t>());

    return slope.at(0);
  }
};

#endif//_SLOPE_H_
