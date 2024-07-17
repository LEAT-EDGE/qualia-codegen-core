#ifndef _PEARSONCORRELATIONCOEFFICIENT_H_
#define _PEARSONCORRELATIONCOEFFICIENT_H_

#include <algorithm>
#include <array>
#include <cmath>
#include <functional>
#include <numeric>

#include "Metric.h"

template<typename T, std::size_t N>
class PearsonCorrelationCoefficient: public Metric<T, N> {
  /* Pearson correlation coefficient metric */
private:
  std::size_t count = 0;

  std::array<double, N> mean_x{};
  std::array<double, N> mean_y{};
  std::array<double, N> mean_xy{};
  std::array<double, N> mean_squared_x{};
  std::array<double, N> mean_squared_y{};
  //std::array<double, N> var_x{};
  //std::array<double, N> var_y{};

public:
  const char *name() {
    return "corr";
  }

  void update(std::array<T, N> preds, std::array<T, N> targets) {
    std::array<double, N> new_mean_x{};
    std::array<double, N> new_mean_y{};
    std::array<double, N> xy;


    std::transform(preds.begin(), preds.end(), targets.begin(), xy.begin(), [](T p, T t) { return static_cast<double>(p) * static_cast<double>(t); });

    if (this->count == 0) {
      // Initial mean, needs conversion from whatever preds and targets are (float, int) to double
      std::transform(preds.begin(), preds.end(), new_mean_x.begin(), [](T v) { return static_cast<double>(v); });
      std::transform(targets.begin(), targets.end(), new_mean_y.begin(), [](T v) { return static_cast<double>(v); });
      this->mean_xy = xy;

      std::transform(preds.begin(), preds.end(), this->mean_squared_x.begin(), [](T v) { return static_cast<double>(v) * static_cast<double>(v); });
      std::transform(targets.begin(), targets.end(), this->mean_squared_y.begin(), [](T v) { return static_cast<double>(v) * static_cast<double>(v); });
      // Variance stays at 0
    } else {
      // Compute new mean
      std::transform(preds.begin(),
                     preds.end(),
                     this->mean_x.begin(),
                     new_mean_x.begin(),
                     [this](T v, double cur_mean) { return (cur_mean * this->count + static_cast<double>(v)) / (this->count + 1); });
      std::transform(targets.begin(),
                     targets.end(),
                     this->mean_y.begin(),
                     new_mean_y.begin(),
                     [this](T v, double cur_mean) { return (cur_mean * this->count + static_cast<double>(v)) / (this->count + 1); });

      std::transform(preds.begin(),
                     preds.end(),
                     this->mean_squared_x.begin(),
                     this->mean_squared_x.begin(),
                     [this](T v, double cur_mean) { return (cur_mean * this->count + static_cast<double>(v) * static_cast<double>(v)) / (this->count + 1); });
      std::transform(targets.begin(),
                     targets.end(),
                     this->mean_squared_y.begin(),
                     this->mean_squared_y.begin(),
                     [this](T v, double cur_mean) { return (cur_mean * this->count + static_cast<double>(v) * static_cast<double>(v)) / (this->count + 1); });
      // xy is already double, no conversion required
      std::transform(xy.begin(),
                     xy.end(),
                     this->mean_xy.begin(),
                     this->mean_xy.begin(),
                     [this](double v, double cur_mean) { return (cur_mean * this->count + v) / (this->count + 1); });

      // Update variance
      /*
      std::array<double, N> diff_mean_x{};
      std::array<double, N> diff_mean_y{};
      std::array<double, N> diff_new_mean_x{};
      std::array<double, N> diff_new_mean_y{};
      std::array<double, N> delta_var_x{};
      std::array<double, N> delta_var_y{};

      std::transform(preds.begin(), preds.end(), new_mean_x.begin(), diff_new_mean_x.begin(), std::minus<double>());
      std::transform(preds.begin(), preds.end(), this->mean_x.begin(), diff_mean_x.begin(), std::minus<double>());
      std::transform(diff_new_mean_x.begin(), diff_new_mean_x.end(), diff_mean_x.begin(), delta_var_x.begin(), std::minus<double>());
      std::transform(delta_var_x.begin(), delta_var_x.end(), this->var_x.begin(), this->var_x.begin(), std::plus<double>());

      std::transform(targets.begin(), targets.end(), new_mean_y.begin(), diff_new_mean_y.begin(), std::minus<double>());
      std::transform(targets.begin(), targets.end(), this->mean_y.begin(), diff_mean_y.begin(), std::minus<double>());
      std::transform(diff_new_mean_y.begin(), diff_new_mean_y.end(), diff_mean_y.begin(), delta_var_y.begin(), std::minus<double>());
      std::transform(delta_var_y.begin(), delta_var_y.end(), this->var_y.begin(), this->var_y.begin(), std::plus<double>());
      */
    }

    this->mean_x = new_mean_x;
    this->mean_y = new_mean_y;
    this->count++;
  }

  float compute() {
    std::array<double, N> squared_mean_x{};
    std::array<double, N> squared_mean_y{};
    std::array<double, N> var_x{};
    std::array<double, N> var_y{};
    std::array<double, N> std_x{};
    std::array<double, N> std_y{};
    std::array<double, N> covar{};
    std::array<double, N> mean_x_mean_y{};
    std::array<double, N> std_x_std_y{};
    std::array<double, N> pcc{};

    std::transform(this->mean_x.begin(), this->mean_x.end(), squared_mean_x.begin(), [](double v) { return v * v; });
    std::transform(this->mean_y.begin(), this->mean_y.end(), squared_mean_y.begin(), [](double v) { return v * v; });

    std::transform(this->mean_squared_x.begin(), this->mean_squared_x.end(), squared_mean_x.begin(), var_x.begin(), std::minus<double>());
    std::transform(this->mean_squared_y.begin(), this->mean_squared_y.end(), squared_mean_y.begin(), var_y.begin(), std::minus<double>());

    std::transform(this->mean_x.begin(), this->mean_x.end(), this->mean_y.begin(), mean_x_mean_y.begin(), std::multiplies<double>());
    std::transform(this->mean_xy.begin(), this->mean_xy.end(), mean_x_mean_y.begin(), covar.begin(), std::minus<double>());

    std::transform(var_x.begin(), var_x.end(), std_x.begin(), [](double v) { return std::sqrt(v); });
    std::transform(var_y.begin(), var_y.end(), std_y.begin(), [](double v) { return std::sqrt(v); });

    std::transform(std_x.begin(), std_x.end(), std_y.begin(), std_x_std_y.begin(), std::multiplies<double>());
    std::transform(covar.begin(), covar.end(), std_x_std_y.begin(), pcc.begin(), std::divides<double>());

    return pcc.at(0);
  }
};

#endif//_PEARSONCORRELATIONCOEFFICIENT_H_
