#ifndef _METRICS_H_
#define _METRICS_H_

#include "Metrics/Metric.h"
#include "model.h"

extern std::array<Metric<metric_return_t, MODEL_OUTPUT_SAMPLES>*, {{ metrics | length }}> metrics;

#endif//_METRICS_H_
