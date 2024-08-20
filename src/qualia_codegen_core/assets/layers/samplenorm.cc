/**
  ******************************************************************************
  * @file    samplenorm.cc
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, Université Côte d'Azur, France
  * @version 1.0.0
  * @date    20 august 2024
  * @brief   SampleNorm
  */

#ifndef SINGLE_FILE
#include "{{ node.layer.name }}.h"
#include "number.h"
#endif

#define INPUT_CHANNELS {{ node.input_shape[0][-1] }}
#define MODE_{{ node.layer.mode.name | upper }}
#define TMP_SCALE_FACTOR {{ [node.q.weights_scale_factor, node.q.bias_scale_factor] | max if node.q.bias_scale_factor is not none else node.q.weights_scale_factor }}
#define INPUT_SCALE_FACTOR {{ node.innodes[0].q.output_scale_factor }}
#define OUTPUT_SCALE_FACTOR {{ node.q.output_scale_factor }}
#define OUTPUT_ROUND_MODE ROUND_MODE_{{ node.q.output_round_mode | upper }}
#define NUMBER_T {{ qtype2ctype(node.q.number_type, node.q.width) }}
#define LONG_NUMBER_T {{ qtype2ctype(node.q.number_type, node.q.long_width) }}

static inline void {{ node.layer.name }}(
  const NUMBER_T input{% for dim in node.input_shape[0][1:] %}[{{ dim }}]{% endfor %}, // IN
  {{ node.layer.name }}_output_type output) {    // OUT
  LONG_NUMBER_T tmp;
  unsigned short k;

#ifdef MODE_MINMAX
  // Per-channel normalization
  for (k = 0; k < INPUT_CHANNELS; k++) {
    NUMBER_T min = input{% for dim in node.input_shape[0][1:-1] %}[0]{% endfor %}[k];
    NUMBER_T max = input{% for dim in node.input_shape[0][1:-1] %}[0]{% endfor %}[k];

    // Compute min and max
{% for dim in node.input_shape[0][1:-1] %}
    for (size_t i_{{ loop.index }} = 0; i_{{ loop.index }} < {{ dim }}; i_{{ loop.index }}++) { 
{% endfor %}
      if (input{% for dim in node.input_shape[0][1:-1] %}[i_{{ loop.index }}]{% endfor %}[k] > max) {
        max = input{% for dim in node.input_shape[0][1:-1] %}[i_{{ loop.index }}]{% endfor %}[k];
      }
      if (input{% for dim in node.input_shape[0][1:-1] %}[i_{{ loop.index }}]{% endfor %}[k] < min) {
        min = input{% for dim in node.input_shape[0][1:-1] %}[i_{{ loop.index }}]{% endfor %}[k];
      }
{% for dim in node.input_shape[0][1:-1] %}
    }
{% endfor %}

    // Normalize
{% for dim in node.input_shape[0][1:-1] %}
    for (size_t i_{{ loop.index }} = 0; i_{{ loop.index }} < {{ dim }}; i_{{ loop.index }}++) { 
{% endfor %}
      tmp = input{% for dim in node.input_shape[0][1:-1] %}[i_{{ loop.index }}]{% endfor %}[k] - min;
      tmp = scale(NUMBER_T, tmp, -(OUTPUT_SCALE_FACTOR), OUTPUT_ROUND_MODE);

{% if node.q.number_type.__name__ == 'int' %}
      if (OUTPUT_ROUND_MODE == ROUND_MODE_NEAREST) {
        tmp += (1 << (INPUT_SCALE_FACTOR + OUTPUT_SCALE_FACTOR - 1)); // +0.5 in fixed-point
      }
{% endif %}

      tmp /= max;
      output{% for dim in node.input_shape[0][1:-1] %}[i_{{ loop.index }}]{% endfor %}[k] = clamp_to(NUMBER_T, tmp);
{% for dim in node.input_shape[0][1:-1] %}
    }
{% endfor %}
  }
#else
#error "Unsupported mode {{ node.layer.mode.name }}"
#endif
}

#undef INPUT_CHANNELS
#undef MODE_{{ node.layer.mode.name | upper }}
#undef TMP_SCALE_FACTOR
#undef INPUT_SCALE_FACTOR
#undef OUTPUT_SCALE_FACTOR
#undef OUTPUT_ROUND_MODE
#undef NUMBER_T
#undef LONG_NUMBER_T
