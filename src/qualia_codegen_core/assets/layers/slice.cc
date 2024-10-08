/**
  ******************************************************************************
  * @file    slice.cc
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, Université Côte d'Azur, France
  * @version 1.0.0
  * @date    20 august 2024
  * @brief   Slice
  */

#ifndef SINGLE_FILE
#include "{{ node.layer.name }}.h"
#include "number.h"
#endif

#define NUMBER_T {{ qtype2ctype(node.q.number_type, node.q.width) }}

static inline void {{ node.layer.name }}(
  const NUMBER_T input{% for dim in node.input_shape[0][1:] %}[{{ dim }}]{% endfor %}, // IN
  {{ node.layer.name }}_output_type output) {    // OUT

{% for dim in node.input_shape[0][1:] %}
  size_t i_{{ loop.index }}, o_{{ loop.index }};
  for (i_{{ loop.index }} = {{ node.layer.slices[loop.index].start if node.layer.slices[loop.index].start is not none else 0 }}, // Input index
    o_{{ loop.index }} = 0; // Output index
    i_{{ loop.index }} < {{ node.layer.slices[loop.index].stop if node.layer.slices[loop.index].stop is not none else dim }};
    i_{{ loop.index }} += {{ node.layer.slices[loop.index].step if node.layer.slices[loop.index].step is not none else 1 }},
    o_{{ loop.index }}++) {
{% endfor %}
    output{% for dim in node.input_shape[0][1:] %}[o_{{ loop.index }}]{% endfor %} = input{% for dim in node.input_shape[0][1:] %}[i_{{ loop.index }}]{% endfor %};
{% for dim in node.input_shape[0][1:] %}
  }
{% endfor %}
}

#undef NUMBER_T
