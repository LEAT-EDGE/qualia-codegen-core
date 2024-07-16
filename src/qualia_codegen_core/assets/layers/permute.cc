/**
  ******************************************************************************
  * @file    permute.cc
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, Université Côte d'Azur, France
  * @version 1.0.0
  * @date    16 july 2024
  * @brief   Permute
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
  for (size_t i_{{ loop.index }} = 0; i_{{ loop.index }} < {{ dim }}; i_{{ loop.index }}++) { 
{% endfor %}
    output{% for dim in node.layer.dims[1:] %}[i_{{ dim }}]{% endfor %} = input{% for dim in node.input_shape[0][1:] %}[i_{{ loop.index }}]{% endfor %};
{% for dim in node.input_shape[0][1:] %}
  }
{% endfor %}
}

#undef NUMBER_T
