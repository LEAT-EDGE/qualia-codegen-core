/**
  ******************************************************************************
  * @file    operator.cc
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version 1.0.0
  * @date    24 march 2020
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#ifndef SINGLE_FILE
#include "{{ node.layer.name }}.h"
#include "number.h"
#endif

// For fixed point quantization
#define NUMBER_T {{ qtype2ctype(node.q.number_type, node.q.width) }}

static inline void {{ node.layer.name }}(
{%- for innode in node.innodes %}
  const NUMBER_T vector_in_{{ loop.index }}{% for dim in node.input_shape[loop.index - 1][1:] %}[{{ dim }}]{% endfor %},
{%- endfor %}
  {{ node.layer.name }}_output_type vector_out) {    // OUT

  size_t input_x;
  size_t output_x = 0;

{% for innode in node.innodes %}
  NUMBER_T *input_{{ loop.index }} = (NUMBER_T*)vector_in_{{ loop.index }};
{%- endfor %}

  NUMBER_T *output = (NUMBER_T*)vector_out;

{% for input_shape in node.input_shape %}
  for (input_x = 0; input_x < {{ input_shape[1:] | join('*') }}; input_x++, output_x++) {
    output[output_x] = input_{{ loop.index }}[input_x];
  }
{% endfor %}
}

#undef NUMBER_T
