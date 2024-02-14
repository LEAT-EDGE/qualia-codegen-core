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

#ifdef WITH_CMSIS_NN
#include "arm_nnfunctions.h"
#endif

#define ACTIVATION_{{ node.layer.activation.name | upper if node.layer.activation is defined else "LINEAR" }}

// For fixed point quantization
#define ACC_SCALE_FACTOR {{ node.innodes | map(attribute="q") | max(attribute="output_scale_factor") | attr("output_scale_factor") }} // Get maximum scale factor of previous layers
#define OUTPUT_SCALE_FACTOR {{ node.q.output_scale_factor }}
#define OUTPUT_ROUND_MODE ROUND_MODE_{{ node.q.output_round_mode | upper }}
#define NUMBER_T {{ qtype2ctype(node.q.number_type, node.q.width) }}
#define LONG_NUMBER_T {{ qtype2ctype(node.q.number_type, node.q.long_width) }}

static inline void {{ node.layer.name }}(
{% for innode in node.innodes %}
  const NUMBER_T vector_in_{{ loop.index }}{% for dim in node.input_shape[loop.index - 1][1:] %}[{{ dim }}]{% endfor %}, // doesn't work with inverted data_format
{% endfor %}
  {{ node.layer.name }}_output_type vector_out) {    // OUT

  size_t x;
  LONG_NUMBER_T output_acc;

{% for innode in node.innodes %}
  NUMBER_T *i_{{ loop.index }} = (NUMBER_T*)vector_in_{{ loop.index }};
{% endfor %}

  NUMBER_T *o = (NUMBER_T*)vector_out;

  for (x = 0; x < {{ node.output_shape[0][1:] | join('*') }}; x++) {
    // scale all fixed point inputs to same factor and add them, negative factor is left shift
    output_acc = {% for s in node.innodes %}
                    + scale(NUMBER_T, (LONG_NUMBER_T)i_{{ loop.index }}[x], {{ s.q.output_scale_factor }} - ACC_SCALE_FACTOR, OUTPUT_ROUND_MODE)
                 {% endfor %};
#ifdef ACTIVATION_LINEAR
    o[x] = scale_and_clamp_to(NUMBER_T, output_acc, ACC_SCALE_FACTOR - OUTPUT_SCALE_FACTOR, OUTPUT_ROUND_MODE);
#elif defined(ACTIVATION_RELU)
    if (output_acc < 0) {
      o[x] = 0;
    } else {
      o[x] = scale_and_clamp_to(NUMBER_T, output_acc, ACC_SCALE_FACTOR - OUTPUT_SCALE_FACTOR, OUTPUT_ROUND_MODE);
    }
#else
#error "Unsupported activation function"
#endif
  }
}

#undef ACTIVATION_{{ node.layer.activation.name | upper if node.layer.activation is defined else "LINEAR" }}
#undef ACC_SCALE_FACTOR
#undef OUTPUT_SCALE_FACTOR
#undef NUMBER_T
#undef LONG_NUMBER_T
