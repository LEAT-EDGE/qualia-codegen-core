/**
  ******************************************************************************
  * @file    sum.cc
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version 1.0.0
  * @date    7 february 2023
  * @brief   Global Sum Pooling
  */

#ifndef SINGLE_FILE
#include "{{ node.layer.name }}.h"
#include "number.h"
#endif

#ifdef WITH_CMSIS_NN
#include "arm_nnfunctions.h"
#endif

#define INPUT_CHANNELS      {{ node.input_shape[0][-1] }}
{% if node.input_shape[0] | length == 3 %}
#define INPUT_SAMPLES       {{ node.input_shape[0][-2] }}
{% elif node.input_shape[0] | length == 4 %}
#define INPUT_HEIGHT        {{ node.input_shape[0][-3] }}
#define INPUT_WIDTH         {{ node.input_shape[0][-2] }}
{% endif %}

// For fixed point quantization
#define INPUT_SCALE_FACTOR {{ node.innodes[0].q.output_scale_factor }}
#define OUTPUT_SCALE_FACTOR {{ node.q.output_scale_factor }}
#define OUTPUT_ROUND_MODE ROUND_MODE_{{ node.q.output_round_mode | upper }}
#define NUMBER_T {{ qtype2ctype(node.q.number_type, node.q.width) }}
#define LONG_NUMBER_T {{ qtype2ctype(node.q.number_type, node.q.long_width) }}

static inline void {{ node.layer.name }}(
{% if node.input_shape[0] | length == 3 %}
  const NUMBER_T input[INPUT_SAMPLES][INPUT_CHANNELS], 			      // IN
{% elif node.input_shape[0] | length == 4 %}
  const NUMBER_T input[INPUT_HEIGHT][INPUT_WIDTH][INPUT_CHANNELS],               // IN
{% endif %}
  {{ node.layer.name }}_output_type output) {    // OUT
  //
  size_t x, y, k;
  static LONG_NUMBER_T output_acc[INPUT_CHANNELS];

  for (k = 0; k < INPUT_CHANNELS; k++) {
    output_acc[k] = 0;
  }

{% if node.input_shape[0] | length == 3 %}
  for (x = 0; x < INPUT_SAMPLES; x++) {
{% elif node.input_shape[0] | length == 4 %}
  for (y = 0; y < INPUT_HEIGHT; y++) {
    for (x = 0; x < INPUT_WIDTH; x++) {
{% endif %}
      for (k = 0; k < INPUT_CHANNELS; k++) {
{% if node.input_shape[0] | length == 4 %}
        output_acc[k] += input[y][x][k];
{% elif node.input_shape[0] | length == 3 %}
        output_acc[k] += input[x][k];
{% endif %}
      }
{% if node.input_shape[0] | length == 4 %}
    }
{% endif %}
  }

  for (k = 0; k < INPUT_CHANNELS; k++) {
    output[k] = scale_and_clamp_to(NUMBER_T, output_acc[k], INPUT_SCALE_FACTOR - OUTPUT_SCALE_FACTOR, OUTPUT_ROUND_MODE);
  }
}

{% if node.input_shape[0] | length == 3 %}
#undef INPUT_SAMPLES
{% elif node.input_shape[0] | length == 4 %}
#undef INPUT_WIDTH
#undef INPUT_HEIGHT
{% endif %}
#undef INPUT_CHANNELS
#undef INPUT_SCALE_FACTOR
#undef OUTPUT_SCALE_FACTOR
#undef NUMBER_T
#undef LONG_NUMBER_T
