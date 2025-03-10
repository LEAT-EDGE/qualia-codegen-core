/**
  ******************************************************************************
  * @file    upsample.cc
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, Université Côte d'Azur, France
  * @version 1.0.0
  * @date    7 march 2025
  * @brief   Upsample
  */

#ifndef SINGLE_FILE
#include "{{ node.layer.name }}.h"
#include "number.h"
#endif

#define INPUT_CHANNELS      {{ node.input_shape[0][-1] }}
{% if node.input_shape[0] | length == 3 %}
#define INPUT_SAMPLES       {{ node.input_shape[0][-2] }}
#define UPSAMPLE_SCALE_SAMPLES  {{ node.layer.scale_factor[0] }}
#define OUTPUT_SAMPLES        ( INPUT_SAMPLES * UPSAMPLE_SCALE_SAMPLES )
{% elif node.input_shape[0] | length == 4 %}
#define INPUT_HEIGHT        {{ node.input_shape[0][-3] }}
#define INPUT_WIDTH         {{ node.input_shape[0][-2] }}
#define UPSAMPLE_SCALE_HEIGHT {{ node.layer.scale_factor[1] }}
#define UPSAMPLE_SCALE_WIDTH  {{ node.layer.scale_factor[0] }}
#define OUTPUT_HEIGHT        ( INPUT_HEIGHT * UPSAMPLE_SCALE_HEIGHT )
#define OUTPUT_WIDTH         ( INPUT_WIDTH * UPSAMPLE_SCALE_WIDTH )
{% endif %}

#define MODE_{{ node.layer.mode.name | upper }}

#define NUMBER_T {{ qtype2ctype(node.q.number_type, node.q.width) }}

static inline void {{ node.layer.name }}(
{% if node.input_shape[0] | length == 3 %}
  const NUMBER_T input[INPUT_SAMPLES][INPUT_CHANNELS], 			      // IN
{% elif node.input_shape[0] | length == 4 %}
  const NUMBER_T input[INPUT_HEIGHT][INPUT_WIDTH][INPUT_CHANNELS],               // IN
{% endif %}
  {{ node.layer.name }}_output_type output) {    // OUT
  //
  size_t x, y, k;

#ifdef MODE_NEAREST
{% if node.input_shape[0] | length == 3 %}
  for (x = 0; x < OUTPUT_SAMPLES; x++) {
    for (k = 0; k < INPUT_CHANNELS; k++) {
        output[x][k] = input[x / UPSAMPLE_SCALE_SAMPLES][k];
    }
  }
{% elif node.input_shape[0] | length == 4 %}
  for (y = 0; y < OUTPUT_HEIGHT; y++) {
    for (x = 0; x < OUTPUT_WIDTH; x++) {
      for (k = 0; k < INPUT_CHANNELS; k++) {
        output[y][x][k] = input[y / UPSAMPLE_SCALE_HEIGHT][x / UPSAMPLE_SCALE_WIDTH][k];
      }
    }
  }
{% else %}
#error "Unsupported number of dimensions"
{% endif %}
#else
#error "Unsupported mode {{ node.layer.mode.name }}"
#endif
}

{% if node.input_shape[0] | length == 3 %}
#undef INPUT_SAMPLES
#undef UPSAMPLE_SCALE_SAMPLES
#undef OUTPUT_SAMPLES
{% elif node.input_shape[0] | length == 4 %}
#undef INPUT_WIDTH
#undef INPUT_HEIGHT
#undef UPSAMPLE_SCALE_WIDTH
#undef UPSAMPLE_SCALE_HEIGHT
#undef OUTPUT_WIDTH
#undef OUTPUT_HEIGHT
{% endif %}
#undef INPUT_CHANNELS
#undef MODE_{{ node.layer.mode.name | upper }}
#undef NUMBER_T
