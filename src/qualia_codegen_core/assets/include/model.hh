/**
  ******************************************************************************
  * @file    model.hh
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version 1.0.0
  * @date    08 july 2020
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */


#ifdef __cplusplus
extern "C" {
#endif

#ifndef __MODEL_H__
#define __MODEL_H__

#ifndef SINGLE_FILE
#include "number.h"

{% for node in nodes[1:] %} // InputLayer is excluded
#include "{{ node.layer.name }}.h"
{%- endfor %}
#endif

{% for dim in nodes[0].output_shape[0][1:] %}
#define MODEL_INPUT_DIM_{{ loop.index - 1 }} {{ dim }}
{%- endfor %}
#define MODEL_INPUT_DIMS {{ nodes[0].output_shape[0][1:] | join(' * ') }}

#define MODEL_OUTPUT_SAMPLES {{ nodes[-1].output_shape[0][-1] }}

#define MODEL_INPUT_SCALE_FACTOR {{ nodes[0].q.output_scale_factor }} // scale factor of InputLayer
#define MODEL_INPUT_ROUND_MODE ROUND_MODE_{{ nodes[0].q.output_round_mode | upper }}
#define MODEL_INPUT_NUMBER_T {{ qtype2ctype(nodes[0].q.number_type, nodes[0].q.width) }}
#define MODEL_INPUT_LONG_NUMBER_T {{ qtype2ctype(nodes[0].q.number_type, nodes[0].q.long_width) }}

// node 0 is InputLayer so use its output shape as input shape of the model
// typedef {{ number_type }} input_t{% for dim in nodes[0].output_shape[0][1:] %}[{{ dim }}]{% endfor %};
typedef {{ qtype2ctype(nodes[0].q.number_type, nodes[0].q.width) }} input_t{% for dim in nodes[0].output_shape[0][1:] %}[{{ dim }}]{% endfor %};
typedef {{ nodes[-1].layer.name }}_output_type output_t;


void cnn(
  const input_t input,
  output_t output);

void reset(void);

#endif//__MODEL_H__


#ifdef __cplusplus
} // extern "C"
#endif
