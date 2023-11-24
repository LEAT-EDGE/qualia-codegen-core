/**
  ******************************************************************************
  * @file    conv1d.hh
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version V2.0
  * @date    24 january 2023
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#ifndef _{{ node.layer.name | upper }}_H_
#define _{{ node.layer.name | upper }}_H_

#ifndef SINGLE_FILE
#include "number.h"
#endif

#define INPUT_CHANNELS      {{ node.input_shape[0][-1] }}
#define INPUT_SAMPLES       {{ node.input_shape[0][-2] }}
#define CONV_FILTERS        {{ node.layer.filters }}
#define CONV_KERNEL_SIZE    {{ node.layer.kernel_size[0] }}
#define CONV_STRIDE         {{ node.layer.strides[0] }}
{% if node.layer.padding == 'valid' %}
#define ZEROPADDING_LEFT    0
#define ZEROPADDING_RIGHT   0
{% else %}
#define ZEROPADDING_LEFT    {{ node.layer.padding[0] }}
#define ZEROPADDING_RIGHT   {{ node.layer.padding[1] }}
{% endif %}
#define CONV_OUTSAMPLES     ( ( (INPUT_SAMPLES - CONV_KERNEL_SIZE + ZEROPADDING_LEFT + ZEROPADDING_RIGHT) / CONV_STRIDE ) + 1 )

typedef {{ qtype2ctype(node.q.number_type, node.q.width) }} {{ node.layer.name }}_output_type[CONV_OUTSAMPLES][CONV_FILTERS];

#if 0
void {{ node.layer.name }}(
  const number_t input[INPUT_SAMPLES][INPUT_CHANNELS],                    // IN
  const number_t kernel[CONV_FILTERS][CONV_KERNEL_SIZE][INPUT_CHANNELS],  // IN
{% if node.layer.use_bias %}
  const number_t bias[CONV_FILTERS],						                          // IN
{% endif %}
  number_t output[CONV_OUTSAMPLES][CONV_FILTERS]);                       // OUT
#endif

#undef INPUT_CHANNELS
#undef INPUT_SAMPLES
#undef CONV_FILTERS
#undef CONV_KERNEL_SIZE
#undef CONV_STRIDE
#undef ZEROPADDING_LEFT
#undef ZEROPADDING_RIGHT
#undef CONV_OUTSAMPLES

#endif//_{{ node.layer.name | upper }}_H_
