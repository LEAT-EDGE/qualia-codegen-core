/**
  ******************************************************************************
  * @file    conv2d.hh
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version V2.0
  * @date    14 december 2022
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#ifndef _{{ node.layer.name | upper }}_H_
#define _{{ node.layer.name | upper }}_H_

#ifndef SINGLE_FILE
#include "number.h"
#endif

#define INPUT_CHANNELS      {{ node.input_shape[0][-1] }}
#define INPUT_HEIGHT        {{ node.input_shape[0][-3] }}
#define INPUT_WIDTH         {{ node.input_shape[0][-2] }}
#define CONV_FILTERS        {{ node.layer.filters }}
#define CONV_KERNEL_SIZE_Y  {{ node.layer.kernel_size[0] }}
#define CONV_KERNEL_SIZE_X  {{ node.layer.kernel_size[1] }}
#define CONV_STRIDE_Y       {{ node.layer.strides[0] }}
#define CONV_STRIDE_X       {{ node.layer.strides[1] }}
{% if node.layer.padding == 'valid' %}
#define ZEROPADDING_TOP     0
#define ZEROPADDING_BOTTOM  0
#define ZEROPADDING_LEFT    0
#define ZEROPADDING_RIGHT   0
{% else %}
#define ZEROPADDING_TOP    {{ node.layer.padding[0][0] }}
#define ZEROPADDING_BOTTOM {{ node.layer.padding[0][1] }}
#define ZEROPADDING_LEFT   {{ node.layer.padding[1][0] }}
#define ZEROPADDING_RIGHT  {{ node.layer.padding[1][1] }}
{% endif %}
#define CONV_OUTHEIGHT     ( ( (INPUT_HEIGHT - CONV_KERNEL_SIZE_Y + ZEROPADDING_TOP + ZEROPADDING_BOTTOM) / CONV_STRIDE_Y ) + 1 )
#define CONV_OUTWIDTH      ( ( (INPUT_WIDTH - CONV_KERNEL_SIZE_X + ZEROPADDING_LEFT + ZEROPADDING_RIGHT) / CONV_STRIDE_X ) + 1 )


typedef {{ qtype2ctype(node.q.number_type, node.q.width) }} {{ node.layer.name }}_output_type[CONV_OUTHEIGHT][CONV_OUTWIDTH][CONV_FILTERS];

#if 0
void {{ node.layer.name }}(
  const number_t input[INPUT_HEIGHT][INPUT_WIDTH][INPUT_CHANNELS],               // IN
  const number_t kernel[CONV_FILTERS][CONV_KERNEL_SIZE_X][CONV_KERNEL_SIZE_Y][INPUT_CHANNELS], // IN
{% if node.layer.use_bias %}
  const number_t bias[CONV_FILTERS],						                // IN
{% endif %}
  number_t output[CONV_OUTHEIGHT][CONV_OUTWIDTH][CONV_FILTERS]);               // OUT
#endif

#undef INPUT_CHANNELS
#undef INPUT_WIDTH
#undef INPUT_HEIGHT
#undef CONV_FILTERS
#undef CONV_KERNEL_SIZE_X
#undef CONV_KERNEL_SIZE_Y
#undef CONV_STRIDE_X
#undef CONV_STRIDE_Y
#undef ZEROPADDING_TOP
#undef ZEROPADDING_BOTTOM
#undef ZEROPADDING_LEFT
#undef ZEROPADDING_RIGHT
#undef CONV_OUTWIDTH
#undef CONV_OUTHEIGHT

#endif//_{{ node.layer.name | upper }}_H_
