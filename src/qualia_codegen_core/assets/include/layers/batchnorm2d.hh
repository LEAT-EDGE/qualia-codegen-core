/**
  ******************************************************************************
  * @file    batchnorm2d.hh
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version V2.0
  * @date    26 june 2023
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

typedef {{ qtype2ctype(node.q.number_type, node.q.width) }} {{ node.layer.name }}_output_type[INPUT_HEIGHT][INPUT_WIDTH][INPUT_CHANNELS];

#if 0
void {{ node.layer.name }}(
  const number_t input[INPUT_HEIGHT][INPUT_WIDTH][INPUT_CHANNELS],  // IN
  const number_t kernel[INPUT_CHANNELS],                // IN
  const number_t bias[INPUT_CHANNELS],                  // IN
  {{ node.layer.name }}_output_type output);                // OUT
#endif

#undef INPUT_CHANNELS
#undef INPUT_HEIGHT
#undef INPUT_WIDTH

#endif//_{{ node.layer.name | upper }}_H_
