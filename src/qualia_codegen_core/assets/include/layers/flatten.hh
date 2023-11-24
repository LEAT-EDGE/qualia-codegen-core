/**
  ******************************************************************************
  * @file    flatten.hh
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

#define OUTPUT_DIM {{ node.output_shape[0][-1] }}

typedef {{ qtype2ctype(node.q.number_type, node.q.width) }} {{ node.layer.name }}_output_type[OUTPUT_DIM];

#if 0
void {{ node.layer.name }}(
  const number_t input{% for dim in node.input_shape[0][1:] %}[{{ dim }}]{% endfor %}, 			      // IN
	number_t output[OUTPUT_DIM]); 			                // OUT
#endif

#undef OUTPUT_DIM

#endif//_{{ node.layer.name | upper }}_H_
