/**
  ******************************************************************************
  * @file    fc.hh
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version V2.0
  * @date    24 january 2023
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#ifndef _{{ node.layer.name | upper }}_H_
#define _{{ node.layer.name | upper }}_H_

#ifndef SINGLE_FILE
#include "number.h"
#include <stdint.h>
#endif

#define INPUT_SAMPLES {{ node.input_shape[0][-1] }}
#define FC_UNITS {{ node.layer.units }}

typedef {{ qtype2ctype(node.q.number_type, node.q.width) }} {{ node.layer.name }}_output_type[FC_UNITS];

#if 0
void {{ node.layer.name }}(
  const number_t input[INPUT_SAMPLES], 			      // IN
	const number_t kernel[FC_UNITS][INPUT_SAMPLES],  // IN
{% if node.layer.use_bias %}
	const number_t bias[FC_UNITS],			              // IN
{% endif %}
	number_t output[FC_UNITS]); 			                // OUT
#endif

#undef INPUT_SAMPLES
#undef FC_UNITS

#endif//_{{ node.layer.name | upper }}_H_
