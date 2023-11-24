/**
  ******************************************************************************
  * @file    operator.hh
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

typedef {{ qtype2ctype(node.q.number_type, node.q.width) }} {{ node.layer.name }}_output_type{% for dim in node.output_shape[0][1:] %}[{{ dim }}]{% endfor %};

#if 0
void {{ node.layer.name }}(
{% for innode in node.innodes %}
  const number_t vector_in_{{ loop.index }}{% for dim in node.input_shape[loop.index - 1][1:] %}[{{ dim }}]{% endfor %}, // doesn't work with inverted data_format
{% endfor %}
  {{ node.layer.name }}_output_type vector_out);     // OUT
#endif

#endif//_{{ node.layer.name | upper }}_H_
