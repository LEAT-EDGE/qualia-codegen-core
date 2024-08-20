/**
  ******************************************************************************
  * @file    slice.hh
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, Université Côte d'Azur, France
  * @version 1.0.0
  * @date    20 august 2024
  * @brief   Slice
  */

#ifndef _{{ node.layer.name | upper }}_H_
#define _{{ node.layer.name | upper }}_H_

#ifndef SINGLE_FILE
#include "number.h"
#endif

typedef {{ qtype2ctype(node.q.number_type, node.q.width) }} {{ node.layer.name }}_output_type{% for dim in node.output_shape[0][1:] %}[{{ dim }}]{% endfor %};

#if 0
void {{ node.layer.name }}(
  const NUMBER_T input{% for dim in node.input_shape[0][1:] %}[{{ dim }}]{% endfor %}, // IN
  {{ node.layer.name }}_output_type output);
#endif

#endif//_{{ node.layer.name | upper }}_H_
