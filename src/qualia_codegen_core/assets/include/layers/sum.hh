/**
  ******************************************************************************
  * @file    sum.hh
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version V2.0
  * @date    7 february 2023
  * @brief   Global Sum Pooling
  */

#ifndef _{{ node.layer.name | upper }}_H_
#define _{{ node.layer.name | upper }}_H_

#ifndef SINGLE_FILE
#include "number.h"
#endif

#define INPUT_CHANNELS      {{ node.input_shape[0][-1] }}
{% if node.input_shape[0] | length == 3 %}
#define INPUT_SAMPLES       {{ node.input_shape[0][-2] }}
{% elif node.input_shape[0] | length == 4 %}
#define INPUT_HEIGHT        {{ node.input_shape[0][-3] }}
#define INPUT_WIDTH         {{ node.input_shape[0][-2] }}
{% endif %}

typedef {{ qtype2ctype(node.q.number_type, node.q.width) }} {{ node.layer.name }}_output_type[INPUT_CHANNELS];

#if 0
void {{ node.layer.name }}(
{% if node.input_shape[0] | length == 3 %}
  const NUMBER_T input[INPUT_SAMPLES][INPUT_CHANNELS], 			      // IN
{% elif node.input_shape[0] | length == 4 %}
  const NUMBER_T input[INPUT_HEIGHT][INPUT_WIDTH][INPUT_CHANNELS],               // IN
{% endif %}
  {{ node.layer.name }}_output_type vector_out);     // OUT
#endif

{% if node.input_shape[0] | length == 3 %}
#undef INPUT_SAMPLES
{% elif node.input_shape[0] | length == 4 %}
#undef INPUT_WIDTH
#undef INPUT_HEIGHT
{% endif %}
#undef INPUT_CHANNELS

#endif//_{{ node.layer.name | upper }}_H_
