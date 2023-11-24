/**
  ******************************************************************************
  * @file    weights/fc.cc
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version 1.0.0
  * @date    24 march 2020
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#include <stdint.h>

#define INPUT_SAMPLES {{ node.input_shape[0][-1] }}
#define FC_UNITS {{ node.layer.units }}

{% if node.layer.use_bias %}
const {{ weights.bias.dtype }} {{ node.layer.name }}_bias[FC_UNITS] = {{ weights.bias.data }};
{% endif %}
const {{ weights.kernel.dtype }} {{ node.layer.name }}_kernel[FC_UNITS][INPUT_SAMPLES] = {{ weights.kernel.data }};

#undef INPUT_SAMPLES
#undef FC_UNITS
