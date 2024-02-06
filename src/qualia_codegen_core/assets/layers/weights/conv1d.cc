/**
  ******************************************************************************
  * @file    weights/conv1d.cc
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version 1.0.0
  * @date    24 march 2020
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#include <stdint.h>

#define INPUT_CHANNELS    {{ node.input_shape[0][-1] }}
#define CONV_FILTERS      {{ node.layer.filters }}
#define CONV_KERNEL_SIZE  {{ node.layer.kernel_size[0] }}
#define CONV_GROUPS       {{ node.layer.groups }}

{% if node.layer.use_bias %}
const {{ weights.bias.dtype }}  {{ node.layer.name }}_bias[CONV_FILTERS] = {{ weights.bias.data }};
{% endif %}
const {{ weights.kernel.dtype }}  {{ node.layer.name }}_kernel[CONV_FILTERS][CONV_KERNEL_SIZE][INPUT_CHANNELS / CONV_GROUPS] = {{ weights.kernel.data }};

#undef INPUT_CHANNELS
#undef CONV_FILTERS
#undef CONV_KERNEL_SIZE
#undef CONV_GROUPS
