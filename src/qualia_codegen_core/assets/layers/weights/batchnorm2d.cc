/**
  ******************************************************************************
  * @file    weights/batchnorm2d.cc
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version 2.0
  * @date    26 june 2023
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#include <stdint.h>

const {{ weights.bias.dtype }} {{ node.layer.name }}_bias[{{ weights.bias.shape[0] }}] = {{ weights.bias.data }};
const {{ weights.kernel.dtype }} {{ node.layer.name }}_kernel[{{ weights.kernel.shape[0] }}] = {{ weights.kernel.data }};
