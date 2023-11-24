/**
  ******************************************************************************
  * @file    weights/batchnorm1d.cc
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version 1.0.0
  * @date    24 august 2020
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#include <stdint.h>

const {{ weights.bias.dtype }} {{ node.layer.name }}_bias[{{ weights.bias.shape[0] }}] = {{ weights.bias.data }};
const {{ weights.kernel.dtype }} {{ node.layer.name }}_kernel[{{ weights.kernel.shape[0] }}] = {{ weights.kernel.data }};
