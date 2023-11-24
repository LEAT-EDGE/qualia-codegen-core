/**
  ******************************************************************************
  * @file    activation.hh
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version V2.0
  * @date    21 april 2023
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#ifndef _{{ node.layer.name | upper }}_H_
#define _{{ node.layer.name | upper }}_H_

#ifndef SINGLE_FILE
#include "number.h"
#include <stdint.h>
#endif

#define SAMPLES {{ node.input_shape[0][-1] }}

typedef float {{ node.layer.name }}_output_type[SAMPLES];

#if 0
#ifdef ACTIVATION_SOFTMAX
void {{ node.layer.name }}(
  const number_t vector_in[SAMPLES], // INT
  float vector_out[SAMPLES]);    // OUT
#endif
#endif

#undef SAMPLES

#endif//_{{ node.layer.name | upper }}_H_
