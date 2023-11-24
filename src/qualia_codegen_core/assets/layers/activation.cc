/**
  ******************************************************************************
  * @file    activation.cc
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version 1.0.0
  * @date    24 march 2020
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#include <math.h>

#ifndef SINGLE_FILE
#include "{{ node.layer.name }}.h"
#include "number.h"
#endif

#define SAMPLES {{ node.input_shape[0][-1] }}
#define ACTIVATION_{{ node.layer.activation.name | upper }}

// For fixed point quantization
#define INPUT_SCALE_FACTOR {{ node.innodes[0].q.output_scale_factor }}
#define NUMBER_T {{ qtype2ctype(node.q.number_type, node.q.width) }}


#ifdef ACTIVATION_SOFTMAX
static inline void {{ node.layer.name }}(
  const NUMBER_T vector_in[SAMPLES], // INT
  float vector_out[SAMPLES]) {    // OUT

  unsigned short k; 
  double sum; 

  sum = 0; 
  for (k=0; k < SAMPLES; k++) 
    sum = sum + exp( (double)vector_in[k] / (1<<INPUT_SCALE_FACTOR) ); 

  for (k=0; k < SAMPLES; k++) 
    vector_out[k] = exp( (double)( (double)vector_in[k] / (1<<INPUT_SCALE_FACTOR) ) ) / sum; 

}
#endif

#undef SAMPLES
#undef ACTIVATION_{{ node.layer.activation.name | upper }}
#undef INPUT_SCALE_FACTOR
