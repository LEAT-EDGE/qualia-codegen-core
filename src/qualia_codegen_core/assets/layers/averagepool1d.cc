/**
  ******************************************************************************
  * @file    averagepool.cc
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version 1.0.0
  * @date    24 march 2020
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#ifndef SINGLE_FILE
#include "{{ node.layer.name }}.h"
#include "number.h"
#endif

#define INPUT_CHANNELS  {{ node.input_shape[0][-1] }}
#define INPUT_SAMPLES   {{ node.input_shape[0][-2] }}
#define POOL_SIZE       {{ node.layer.pool_size[0] }}
#define POOL_STRIDE     {{ node.layer.strides[0] }}
#define POOL_PAD        0 // Unsupported
#define POOL_LENGTH	    ( ( (INPUT_SAMPLES - POOL_SIZE + (2*POOL_PAD) ) / POOL_STRIDE ) + 1 )

#define ACTIVATION_{{ node.layer.activation.name | upper if node.layer.activation is defined else "LINEAR" }}

// For fixed point quantization
#define INPUT_SCALE_FACTOR {{ node.innodes[0].q.output_scale_factor }}
#define OUTPUT_SCALE_FACTOR {{ node.q.output_scale_factor }}
#define NUMBER_T {{ qtype2ctype(node.q.number_type, node.q.width) }}
#define LONG_NUMBER_T {{ qtype2ctype(node.q.number_type, node.q.long_width) }}


static inline void {{ node.layer.name }}(
  const NUMBER_T input[INPUT_SAMPLES][INPUT_CHANNELS], 	    // IN
  NUMBER_T output[POOL_LENGTH][INPUT_CHANNELS]) {	// OUT

  unsigned short pos_x, k; 	// loop indexes for output volume
  unsigned int x;
  LONG_NUMBER_T avg, tmp;

  for (k = 0; k < INPUT_CHANNELS; k++) 
    for (pos_x = 0; pos_x < POOL_LENGTH; pos_x++) {
      tmp = 0;
      for (x = 0; x < POOL_SIZE; x++) {
        tmp += input[(pos_x*POOL_STRIDE)+x][k];
      }
#ifdef ACTIVATION_RELU
      if (tmp < 0) {
        tmp = 0;
      }
#endif
      avg = tmp / POOL_SIZE;
      avg = scale(NUMBER_T, avg, INPUT_SCALE_FACTOR - OUTPUT_SCALE_FACTOR);
      output[pos_x][k] = clamp_to(NUMBER_T, avg);
    }
}

#undef INPUT_CHANNELS  
#undef INPUT_SAMPLES
#undef POOL_SIZE
#undef POOL_STRIDE
#undef POOL_PAD
#undef POOL_LENGTH
#undef ACTIVATION_{{ node.layer.activation.name | upper if node.layer.activation is defined else "LINEAR" }}
#undef INPUT_SCALE_FACTOR
#undef OUTPUT_SCALE_FACTOR
