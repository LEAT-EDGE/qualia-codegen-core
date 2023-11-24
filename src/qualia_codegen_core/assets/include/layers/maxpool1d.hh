/**
  ******************************************************************************
  * @file    maxpool1d.hh
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

#define INPUT_CHANNELS  {{ node.input_shape[0][-1] }}
#define INPUT_SAMPLES   {{ node.input_shape[0][-2] }}
#define POOL_SIZE       {{ node.layer.pool_size[0] }}
#define POOL_STRIDE     {{ node.layer.strides[0] }}
#define POOL_PAD        0 // Unsupported
#define POOL_LENGTH	    ( ( (INPUT_SAMPLES - POOL_SIZE + (2*POOL_PAD) ) / POOL_STRIDE ) + 1 )

typedef {{ qtype2ctype(node.q.number_type, node.q.width) }} {{ node.layer.name }}_output_type[POOL_LENGTH][INPUT_CHANNELS];

#if 0
void {{ node.layer.name }}(
  const number_t input[INPUT_SAMPLES][INPUT_CHANNELS], 	    // IN
  number_t output[POOL_LENGTH][INPUT_CHANNELS]); 	// OUT
#endif

#undef INPUT_CHANNELS  
#undef INPUT_SAMPLES
#undef POOL_SIZE
#undef POOL_STRIDE
#undef POOL_PAD
#undef POOL_LENGTH

#endif//_{{ node.layer.name | upper }}_H_
