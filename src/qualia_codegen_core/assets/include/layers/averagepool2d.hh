/**
  ******************************************************************************
  * @file    averagepool2d.hh
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version V2.0
  * @date    21 april 2023
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#ifndef _{{ node.layer.name | upper }}_H_
#define _{{ node.layer.name | upper }}_H_

#ifndef SINGLE_FILE
#include "number.h"
#endif

#define INPUT_CHANNELS  {{ node.input_shape[0][-1] }}
#define INPUT_HEIGHT    {{ node.input_shape[0][-3] }}
#define INPUT_WIDTH     {{ node.input_shape[0][-2] }}
#define POOL_SIZE_Y     {{ node.layer.pool_size[0] }}
#define POOL_SIZE_X     {{ node.layer.pool_size[1] if node.layer.pool_size | length > 1 else node.layer.pool_size[0] }}
#define POOL_STRIDE_Y   {{ node.layer.strides[0] }}
#define POOL_STRIDE_X   {{ node.layer.strides[1] if node.layer.strides | length > 1 else node.layer.strides[0] }}
#define POOL_PAD_Y      0 // Unsupported
#define POOL_PAD_X      0 // Unsupported
#define POOL_HEIGHT	    ( ( (INPUT_HEIGHT - POOL_SIZE_Y + (2*POOL_PAD_Y) ) / POOL_STRIDE_Y ) + 1 )
#define POOL_WIDTH	    ( ( (INPUT_WIDTH - POOL_SIZE_X + (2*POOL_PAD_X) ) / POOL_STRIDE_X ) + 1 )

typedef {{ qtype2ctype(node.q.number_type, node.q.width) }} {{ node.layer.name }}_output_type[POOL_HEIGHT][POOL_WIDTH][INPUT_CHANNELS];

#if 0
void {{ node.layer.name }}(
  const number_t input[INPUT_HEIGHT][INPUT_WIDTH][INPUT_CHANNELS], 	    // IN
  number_t output[POOL_HEIGHT][POOL_WIDTH][INPUT_CHANNELS]);	// OUT
#endif

#undef INPUT_CHANNELS 
#undef INPUT_WIDTH
#undef INPUT_HEIGHT
#undef POOL_SIZE_X
#undef POOL_SIZE_Y
#undef POOL_STRIDE_X
#undef POOL_STRIDE_Y
#undef POOL_PAD_X
#undef POOL_PAD_Y
#undef POOL_HEIGHT
#undef POOL_WIDTH

#endif//_{{ node.layer.name | upper }}_H_
