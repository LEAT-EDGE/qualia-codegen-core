/**
  ******************************************************************************
  * @file    maxpool2d.cc
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version V2.0
  * @date    24 november 2021
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#ifndef SINGLE_FILE
#include "{{ node.layer.name }}.h"
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

#define ACTIVATION_{{ node.layer.activation.name | upper if node.layer.activation is defined else "LINEAR" }}

// For fixed point quantization
#define INPUT_SCALE_FACTOR {{ node.innodes[0].q.output_scale_factor }}
#define OUTPUT_SCALE_FACTOR {{ node.q.output_scale_factor }}
#define OUTPUT_ROUND_MODE ROUND_MODE_{{ node.q.output_round_mode | upper }}
#define NUMBER_T {{ qtype2ctype(node.q.number_type, node.q.width) }}
#define LONG_NUMBER_T {{ qtype2ctype(node.q.number_type, node.q.long_width) }}


static inline void {{ node.layer.name }}(
  const NUMBER_T input[INPUT_HEIGHT][INPUT_WIDTH][INPUT_CHANNELS], 	    // IN
  NUMBER_T output[POOL_HEIGHT][POOL_WIDTH][INPUT_CHANNELS]) {	// OUT

  unsigned short pos_x, pos_y, k; 	// loop indexes for output volume
  unsigned int x, y;
  LONG_NUMBER_T max, tmp;

  for (k = 0; k < INPUT_CHANNELS; k++) 
    for (pos_y = 0; pos_y < POOL_HEIGHT; pos_y++) {
      for (pos_x = 0; pos_x < POOL_WIDTH; pos_x++) {
#ifdef ACTIVATION_LINEAR
        max = input[pos_y*POOL_STRIDE_Y][pos_x*POOL_STRIDE_X][k];
        x = 1;
#elif defined(ACTIVATION_RELU)
        max = 0;
        x = 0;
#else
#error "Unsupported activation function"
#endif
        for (y = 0; y < POOL_SIZE_Y; y++) {
          for (; x < POOL_SIZE_X; x++) {
            tmp = input[(pos_y*POOL_STRIDE_Y)+y][(pos_x*POOL_STRIDE_X)+x][k];
            if (max < tmp)
              max = tmp;
          }
          x = 0;
        }

        output[pos_y][pos_x][k] = scale_and_clamp_to(NUMBER_T, max, INPUT_SCALE_FACTOR - OUTPUT_SCALE_FACTOR, OUTPUT_ROUND_MODE);
      }
    }
}

#undef INPUT_CHANNELS  
#undef INPUT_WIDTH
#undef INPUT_HEIGHT
#undef POOL_SIZE_X
#undef POOL_SIZE_Y
#undef POOL_STRIDE_X
#undef POOL_STRIDE_Y
#undef POOL_PAD_X
#undef POOL_PAD_Y
#undef POOL_WIDTH
#undef POOL_HEIGHT
#undef ACTIVATION_{{ node.layer.activation.name | upper if node.layer.activation is defined else "LINEAR" }}
#undef INPUT_SCALE_FACTOR
#undef OUTPUT_SCALE_FACTOR
