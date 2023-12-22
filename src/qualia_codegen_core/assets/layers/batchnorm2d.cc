/**
  ******************************************************************************
  * @file    batchnorm2d.cc
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version 2.0
  * @date    26 june 2023
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#ifndef SINGLE_FILE
#include "{{ node.layer.name }}.h"
#include "number.h"
#endif

#define INPUT_CHANNELS      {{ node.input_shape[0][-1] }}
#define INPUT_HEIGHT        {{ node.input_shape[0][-3] }}
#define INPUT_WIDTH         {{ node.input_shape[0][-2] }}
#define ACTIVATION_{{ node.layer.activation.name | upper if node.layer.activation is defined else "LINEAR" }}

// For fixed point quantization
#define WEIGHTS_SCALE_FACTOR {{ node.q.weights_scale_factor }}
#define INPUT_SCALE_FACTOR {{ node.innodes[0].q.output_scale_factor }}
#define OUTPUT_SCALE_FACTOR {{ node.q.output_scale_factor }}
#define OUTPUT_ROUND_MODE ROUND_MODE_{{ node.q.output_round_mode | upper }}
#define NUMBER_T {{ qtype2ctype(node.q.number_type, node.q.width) }}
#define LONG_NUMBER_T {{ qtype2ctype(node.q.number_type, node.q.long_width) }}


static inline void {{ node.layer.name }}(
  const NUMBER_T input[INPUT_HEIGHT][INPUT_WIDTH][INPUT_CHANNELS],  // IN
  const NUMBER_T kernel[INPUT_CHANNELS],                // IN
  const NUMBER_T bias[INPUT_CHANNELS],                  // IN
  {{ node.layer.name }}_output_type output) {                // OUT

  LONG_NUMBER_T tmp;

  for (size_t y = 0; y < INPUT_HEIGHT; y++) {
    for (size_t x = 0; x < INPUT_WIDTH; x++) {
      for (size_t z = 0; z < INPUT_CHANNELS; z++) {
        tmp = scale(NUMBER_T, (LONG_NUMBER_T)bias[z], -INPUT_SCALE_FACTOR, OUTPUT_ROUND_MODE);
        tmp += (LONG_NUMBER_T)input[y][x][z] * (LONG_NUMBER_T)kernel[z];

        // Activation function
#ifdef ACTIVATION_LINEAR
        // Linear (MEANS NONE)
        tmp = scale(NUMBER_T, tmp, INPUT_SCALE_FACTOR + WEIGHTS_SCALE_FACTOR - OUTPUT_SCALE_FACTOR, OUTPUT_ROUND_MODE);
        output[y][x][z] = clamp_to(NUMBER_T, tmp);
#elif defined(ACTIVATION_RELU)
        // ReLU
        if (tmp < 0) {
          output[y][x][z] = 0;
        } else {
          tmp = scale(NUMBER_T, tmp, INPUT_SCALE_FACTOR + WEIGHTS_SCALE_FACTOR - OUTPUT_SCALE_FACTOR, OUTPUT_ROUND_MODE);
          output[y][x][z] = clamp_to(NUMBER_T, tmp);
        }
#endif
      }
    }
  }
}

#undef INPUT_CHANNELS
#undef INPUT_HEIGHT
#undef INPUT_WIDTH
#undef ACTIVATION_{{ node.layer.activation.name | upper if node.layer.activation is defined else "LINEAR" }}
#undef WEIGHTS_SCALE_FACTOR
#undef INPUT_SCALE_FACTOR
#undef OUTPUT_SCALE_FACTOR
