/**
  ******************************************************************************
  * @file    batchnorm1d.cc
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version 1.0.0
  * @date    24 august 2020
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#ifndef SINGLE_FILE
#include "{{ node.layer.name }}.h"
#include "number.h"
#endif

#define INPUT_CHANNELS      {{ node.input_shape[0][-1] }}
#define INPUT_SAMPLES       {{ node.input_shape[0][1] }}
#define ACTIVATION_{{ node.layer.activation.name | upper if node.layer.activation is defined else "LINEAR" }}

// For fixed point quantization
#define WEIGHTS_SCALE_FACTOR {{ node.q.weights_scale_factor }}
#define BIASES_SCALE_FACTOR {{ node.q.bias_scale_factor if node.q.bias_scale_factor is not none else node.q.weights_scale_factor }}
#define TMP_SCALE_FACTOR {{ [node.q.weights_scale_factor, node.q.bias_scale_factor] | max if node.q.bias_scale_factor is not none else node.q.weights_scale_factor }}
#define INPUT_SCALE_FACTOR {{ node.innodes[0].q.output_scale_factor }}
#define OUTPUT_SCALE_FACTOR {{ node.q.output_scale_factor }}
#define OUTPUT_ROUND_MODE ROUND_MODE_{{ node.q.output_round_mode | upper }}
#define NUMBER_T {{ qtype2ctype(node.q.number_type, node.q.width) }}
#define LONG_NUMBER_T {{ qtype2ctype(node.q.number_type, node.q.long_width) }}


static inline void {{ node.layer.name }}(
  const NUMBER_T input[INPUT_SAMPLES][INPUT_CHANNELS],  // IN
  const NUMBER_T kernel[INPUT_CHANNELS],                // IN
  const NUMBER_T bias[INPUT_CHANNELS],                  // IN
  {{ node.layer.name }}_output_type output) {                // OUT

  LONG_NUMBER_T tmp;

  for (int x = 0; x < INPUT_SAMPLES; x++) {
    for (int z = 0; z < INPUT_CHANNELS; z++) {
      tmp = (LONG_NUMBER_T)input[x][z] * (LONG_NUMBER_T)kernel[z];

      // Scale for possible additional precision of bias
      tmp = scale(NUMBER_T, tmp, WEIGHTS_SCALE_FACTOR - TMP_SCALE_FACTOR, OUTPUT_ROUND_MODE);
      // Scale bias to match accumulator
      tmp += scale(NUMBER_T, (LONG_NUMBER_T)bias[z], BIASES_SCALE_FACTOR - TMP_SCALE_FACTOR - INPUT_SCALE_FACTOR, OUTPUT_ROUND_MODE);

      // Activation function
#ifdef ACTIVATION_LINEAR
      // Linear (MEANS NONE)
      output[x][z] = scale_and_clamp_to(NUMBER_T, tmp, INPUT_SCALE_FACTOR + TMP_SCALE_FACTOR - OUTPUT_SCALE_FACTOR, OUTPUT_ROUND_MODE);
#elif defined(ACTIVATION_RELU) || defined(ACTIVATION_RELU6)
      // ReLU
      if (tmp < 0) {
        output[x][z] = 0;
      } else {
#if defined(ACTIVATION_RELU6)
        if (tmp > scale(NUMBER_T, 6, -(INPUT_SCALE_FACTOR + TMP_SCALE_FACTOR), OUTPUT_ROUND_MODE)) {
          tmp = scale(NUMBER_T, 6, -(INPUT_SCALE_FACTOR + TMP_SCALE_FACTOR), OUTPUT_ROUND_MODE);
        }
#endif
        output[x][z] = scale_and_clamp_to(NUMBER_T, tmp, INPUT_SCALE_FACTOR + TMP_SCALE_FACTOR - OUTPUT_SCALE_FACTOR, OUTPUT_ROUND_MODE);
      }
#else
#error "Unsupported activation function"
#endif
    }
  }
}

#undef INPUT_CHANNELS
#undef INPUT_SAMPLES
#undef ACTIVATION_{{ node.layer.activation.name | upper if node.layer.activation is defined else "LINEAR" }}
#undef WEIGHTS_SCALE_FACTOR
#undef BIASES_SCALE_FACTOR
#undef TMP_SCALE_FACTOR
#undef INPUT_SCALE_FACTOR
#undef OUTPUT_SCALE_FACTOR
