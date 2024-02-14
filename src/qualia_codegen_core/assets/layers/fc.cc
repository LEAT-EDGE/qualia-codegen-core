/**
  ******************************************************************************
  * @file    fc.cc
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version 1.0.0
  * @date    24 march 2020
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#ifndef SINGLE_FILE
#include "{{ node.layer.name }}.h"
#include "number.h"
#endif

#ifdef WITH_CMSIS_NN
#include "arm_nnfunctions.h"
#elif defined(WITH_NMSIS_NN)
#include "riscv_nnfunctions.h"
#endif

#define INPUT_SAMPLES {{ node.input_shape[0][-1] }}
#define FC_UNITS {{ node.layer.units }}
#define ACTIVATION_{{ node.layer.activation.name | upper }}

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
  const NUMBER_T input[INPUT_SAMPLES], 			      // IN
	const NUMBER_T kernel[FC_UNITS][INPUT_SAMPLES],  // IN
{% if node.layer.use_bias %}
	const NUMBER_T bias[FC_UNITS],			              // IN
{% endif %}
	NUMBER_T output[FC_UNITS]) {			                // OUT

#if !defined(WITH_CMSIS_NN) && !defined(WITH_NMSIS_NN)
  unsigned short k, z; 
  LONG_NUMBER_T output_acc;

  for (k = 0; k < FC_UNITS; k++) { 
    output_acc = 0;
    for (z = 0; z < INPUT_SAMPLES; z++) 
      output_acc = output_acc + ((LONG_NUMBER_T)kernel[k][z] * (LONG_NUMBER_T)input[z]);

    output_acc = scale(NUMBER_T, output_acc, WEIGHTS_SCALE_FACTOR - TMP_SCALE_FACTOR, OUTPUT_ROUND_MODE);
{% if node.layer.use_bias %}
    output_acc += scale(NUMBER_T, (LONG_NUMBER_T)bias[k], BIASES_SCALE_FACTOR - TMP_SCALE_FACTOR - INPUT_SCALE_FACTOR, OUTPUT_ROUND_MODE);
{% endif %}

    // Activation function
#ifdef ACTIVATION_LINEAR
    // Linear (MEANS NONE)
    output[k] = scale_and_clamp_to(NUMBER_T, output_acc, INPUT_SCALE_FACTOR + TMP_SCALE_FACTOR - OUTPUT_SCALE_FACTOR, OUTPUT_ROUND_MODE);
#elif defined(ACTIVATION_RELU) || defined(ACTIVATION_RELU6)
    // ReLU
    if (output_acc < 0) {
      output[k] = 0;
    } else {
#if defined(ACTIVATION_RELU6)
      if (output_acc > scale(NUMBER_T, 6, -(INPUT_SCALE_FACTOR + TMP_SCALE_FACTOR), OUTPUT_ROUND_MODE)) {
        output_acc = scale(NUMBER_T, 6, -(INPUT_SCALE_FACTOR + TMP_SCALE_FACTOR), OUTPUT_ROUND_MODE);
      }
#endif
      output[k] = scale_and_clamp_to(NUMBER_T, output_acc, INPUT_SCALE_FACTOR + TMP_SCALE_FACTOR - OUTPUT_SCALE_FACTOR, OUTPUT_ROUND_MODE);
    }
#else
#error "Unsupported activation function"
#endif
  }
#else
{% if not node.layer.use_bias %}
#error "CMSIS-NN requires the use of bias"
{% endif %}
#if BIASES_SCALE_FACTOR > WEIGHTS_SCALE_FACTOR
#error "CMSIS-NN does not support BIASES_SCALE_FACTOR larger than WEIGHTS_SCALE_FACTOR"
#endif
{% if qtype2ctype(node.q.number_type, node.q.width) == 'int8_t' %}
  static q15_t bufferA[INPUT_SAMPLES];
#ifdef WITH_CMSIS_NN
  arm_fully_connected_q7(
#elif defined(WITH_NMSIS_NN)
  riscv_fully_connected_q7(
#endif
                             (q7_t*)input,
                             (q7_t*)kernel,
                             INPUT_SAMPLES,
                             FC_UNITS,
                             INPUT_SCALE_FACTOR + WEIGHTS_SCALE_FACTOR - BIASES_SCALE_FACTOR, //bias_shift
                             INPUT_SCALE_FACTOR + WEIGHTS_SCALE_FACTOR - OUTPUT_SCALE_FACTOR,
                             (q7_t*)bias,
                             (q7_t*)output,
                             (q15_t*)bufferA);
#ifdef ACTIVATION_RELU
#ifdef WITH_CMSIS_NN
  arm_relu_q7((q7_t*)output, FC_UNITS);
#elif defined(WITH_NMSIS_NN)
  riscv_relu_q7((q7_t*)output, FC_UNITS);
#endif
#elif !defined(ACTIVATION_LINEAR)
#error "Unsupported activation with CMSIS-NN"
#endif

{% elif qtype2ctype(node.q.number_type, node.q.width) == 'int16_t' %}
  static q15_t bufferA[INPUT_SAMPLES];
#ifdef WITH_CMSIS_NN
  arm_fully_connected_q15(
#elif defined(WITH_NMSIS_NN)
  riscv_fully_connected_q15(
#endif
                             (q15_t*)input,
                             (q15_t*)kernel,
                             INPUT_SAMPLES,
                             FC_UNITS,
                             INPUT_SCALE_FACTOR + WEIGHTS_SCALE_FACTOR - BIASES_SCALE_FACTOR,
                             INPUT_SCALE_FACTOR + WEIGHTS_SCALE_FACTOR - OUTPUT_SCALE_FACTOR,
                             (q15_t*)bias,
                             (q15_t*)output,
                             (q15_t*)bufferA);
#ifdef ACTIVATION_RELU
#ifdef WITH_CMSIS_NN
  arm_relu_q15((q15_t*)output, FC_UNITS);
#elif defined(WITH_NMSIS_NN)
  riscv_relu_q15((q15_t*)output, FC_UNITS);
#endif
#elif !defined(ACTIVATION_LINEAR)
#error "Unsupported activation with CMSIS-NN"
#endif

{% else %}
#error "Data type unsupported by CMSIS-NN"
{% endif %}
#endif
}

#undef INPUT_SAMPLES
#undef FC_UNITS
#undef ACTIVATION_{{ node.layer.activation.name | upper }}
#undef WEIGHTS_SCALE_FACTOR
#undef BIASES_SCALE_FACTOR
#undef INPUT_SCALE_FACTOR
#undef OUTPUT_SCALE_FACTOR
#undef NUMBER_T
#undef LONG_NUMBER_T
