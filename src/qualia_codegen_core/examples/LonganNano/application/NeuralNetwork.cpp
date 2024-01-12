// Copyright 2021 (c) Pierre-Emmanuel Novac <penovac@unice.fr> Université Côte d'Azur, CNRS, LEAT. All rights reserved.

#include "NeuralNetwork.h"

#include <string.h>
#include <cstdint>
#include <cmath>

extern "C" {
#ifdef WITH_NMSIS_NN
#include "riscv_nnfunctions.h"
#endif
#include "full_model.h"
}

unsigned int inference_count = 0;

static float input[MODEL_INPUT_DIMS];

extern "C" {

float *serialBufToFloats(char buf[], size_t buflen) {
  auto *pbuf = buf;

  unsigned int i = 0;
  while ((pbuf - buf) < (int)buflen && *pbuf != '\r' && *pbuf != '\n') {
    input[i] = strtof(pbuf, &pbuf);
    i++;
    pbuf++;//skip delimiter
  }

  return input;
}

static inline float round_with_mode(float v, round_mode_t round_mode) {
	if (round_mode == ROUND_MODE_FLOOR) {
		return floorf(v);
	} else if (round_mode == ROUND_MODE_NEAREST) {
		return floorf(v + 0.5f);
	} else {
		return v;
	}
}

struct NNResult neuralNetworkInfer(float input[]) {
	static input_t inputs;
	static output_t outputs;

	MODEL_INPUT_NUMBER_T *input_flat = (MODEL_INPUT_NUMBER_T*)inputs;

	// Prepare inputs
	for (size_t i = 0; i < MODEL_INPUT_DIMS; i++) {
		// Fixed-point conversion if model input is integer
		if constexpr(std::is_integral_v<MODEL_INPUT_NUMBER_T>) {
#ifdef WITH_CMSIS_NN
// Not really CMSIS-NN, use SSAT anyway
			input_flat[i] = __SSAT((MODEL_INPUT_LONG_NUMBER_T)round_with_mode(input[i] * (1<<MODEL_INPUT_SCALE_FACTOR), MODEL_INPUT_ROUND_MODE), sizeof(MODEL_INPUT_NUMBER_T) * 8);
#else
			input_flat[i] = clamp_to(MODEL_INPUT_NUMBER_T, (MODEL_INPUT_LONG_NUMBER_T)round_with_mode(input[i] * (1<<MODEL_INPUT_SCALE_FACTOR), MODEL_INPUT_ROUND_MODE));
#endif
		} else {
			input_flat[i] = input[i];
		}
	}

	// Run inference
	cnn(inputs, outputs);

	// Get output class
	unsigned int label = 0;
	float max_val = outputs[0];
	for (unsigned int i = 1; i < MODEL_OUTPUT_SAMPLES; i++) {
		if (max_val < outputs[i]) {
			max_val = outputs[i];
			label = i;
		}
	}

	inference_count++;

	return {inference_count, label, max_val};
}

}
