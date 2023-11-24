// Copyright 2021 (c) Pierre-Emmanuel Novac <penovac@unice.fr> Université Côte d'Azur, CNRS, LEAT. All rights reserved.

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <type_traits>

//#include "output/number.h"
#include "model.h"

int main(int argc, const char*argv[]) {
	if (argc < 2) {
		printf("Usage: %s <test vector>\n", argv[0]);
		return 1;
	}

	input_t input;
	output_t output;

	MODEL_INPUT_NUMBER_T *input_flat = (MODEL_INPUT_NUMBER_T*)input;

	for (size_t i = 0; i < MODEL_INPUT_DIMS; i++) {
		// Fixed-point conversion if model input is integer
		if constexpr(std::is_integral_v<MODEL_INPUT_NUMBER_T>) {
			input_flat[i] = clamp_to(MODEL_INPUT_NUMBER_T, (MODEL_INPUT_LONG_NUMBER_T)floor(strtof(argv[i + 1], NULL) * (1<<MODEL_INPUT_SCALE_FACTOR)));
		} else {
			input_flat[i] = strtof(argv[i + 1], NULL);
		}
	}

	cnn(input, output);

	for (int i = 0; i < MODEL_OUTPUT_SAMPLES; i++) {
		if constexpr(std::is_integral_v<MODEL_INPUT_NUMBER_T>) {
			printf("%d\n", output[i]);
		} else {
			printf("%f\n", output[i]);
		}
	}

	return 0;
}
