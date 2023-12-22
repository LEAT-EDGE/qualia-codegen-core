// Copyright 2021 (c) Pierre-Emmanuel Novac <penovac@unice.fr> Université Côte d'Azur, CNRS, LEAT. All rights reserved.

extern "C" {
#include "am_bsp.h"  // NOLINT
#ifdef WITH_CMSIS_NN
#include "arm_nnfunctions.h"
#endif
}
#include <cstdlib>
#include "serial.h"
#include <cmath>

extern "C" {
#include "model.h"
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

int main()
{
	am_hal_clkgen_control(AM_HAL_CLKGEN_CONTROL_SYSCLK_MAX, 0);
	am_hal_cachectrl_config(&am_hal_cachectrl_defaults);
	am_hal_cachectrl_enable();
	am_bsp_low_power_init();


	unsigned int inference_count = 0;
	uart_init();
	printf("%s\r\n", "READY");

	while (true) {
		static uint32_t msg_len = 0;
		static float input[MODEL_INPUT_DIMS];
		static input_t inputs;
		static output_t outputs;

		do { 
			msg_len = serialBufToFloats(input);
		} while (!msg_len);


		printf("%d\r\n", msg_len);

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


		printf("%d,%d,%f\r\n", inference_count, label, (double)max_val); // force double cast to workaround -Werror=double-promotion since printf uses variadic arguments so promotes to double automatically

	}
}
