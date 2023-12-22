// Copyright 2021 (c) Pierre-Emmanuel Novac <penovac@unice.fr> Université Côte d'Azur, CNRS, LEAT. All rights reserved.

#include <algorithm>
#include <array>
#include <fstream>
#include <iostream>
#include <sstream>
#include <fstream> 
#include <vector>
#include <cmath>

#include "model.h"

template<int N>
std::vector<std::array<float, N>> readInputsFromFile(const char *filename) {
	// Read training vectors from CSV file
	std::vector<std::array<float, N>> inputs;
	std::ifstream fin(filename);
	std::string linestr;
	while (std::getline(fin, linestr)) {
		std::istringstream linestrs(linestr);
		std::string floatstr;
		std::array<float, N> floats;
		for (int i = 0; std::getline(linestrs, floatstr, ','); i++) {
			floats.at(i) = std::strtof(floatstr.c_str(), NULL);
		}
		inputs.push_back(floats);
	}
	return inputs;
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

template<size_t InputDims>
void convert_input_vector(const std::array<float, InputDims> &input, input_t out) {
	static_assert(InputDims == sizeof(input_t) / sizeof(MODEL_INPUT_NUMBER_T));
	MODEL_INPUT_NUMBER_T *out_flat = reinterpret_cast<MODEL_INPUT_NUMBER_T *>(out);

	for (size_t i = 0; i < InputDims; i++) {
		// Fixed-point conversion if model input is integer
		if constexpr(std::is_integral_v<MODEL_INPUT_NUMBER_T>) {
			out_flat[i] = clamp_to(MODEL_INPUT_NUMBER_T, (MODEL_INPUT_LONG_NUMBER_T)round_with_mode(input.at(i) * (1<<MODEL_INPUT_SCALE_FACTOR), MODEL_INPUT_ROUND_MODE));
		} else {
			out_flat[i] = input.at(i);
		}
	}
}

//Compute testing accuracy
template<size_t InputDims, size_t OutputDims>
float evaluate(const std::vector<std::array<float, InputDims>> &inputs, const std::vector<std::array<float, OutputDims>> &labels) {
	int rightlabels = 0;
	output_t outputs;

	// file creation
	for (size_t i = 0;  i < inputs.size() ; i++){ //&& i < labels.size()-1000; i++) {
		input_t converted_input; 

		convert_input_vector(inputs.at(i), converted_input);

		cnn(converted_input, outputs);

		// Max Element
		// Get output class
		unsigned int label = 0;
		float max_val = outputs[0];
		for (unsigned int i = 1; i < MODEL_OUTPUT_SAMPLES; i++) {
			if (max_val < outputs[i]) {
				max_val = outputs[i];
				label = i;
			}
		}

		//std::cout << std::endl;
		// std::cout << " target = " << labels.at(i)[0] << " found = " << cls << std::endl;

		if (labels.at(i).at(label) > 0) {
			rightlabels++;
		}
	}

	return rightlabels/(float)inputs.size();
}

int main(int argc, const char *argv[]) {
	if (argc != 3) {
		std::cerr << "Usage: " << argv[0] << " testX.csv testY.csv" << std::endl;
		exit(1);
	}

	auto inputs = readInputsFromFile<MODEL_INPUT_DIMS>(argv[1]);
	auto labels = readInputsFromFile<MODEL_OUTPUT_SAMPLES>(argv[2]);

	auto acc = evaluate(inputs, labels);

	std::cerr << "Testing accuracy: " << acc << std::endl;

	return 0;
}
