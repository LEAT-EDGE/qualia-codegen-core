// Copyright 2021 (c) Pierre-Emmanuel Novac <penovac@unice.fr> Université Côte d'Azur, CNRS, LEAT. All rights reserved.

#include <algorithm>
#include <array>
#include <fstream>
#include <iostream>
#include <memory>
#include <optional>
#include <sstream>
#include <fstream> 
#include <vector>
#include <cmath>
#include "NeuralNetwork.h"
#include "metrics.h"

template<int N>
std::vector<std::array<float, N>> readInputsFromFile(const char *filename) {
	// Read training vectors from CSV file
	std::vector<std::array<float, N>> inputs;
	std::ifstream fin(filename);
	std::string linestr;
	while (std::getline(fin, linestr)) {
		std::istringstream linestrs(linestr);
		std::string floatstr;
		std::array<float, N> floats{};
		for (int i = 0; std::getline(linestrs, floatstr, ','); i++) {
			floats.at(i) = std::strtof(floatstr.c_str(), NULL);
		}
		inputs.push_back(floats);
	}
	return inputs;
}

template<typename T, size_t OutputDims>
void writePredsToFile(const std::array<T, OutputDims> &preds, std::ofstream &fout) {
	for (size_t j = 0; j < preds.size(); j++) {
		if (j != 0) {
			fout << ',';
		}
		fout << preds[j];
	}
	fout << std::endl;
}

//Compute testing accuracy
template<size_t InputDims, size_t OutputDims>
void evaluate(
		const std::vector<std::array<float, InputDims>> &inputs,
		const std::vector<std::array<float, OutputDims>> &targets,
		std::optional<std::ofstream> &fout) {
	static NeuralNetwork nn{metrics};

	for (size_t i = 0;  i < inputs.size() ; i++){ //&& i < labels.size()-1000; i++) {
		auto preds = nn.evaluate(inputs.at(i), targets.at(i));

		if (fout) {
			writePredsToFile(preds, fout.value());
		}
	}

	auto metrics_result = nn.getMetricsResult();

	for (size_t i = 0; i < metrics.size() && i < metrics_result.size(); i++) {
		std::cerr << metrics[i]->name() << "=" << metrics_result[i] << std::endl;
	}
}

int main(int argc, const char *argv[]) {
	if (argc < 3) {
		std::cerr << "Usage: " << argv[0] << " test_x.csv test_y.csv [preds.csv]" << std::endl;
		exit(1);
	}

	auto inputs = readInputsFromFile<MODEL_INPUT_DIMS>(argv[1]);
	auto labels = readInputsFromFile<MODEL_OUTPUT_SAMPLES>(argv[2]);

	std::optional<std::ofstream> fout;
	if (argc == 4) {
		fout = std::ofstream{argv[3]};
	}

	evaluate(inputs, labels, fout);

	return 0;
}
