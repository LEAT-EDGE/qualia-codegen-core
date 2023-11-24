// Copyright 2021 (c) Pierre-Emmanuel Novac <penovac@unice.fr> Université Côte d'Azur, CNRS, LEAT. All rights reserved.
// April 29, 2021

#ifndef _NEURALNETWORK_H_
#define _NEURALNETWORK_H_

#include "stdlib.h"

#ifdef __cplusplus
extern "C" {
#endif
struct NNResult {
	unsigned int inference_count;
	unsigned int label;
	float dist;
};

float *serialBufToFloats(char buf[], size_t buflen);
struct NNResult neuralNetworkInfer(float input[]);
#ifdef __cplusplus
}
#endif
#endif
