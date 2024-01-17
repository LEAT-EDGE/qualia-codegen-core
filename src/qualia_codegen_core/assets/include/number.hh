/**
  ******************************************************************************
  * @file    number.hh
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version 1.0.0
  * @date    2 february 2021
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#ifdef __cplusplus
extern "C" {
#endif

#ifndef __NUMBER_H__
#define __NUMBER_H__

#include <stdint.h>
#include <stddef.h>
#include <math.h>

#ifdef TRAPV_SHIFT
#include <limits.h>
#include <stdio.h>
#include <assert.h>
#endif

#ifdef WITH_CMSIS_NN
#include "arm_nnfunctions.h"
#endif

#define _clamp_to(type, number) clamp_to_number_t_ ## type (number)
#define clamp_to(type, number) _clamp_to(type, number)
#define _scale(type, number, scale_factor, round_mode) scale_number_t_ ## type (number, scale_factor, round_mode)
#define scale(type, number, scale_factor, round_mode) _scale(type, number, scale_factor, round_mode)
#define _scale_and_clamp_to(type, number, scale_factor, round_mode) scale_and_clamp_to_number_t_ ## type (number, scale_factor, round_mode)
#define scale_and_clamp_to(type, number, scale_factor, round_mode) _scale_and_clamp_to(type, number, scale_factor, round_mode)

typedef enum {
  ROUND_MODE_NONE,
  ROUND_MODE_FLOOR,
  ROUND_MODE_NEAREST,
} round_mode_t;

// Idea 1: Write the smallest min max interval of the net, could be an issue for hybrid int type network
// Idea 2: listing any interval and add type in name in a switch case like <- better but painfull
// #define NUMBER_MIN	{{ number_min }}	// Max value for this numeric type
// #define NUMBER_MAX	{{ number_max }}	// Min value for this numeric type

// // Idea 1: List of all types and write any corresponding function 
// typedef {{ number_type }} number_t;		// Standard size numeric type used for weights and activations
// typedef {{ long_number_type }} long_number_t;	// Long numeric type used for intermediate results

{% for number_type in number_types -%}
#define NUMBER_MIN_{{ qtype2ctype(number_type.number_type, number_type.width) | upper }} {{ number_type.min_val }}
#define NUMBER_MAX_{{ qtype2ctype(number_type.number_type, number_type.width) | upper }} {{ number_type.max_val }}

static inline {{ qtype2ctype(number_type.number_type, number_type.long_width) }} min_{{ qtype2ctype(number_type.number_type, number_type.width) }}(
    {{ qtype2ctype(number_type.number_type, number_type.long_width) }} a,
    {{ qtype2ctype(number_type.number_type, number_type.long_width) }} b) {
	if (a <= b)
		return a;
	return b;
}

static inline {{ qtype2ctype(number_type.number_type, number_type.long_width) }} max_{{ qtype2ctype(number_type.number_type, number_type.width) }}(
    {{ qtype2ctype(number_type.number_type, number_type.long_width) }} a,
    {{ qtype2ctype(number_type.number_type, number_type.long_width) }} b) {
	if (a >= b)
		return a;
	return b;
}

{% if number_type.number_type.__name__ == 'float' -%}
static inline {{ qtype2ctype(number_type.number_type, number_type.long_width) }} scale_number_t_{{ qtype2ctype(number_type.number_type, number_type.width) }}(
  {{ qtype2ctype(number_type.number_type, number_type.long_width) }} number, int scale_factor, round_mode_t round_mode) {
	return number;
}
static inline {{ qtype2ctype(number_type.number_type, number_type.width) }} clamp_to_number_t_{{qtype2ctype(number_type.number_type, number_type.width)}}(
  {{ qtype2ctype(number_type.number_type, number_type.long_width) }} number) {
	return ({{ qtype2ctype(number_type.number_type, number_type.width) }}) number;
}
static inline {{ qtype2ctype(number_type.number_type, number_type.width) }} scale_and_clamp_to_number_t_{{qtype2ctype(number_type.number_type, number_type.width)}}(
  {{ qtype2ctype(number_type.number_type, number_type.long_width) }} number, int scale_factor, round_mode_t round_mode) {
	return ({{ qtype2ctype(number_type.number_type, number_type.width) }}) number;
}
{% else -%}
static inline {{ qtype2ctype(number_type.number_type, number_type.long_width) }} scale_number_t_{{ qtype2ctype(number_type.number_type, number_type.width) }}(
  {{ qtype2ctype(number_type.number_type, number_type.long_width) }} number, int scale_factor, round_mode_t round_mode) {


  if (scale_factor <= 0) {
#ifdef TRAPV_SHIFT
    // Check for possible overflow of left shift
    if (number > {{ number_type.number_type.__name__ | upper }}{{ number_type.long_width }}_MAX >> -scale_factor) {
      fprintf(stderr,
              "Error: scale() overflow, number={{ '%ld' if number_type.long_width > 32 else '%d' }}, scale_factor=%d, limit=%d\n",
              number,
              scale_factor,
              INT16_MAX >> -scale_factor);
      assert(number <= {{ number_type.number_type.__name__ | upper }}{{ number_type.long_width }}_MAX >> -scale_factor);
    }
#endif
    // No rounding to apply when shifting left
    return number << - scale_factor;
  } else {
    if (round_mode == ROUND_MODE_NEAREST) {
      number += (1 << (scale_factor - 1)); // +0.5 in fixed-point
    }
    return number >> scale_factor;
  }
}
static inline {{ qtype2ctype(number_type.number_type, number_type.width) }} clamp_to_number_t_{{ qtype2ctype(number_type.number_type, number_type.width) }}(
  {{ qtype2ctype(number_type.number_type, number_type.long_width) }} number) {
	return ({{ qtype2ctype(number_type.number_type, number_type.width) }}) max_{{ qtype2ctype(number_type.number_type, number_type.width) }}(
      NUMBER_MIN_{{ qtype2ctype(number_type.number_type, number_type.width) | upper }},
      min_{{ qtype2ctype(number_type.number_type, number_type.width) }}(
        NUMBER_MAX_{{ qtype2ctype(number_type.number_type, number_type.width) | upper }}, number));
}
static inline {{ qtype2ctype(number_type.number_type, number_type.width) }} scale_and_clamp_to_number_t_{{ qtype2ctype(number_type.number_type, number_type.width) }}(
  {{ qtype2ctype(number_type.number_type, number_type.long_width) }} number, int scale_factor, round_mode_t round_mode) {
#ifdef WITH_CMSIS_NN
  // Not really CMSIS-NN but use SSAT anyway
  if (scale_factor <= 0) {
    // No rounding to apply when shifting left
    return __SSAT(number << - scale_factor, sizeof({{ qtype2ctype(number_type.number_type, number_type.width) }}) * 8);
  } else {
    if (round_mode == ROUND_MODE_NEAREST) {
      number += (1 << (scale_factor - 1)); // +0.5 in fixed-point
    }
    return __SSAT(number >> scale_factor, sizeof({{ qtype2ctype(number_type.number_type, number_type.width) }}) * 8);
  }
#else
  number = scale_number_t_{{ qtype2ctype(number_type.number_type, number_type.width) }}(number, scale_factor, round_mode);
  return clamp_to_number_t_{{ qtype2ctype(number_type.number_type, number_type.width) }}(number);
#endif
}
{%- endif %}

{% endfor %}


static inline void int64_t_to_float(int64_t * tabint, float * tabfloat, long tabdim, int scale_factor){
  for (int i=0; i<tabdim; i++){
    tabfloat[i] = (float)tabint[i] / (1<<scale_factor);
  }
}

static inline void int32_t_to_float(int32_t * tabint, float * tabfloat, long tabdim, int scale_factor){
  for (int i=0; i<tabdim; i++){
    tabfloat[i] = (float)tabint[i] / (1<<scale_factor);
  }
}

static inline void int16_t_to_float(int16_t * tabint, float * tabfloat, long tabdim, int scale_factor){
  for (int i=0; i<tabdim; i++){
    tabfloat[i] = ((float)tabint[i]) / (1<<scale_factor);
  }
}

static inline void int8_t_to_float(int8_t * tabint, float * tabfloat, long tabdim, int scale_factor){
  for (int i=0; i<tabdim; i++){
    tabfloat[i] = ((float)tabint[i]) / (1<<scale_factor);
  }
}
#endif //__NUMBER_H__

#ifdef __cplusplus
} // extern "C"
#endif

