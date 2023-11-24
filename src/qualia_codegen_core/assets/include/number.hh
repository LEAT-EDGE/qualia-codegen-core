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

#define True 1
#define False 0

#define _clamp_to(type, number) clamp_to_number_t_ ## type (number)
#define clamp_to(type, number) _clamp_to(type, number)
#define _scale(type, number, scale_factor) scale_number_t_ ## type (number, scale_factor)
#define scale(type, number, scale_factor) _scale(type, number, scale_factor)

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
  {{ qtype2ctype(number_type.number_type, number_type.long_width) }} number, int scale_factor) {
	return number;
}
static inline {{ qtype2ctype(number_type.number_type, number_type.width) }} clamp_to_number_t_{{qtype2ctype(number_type.number_type, number_type.width)}}(
  {{ qtype2ctype(number_type.number_type, number_type.long_width) }} number) {
	return ({{ qtype2ctype(number_type.number_type, number_type.width) }}) number;
}
{% else -%}
static inline {{ qtype2ctype(number_type.number_type, number_type.long_width) }} scale_number_t_{{ qtype2ctype(number_type.number_type, number_type.width) }}(
  {{ qtype2ctype(number_type.number_type, number_type.long_width) }} number, int scale_factor) {
  if (scale_factor < 0)
    return number << - scale_factor;
  else 
    return number >> scale_factor;
}
static inline {{ qtype2ctype(number_type.number_type, number_type.width) }} clamp_to_number_t_{{ qtype2ctype(number_type.number_type, number_type.width) }}(
  {{ qtype2ctype(number_type.number_type, number_type.long_width) }} number) {
	return ({{ qtype2ctype(number_type.number_type, number_type.width) }}) max_{{ qtype2ctype(number_type.number_type, number_type.width) }}(
      NUMBER_MIN_{{ qtype2ctype(number_type.number_type, number_type.width) | upper }},
      min_{{ qtype2ctype(number_type.number_type, number_type.width) }}(
        NUMBER_MAX_{{ qtype2ctype(number_type.number_type, number_type.width) | upper }}, number));
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

