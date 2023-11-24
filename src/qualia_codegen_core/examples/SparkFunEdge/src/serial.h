// Copyright 2021 (c) Pierre-Emmanuel Novac <penovac@unice.fr> Université Côte d'Azur, CNRS, LEAT. All rights reserved.

#ifndef __SERIAL_H__
#define __SERIAL_H__

#include <stdint.h>

void uart_init(void);
void uart_string_print(char *pcString);
uint32_t printf(const char *pcFmt, ...);

int serialBufToFloats(float input[]);

#endif //__SERIAL_H__
