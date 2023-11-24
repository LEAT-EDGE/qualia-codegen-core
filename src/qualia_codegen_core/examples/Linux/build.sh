#!/bin/sh

# Copyright 2021 (c) Pierre-Emmanuel Novac <penovac@unice.fr> Université Côte d'Azur, CNRS, LEAT. All rights reserved.

if [ "$#" -ne 1 ]; then
	echo "Usage: $0 <C model directory>" 1>&2
	exit 1
fi

g++ -ftrapv -Wall -Wextra -std=c++17 -pedantic -o main main.cpp "$1/model.c" -I"$1" -I"$1/include"
g++ -ftrapv -Wall -Wextra -std=c++17 -pedantic -lm -o single single.cpp "$1/model.c" -I"$1" -I"$1/include"

# g++ -Wall -Wextra -Wdouble-promotion -pedantic -Ofast -ffunction-sections -fdata-sections -fgraphite-identity -floop-nest-optimize -floop-parallelize-all -o main main.cpp
