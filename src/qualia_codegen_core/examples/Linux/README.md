Copyright 2021 (c) Pierre-Emmanuel Novac <penovac@unice.fr> Université Côte d'Azur, CNRS, LEAT. All rights reserved.

# Command line demo app to get accuracy over test dataset
## Build
```
./build.sh  <C model directory>
```
Replace `<C model directory>` with the directory where the C code was generated (with the `model.c` file and layer files).

## Run
```
./main testX.csv testY.csv
```
`testX.csv` is a CSV file containing one test vector per line

`testY.csv` is a CSV file containing one label vector per line corresponding to the test vector. Label vectors are one hot encoded, meaning that the dimension of the vector is the number of output classes, and the element corresponding to the correct class is set to 1 while the others are set to 0.
