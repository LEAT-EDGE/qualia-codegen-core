Copyright 2021 (c) Pierre-Emmanuel Novac <penovac@unice.fr> Université Côte d'Azur, CNRS, LEAT. All rights reserved.

# Qualia-CodeGen-Core

Converts a pre-trained Keras .h5 or PyTorch model to C code for inference.

Generated C code uses `channels_last` data format.

## Supported layers

 - **Activation**: ReLU (combined to previous Conv1D, Dense, MaxPooling1D, AveragePooling1D AddV2), Softmax
 - **Conv1D**: optional bias, valid padding only
 - **Dense**: optional bias
 - **MaxPooling1D**: valid padding only
 - **AveragePooling1D**: valid padding only
 - **Flatten**: implies reordering next layer's kernel for data format conversion
 - **ZeroPadding1D**: combined with next Conv1D
 - **AddV2**


## Dependencies

```
python >= 3.9
```
Python:
```
jinja2
numpy
```

### Keras
Python:
```
tensorflow >= 2.6.0
keras >= 2.6.0 
```
### PyTorch
Python:
```
torch >= 1.8.0
```

## Installation
```
pip install -e .
```

## Usage

### Generate C code from Keras .h5

```
qualia_codegen <model.h5> <output directory>
```

### Use in your C code

Include the model: (can also be built as a separate object)
```
#include "model.h"
```

Allocate `inputs` and `outputs` arrays with correct dimensions. Remember that `inputs` must have `channels_last` data format.

Call it in your C code:
```
cnn(inputs, outputs);
```

Add the source file `model.c` to your build system. It includes all the other source files for layers, don't add these to the build system.

## Examples
See the `src/qualia_codegen_core/examples/Linux` directory for a demo console application to evaluate model accuracy.

`src/qualia_codegen_core/examples/qualia_codegen-NucleoL476RG` contains an STM32CubeIDE project for the Nucleo-L476RG board that's currently broken due to some recent changes

## Documentation
Nothing much…

### Source tree
`src/qualia_codegen_core/Allocator.py`: manages activation buffer allocation. Tries to group all buffers into one, except when they cannot be overwritten (dependencies).

`src/qualia_codegen_core/Converter.py`: the actual conversion code, parses a Keras model and use the template file associated to each layer to generate C code. When weights have to be written, they are optionally quantized to fixed-point by setting the appropriate parameters of `Converter` constructor (see its definition)

`src/qualia_codegen_core/Validator.py`: work in progress, should contain functions to check if a model can be successfully converted. For now only check activation function.

`src/qualia_codegen_core/assets/`: contains the templates to generate C inference code

`src/qualia_codegen_core/assets/layers/`: contains the implementation of the various supported layers

`src/qualia_codegen_core/assets/layers/weights`: contains the support for the trainable layers weights
