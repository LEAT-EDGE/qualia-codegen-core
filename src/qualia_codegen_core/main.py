#!/usr/bin/env python3

# Copyright 2021 (c) Pierre-Emmanuel Novac <penovac@unice.fr> Université Côte d'Azur, CNRS, LEAT. All rights reserved.
# April 29, 2021
from __future__ import annotations

import copy
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

from qualia_codegen_core import Converter
from qualia_codegen_core.graph import ModelGraph, Quantization


@dataclass
class ActivationRange:
    input_max: float | None
    output_max: float | None
    input_q: int | None
    activation_q: int | None
    weights_q: int | None

logger = logging.getLogger(__name__)

def is_hdf5(filepath: Path) -> bool:
    with filepath.open('rb') as f:
        header = f.read(8)
        return header == b'\x89\x48\x44\x46\x0d\x0a\x1a\x0a'

def load_modelgraph(filepath: Path, module_name: str = '', *strargs: str) -> ModelGraph | None:
    if module_name:  # PyTorch
        import importlib.util

        import torch

        args = [eval(arg) for arg in strargs] # noqa: PGH001 S307 eval() is used to convert string to expression for any arg type

        modname, classname = module_name.rsplit('.', 1)
        mod = importlib.import_module(modname)
        tmodel = getattr(mod, classname)(*args)
        tmodel.eval()
        tmodel.load_state_dict(torch.load(filepath))
        logger.info('PyTorch model: %s', tmodel)

        from qualia_codegen_core.graph.TorchModelGraph import TorchModelGraph

        return TorchModelGraph(tmodel).convert()

    if is_hdf5(filepath):  # Keras
        import tensorflow as tf  # type: ignore[import-untyped]
        from keras.models import load_model  # type: ignore[import-untyped] # No stubs for keras package

        from .graph.KerasModelGraph import KerasModelGraph

        # We don't need a GPU, don't request it
        tf.config.set_visible_devices([], 'GPU')

        kmodel = load_model(filepath)
        logger.info('Keras model:')
        kmodel.summary()

        return KerasModelGraph(kmodel).convert()

    logger.error('Weights file is not HDF5 and no PyTorch module name specified')
    return None

def load_activations_range(path: Path) -> dict[str, ActivationRange]:
    activations_range: dict[str, ActivationRange] = {}
    first_q = None
    with path.open() as f:
        for line in f:
            r = line.split(',')
            activations_range[r[0]] = ActivationRange(float(r[1]), float(r[2]), int(r[3]), int(r[4]), int(r[5]))
            if first_q is None:
                first_q = int(r[3])

    return activations_range

def annotate_quantization(
        modelgraph: ModelGraph,
        activations_range: dict[str, ActivationRange],
        number_type: type[int | float],
        width: int,
        long_width: int) -> bool:

    if number_type == int: # Activation range only when using fixed-point quantization
        if not activations_range:
            logger.error('No activations range data available, required for fixed-point quantization')
            return False

        # Populate quantization information for all layers from activations_range
        for node in modelgraph.nodes:
            if node.layer.name in activations_range:
                node.q = Quantization(
                        number_type=number_type,
                        width=width,
                        long_width=long_width,
                        weights_scale_factor=activations_range[node.layer.name].weights_q,
                        output_scale_factor=activations_range[node.layer.name].activation_q,
                        )
            elif not node.innodes:
                logger.warning('No quantization information for %s, looking for a subsequent layer with information',
                               node.layer.name)

                nextnode = node.outnodes[0]
                while nextnode.layer.name not in activations_range and nextnode.outnodes:
                    nextnode = nextnode.outnodes[0]

                if nextnode.layer.name in activations_range:
                    logger.warning('Applying layer %s quantization information to %s', nextnode.layer.name, node.layer.name)
                    node.q = Quantization(
                            number_type=number_type,
                            width=width,
                            long_width=long_width,
                            weights_scale_factor=activations_range[nextnode.layer.name].weights_q,
                            output_scale_factor=activations_range[nextnode.layer.name].activation_q,
                            )
                else:
                    logger.error('No quantization information for %s, and no previous layer to copy from', node.layer.name)
            else:
                logger.warning('No quantization information for %s, applying first previous layer %s information',
                               node.layer.name, node.innodes[0].layer.name)
                node.q = copy.deepcopy(node.innodes[0].q)

    else:
        for node in modelgraph.nodes:
            # No scale factor if not fixed-point quantization on integers
            node.q = Quantization(
                    number_type=number_type,
                    width=width,
                    long_width=long_width,
                    weights_scale_factor=0,
                    output_scale_factor=0,
                    )
    return True

def qualia_codegen(filename: str,
               quantize: str = 'float32',
               activations_range_file: str = '',
               module_name: str = '', # PyTorch module name
               *args: str) -> str | None: # PyTorch module args
    filepath = Path(filename)
    fname = filepath.stem

    number_type: type[int | float]

    if quantize == 'float32':
        number_type = float
        width = 32
        long_width = 32
    elif quantize == 'int16':
        number_type = int
        width = 16
        long_width = 32
    elif quantize == 'int8':
        number_type = int
        width = 8
        long_width = 16
    else:
        logger.error('Qualia-CodeGen only supports no (float32) quantization, int8 or int16 quantization, got %s', quantize)
        return None

    modelgraph = load_modelgraph(filepath, module_name, *args)
    if not modelgraph:
        return None

    activations_range = {}
    if activations_range_file:
        activations_range = load_activations_range(Path(activations_range_file))

    if not annotate_quantization(modelgraph, activations_range, number_type, width, long_width):
        return None

    converter = Converter(output_path=Path('out')/'qualia_codegen'/fname)

    fullmodel_h = converter.convert_model(modelgraph)

    if fullmodel_h:
        with (Path('out')/'qualia_codegen'/fname/'full_model.h').open('w') as f:
            _ = f.write(fullmodel_h)

    return fullmodel_h

def main() -> int:
    if len(sys.argv) < 2: # noqa: PLR2004 Only required arg is weights file
        logger.error('Usage: %s <weights_file>'
                     ' [quantization] [activations_range_file] [pytorch_module_name] [pytorch_module_args]', sys.argv[0])
        sys.exit(1)

    logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)

    return 0 if qualia_codegen(*sys.argv[1:]) is not None else 1

if __name__ == '__main__':
    sys.exit(main())
