from __future__ import annotations

import logging
from typing import ClassVar

from .Converter import Converter

logger = logging.getLogger(__name__)


class MetricsConverter(Converter):
    metric_classes: ClassVar[dict[str, str]] = {
        'acc': 'Accuracy',
        'mse': 'MeanSquaredError',
        'corr': 'PearsonCorrelationCoefficient',
    }

    def write_metrics_header(self, metrics: list[str]) -> str:
        return self.render_template('include/metrics.hh', self.output_path_header / 'metrics.h', metrics=metrics)

    def write_metrics(self, metrics: list[str]) -> str:
        return self.render_template('metrics.cppp', self.output_path / 'metrics.cpp', metrics=metrics)

    def convert_metrics(self, metrics: list[str]) -> str | None:
        if self._template_path is None:
            logger.error('Could not discover template path from module')
            return None

        metrics_classes = [self.metric_classes[metric] for metric in metrics]

        # Used to ignore includes in generated files for combined returned code
        rendered = '#define SINGLE_FILE\n'

        rendered += self.write_metrics_header(metrics=metrics_classes) + '\n'
        rendered += self.write_metrics(metrics=metrics_classes) + '\n'

        return rendered
