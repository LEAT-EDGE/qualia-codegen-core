{% import 'dump_featuremaps.cc' as featuremaps %}

/**
  ******************************************************************************
  * @file    model.cc
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, LEAT, CNRS, Université Côte d'Azur, France
  * @version 1.0.0
  * @date    24 march 2020
  * @brief   Template generating plain C code for the implementation of Convolutional Neural Networks on MCU
  */

#ifdef __cplusplus
extern "C" {
#endif

#ifndef SINGLE_FILE
#include "number.h"
#include "model.h"
// #include <chrono>
{% if dump_featuremaps %}
  {{ featuremaps.includes() }}
{% endif -%}

{% for node in nodes[1:] %} // InputLayer is excluded
#include "{{ node.layer.name }}.c"
  {%- if node.layer.weights | length > 0 %}
#include "weights/{{ node.layer.name }}.c"
  {%- endif %}
{%- endfor %}
#endif

{% if dump_featuremaps %}
static int sample = 0; // Track current sample
{% endif -%}


void cnn(
  const input_t input,
  {{ nodes[-1].layer.name }}_output_type {{ nodes[-1].layer.name }}_output) {
  
  // Output array allocation
{%- for pool in allocation.pools %}
  static union {
  {%- for node in pool %}
    {{ node.layer.name }}_output_type {{ node.layer.name }}_output;
  {%- endfor %}
  } activations{{ loop.index }};
{% endfor %}

{% if dump_featuremaps %}
  char path[FILENAME_MAX] = { '\0' };
  // Prepare output file name
  snprintf(path, FILENAME_MAX, "{{ dump_featuremaps_path }}/%d/{{ nodes[0].layer.name }}.csv", sample);

  // Input
  {{ featuremaps.write(nodes, allocation, nodes[0]) }}
{% endif -%}

// Model layers call chain {# InputLayer is excluded #}
{%- for node in nodes[1:] -%}  
  {# Write function conversion if there is a type mismatch between two layer of the network #}
  {% set outer_loop = loop %}
  {%- for innode in node.innodes -%}
    {%- if innode.q.number_type != node.q.number_type or innode.q.width != node.q.width +%}
  // TYPE WARNING for {{node.layer.name}} Innode {{innode.layer.name}} 
  // innode is {{qtype2ctype(innode.q.number_type, innode.q.width)}} type and layer is {{qtype2ctype(node.q.number_type, node.q.width)}} type
  {{ qtype2ctype(node.q.number_type, node.q.width) }} {{innode.layer.name}}_output_convert_{{outer_loop.index}}
      {%- for dim in node.input_shape[loop.index - 1][1:] -%}
          [{{dim}}]
      {%- endfor -%};
  {{ qtype2ctype(innode.q.number_type, innode.q.width) }}_to_{{ qtype2ctype(node.q.number_type, node.q.width) }}(
      {%- if innode.layer.__class__.__name__ == 'TInputLayer' %} // Model input is passed as model parameter
        ({{ qtype2ctype(innode.q.number_type, innode.q.width) }}*)input,
      {%- else %}
        ({{ qtype2ctype(innode.q.number_type, innode.q.width) }}*)activations{{ allocation.index[innode] }}.{{ innode.layer.name }}_output,
      {%- endif -%}
        ({{ qtype2ctype(node.q.number_type, node.q.width) }}*){{innode.layer.name}}_output_convert_{{outer_loop.index}},
        {{ node.input_shape[loop.index - 1][1:] | join('*') }},
        {{innode.q.output_scale_factor}});
    {%- endif -%}
  {%- endfor %}
  {# type mismatch fix - end #}
  {{ node.layer.name }}(
    {%- for innode in node.innodes %}
      {%- if innode.q.number_type != node.q.number_type or innode.q.width != node.q.width %}
    // type warning, use instead :
    {{innode.layer.name}}_output_convert_{{outer_loop.index}},
    //activations{{ allocation.index[innode] }}.{{ innode.layer.name }}_output,
      {%- elif innode.layer.__class__.__name__ == 'TInputLayer' %} // Model input is passed as model parameter
    input,
      {%- else %}
    activations{{ allocation.index[innode] }}.{{ innode.layer.name }}_output,
      {%- endif %}
    {%- endfor %}
    {%- for weights_name in node.layer.weights.keys() %}
    {{ node.layer.name}}_{{weights_name}},
    {%- endfor %}
    {%- if node != nodes[-1] %}
    activations{{ allocation.index[node] }}.{{ node.layer.name }}_output
    {% else -%} // Last layer uses output passed as model parameter
    {{ node.layer.name }}_output
    {% endif -%}
  );

  {% if dump_featuremaps %}
  // Prepare output file name
  snprintf(path, FILENAME_MAX, "{{ dump_featuremaps_path }}/%d/{{ node.layer.name }}.csv", sample);
  {{ featuremaps.write(nodes, allocation, node) }}
  {% endif -%}
{%- endfor %}

{% if dump_featuremaps %}
  sample++; // Increment sample count
{% endif -%}
}

#ifdef __cplusplus
} // extern "C"
#endif
