/**
  ******************************************************************************
  * @file    defines.hh
  * @author  Pierre-Emmanuel Novac <penovac@unice.fr>, Université Côte d'Azur, LEAT, France
  * @version 2.1.0
  * @date    10 january 2024
  * @brief   Global C pre-processor definitions to use to build all source files (incl. CMSIS-NN)
  */

/* CMSIS-NN round mode definition */
#if defined(WITH_CMSIS_NN) || defined(WITH_NMSIS_NN)
{%- set first_round_mode = nodes[0].q.output_round_mode %}
{% for output_round_mode in nodes | map(attribute='q') | map(attribute='output_round_mode') -%}
{% if output_round_mode != first_round_mode %}
#error "CMSIS-NN requires all round modes to be identical, got output_round_mode {{ first_round_mode }} for {{ nodes[0].layer.name }} and {{ output_round_mode }} for {{ nodes[loop.index - 1].layer.name }}"
{%- endif %}
{%- endfor %}
{% for weights_round_mode in nodes | map(attribute='q') | map(attribute='weights_round_mode') -%}
{% if weights_round_mode is not none and weights_round_mode != first_round_mode %}
#error "CMSIS-NN requires all round modes to be identical, got output_round_mode {{ first_round_mode }} for {{ nodes[0].layer.name }} and weights_round_mode {{ weights_round_mode }} for {{ nodes[loop.index - 1].layer.name }}"
{%- endif %}
{%- endfor %}
{% if first_round_mode == "floor" -%}
#define ARM_NN_TRUNCATE 1
#define RISCV_NN_TRUNCATE 1
{% elif first_round_mode == "nearest" %}
#undef ARM_NN_TRUNCATE
#undef RISCV_NN_TRUNCATE
{% else %}
#error "Unrecognized round mode, only floor and nearest are supported by CMSIS-NN"
{% endif %}
#endif // defined(WITH_CMSIS_NN) || defined(WITH_NMSIS_NN)

