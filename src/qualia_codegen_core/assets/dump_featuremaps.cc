{% macro includes() %}
#include <errno.h>
#include <stdio.h>
#include <sys/stat.h>
#include <sys/types.h>
{% endmacro %}

{% macro write(nodes, allocation, node) %}
  // Dump feature maps for {{ node.layer.name }}
  {
    // Recursive creation of output directory
    for (char *p = path + 1; *p && p < path + FILENAME_MAX; p++) {
      if (*p == '/') {
        *p = 0;
        if (mkdir(path, S_IRWXU | S_IRWXG | S_IRWXO) == -1) {
          if (errno != EEXIST) { // No error if directory exists
            fprintf(stderr, "Could not create directory \"");
            fprintf(stderr, path);
            fprintf(stderr, "\": ");
            perror(NULL);
          }
        }
        *p = '/';
      }
    }

    FILE *f = fopen(path, "w");
    if (f == NULL) {
      fprintf(stderr, "Could not open \"");
      fprintf(stderr, path);
      fprintf(stderr, "\": ");
      perror(NULL);
    } else {
      size_t i = 0; // Total count

      // Loop over all dimensions of first output feature maps
      {%- for dim in node.output_shape[0][1:] %} 
      {%- filter indent(2 * loop.index0) %}
      for (size_t i_{{ loop.index }} = 0; i_{{ loop.index }} < {{ dim }}; i_{{ loop.index }}++) {
      {%- endfilter %}
      {%- endfor %}
      {%- filter indent(2 * (node.output_shape[0][1:] | length - 1)) %}
        if (i != 0) {
          fprintf(f, ",");
        }
        fprintf(f,
        {%- if node.q.number_type.__name__ == 'int' -%}
          "%d",
        {%- else -%}
          "%f",
        {%- endif %}
        {%- if node.layer.__class__.__name__ == 'TInputLayer' -%} // Model input is passed as model parameter
          input
        {%- elif node != nodes[-1] -%}
          activations{{ allocation.index[node] }}.{{ node.layer.name }}_output
        {%- else -%} // Last layer uses output passed as model parameter
          {{ node.layer.name }}_output
        {%- endif %}
          {%- for dim in node.output_shape[0][1:] -%}[i_{{ loop.index }}]{%- endfor -%}
        );
        i++;
      {%- endfilter %}
      {%- for dim in node.output_shape[0][1:] %}
      {%- filter indent(2 * loop.revindex0) %}
      }
      {%- endfilter %}
      {%- endfor %}
      fprintf(f, "\n");

      fclose(f);
    }
  }
{% endmacro %}
