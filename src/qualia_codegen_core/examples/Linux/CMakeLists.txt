cmake_minimum_required(VERSION 3.13)

project(Linux C CXX ASM)

set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)

set(Linux_CFLAGS
  "-Wall"
  "-Wextra"
  "-Werror=double-promotion"
  "-pedantic"

  "-Ofast"

  "-fdata-sections"
  "-ffunction-sections"
  "-fgraphite-identity"
  "-floop-nest-optimize"
  "-floop-parallelize-all"

  "-fsanitize=signed-integer-overflow"
  "-fno-sanitize-recover"
  "-DTRAPV_SHIFT"

  "-fdiagnostics-color=always"
)

add_compile_options(
  ${Linux_CFLAGS}
)

add_link_options(
  ${Linux_CFLAGS}
  "-Wl,--print-memory-usage"
)

add_subdirectory(../libqualia-neuralnetwork ${CMAKE_CURRENT_BINARY_DIR}/libqualia-neuralnetwork)

add_executable(${PROJECT_NAME}
  main.cpp
)

#target_include_directories(${PROJECT_NAME} PUBLIC
#)

target_link_libraries(${PROJECT_NAME} PUBLIC
  qualia-neuralnetwork
)

#target_compile_definitions(${PROJECT_NAME} PUBLIC
#)

target_compile_features(${PROJECT_NAME} PRIVATE
  cxx_std_20
)

add_executable(single
  single.cpp)

target_compile_features(single PRIVATE
  cxx_std_20
)

target_link_libraries(single PUBLIC
  qualia-neuralnetwork
)
