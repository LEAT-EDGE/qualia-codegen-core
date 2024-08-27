cmake_minimum_required(VERSION 3.13)

project(Linux C CXX ASM)

find_package(Python3 REQUIRED
  COMPONENTS Interpreter
)

message(STATUS "Searching for libqualia-neuralnetwork sources")
execute_process(
  COMMAND ${Python3_EXECUTABLE} -c "from importlib.resources import files; print(files('libqualia-neuralnetwork'))"
  RESULT_VARIABLE LIBQUALIA_NEURALNETWORK_RESULT
  OUTPUT_VARIABLE LIBQUALIA_NEURALNETWORK_SOURCE_DIR
  ERROR_VARIABLE LIBQUALIA_NEURALNETWORK_ERROR
  OUTPUT_STRIP_TRAILING_WHITESPACE
  ERROR_STRIP_TRAILING_WHITESPACE
)

if (NOT "${LIBQUALIA_NEURALNETWORK_ERROR}" STREQUAL "")
  message(FATAL_ERROR ${LIBQUALIA_NEURALNETWORK_ERROR})
endif()
if (NOT ${LIBQUALIA_NEURALNETWORK_RESULT} EQUAL 0)
  message(FATAL_ERROR ${LIBQUALIA_NEURALNETWORK_RESULT})
endif()
message(STATUS "Found libqualia-neuralnetwork: ${LIBQUALIA_NEURALNETWORK_SOURCE_DIR}")

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

add_subdirectory(${LIBQUALIA_NEURALNETWORK_SOURCE_DIR} ${CMAKE_CURRENT_BINARY_DIR}/libqualia-neuralnetwork)

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
