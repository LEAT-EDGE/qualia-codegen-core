set(MODEL_DIR "${CMAKE_CURRENT_SOURCE_DIR}/model" CACHE PATH "Path to generated C model")
set(WITH_CMSIS_NN False CACHE BOOL "Use CMSIS-NN library for optimizations")
set(CMSIS_NN_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../third_party/cmsis/CMSIS/NN" CACHE PATH "Path to CMSIS-NN library sources")

set(LIBQUALIA_NEURALNETWORK_CFLAGS
  -Ofast)

add_library(qualia-neuralnetwork
  NeuralNetwork.cpp
  ${MODEL_DIR}/model.c)

target_compile_options(qualia-neuralnetwork PRIVATE
  ${LIBQUALIA_NEURALNETWORK_CFLAGS}
  "-include" "${MODEL_DIR}/include/defines.h"
)

if(WITH_CMSIS_NN)
  target_compile_definitions(qualia-neuralnetwork PRIVATE
    "WITH_CMSIS_NN"
  )

  add_library(cmsis-nn
    # Optimized INT16 functions
    ${CMSIS_NN_PATH}/Source/ConvolutionFunctions/arm_convolve_HWC_q15_basic_nonsquare.c
    ${CMSIS_NN_PATH}/Source/ConvolutionFunctions/arm_convolve_HWC_q15_fast_nonsquare.c
    ${CMSIS_NN_PATH}/Source/ActivationFunctions/arm_relu_q15.c
    ${CMSIS_NN_PATH}/Source/FullyConnectedFunctions/arm_fully_connected_q15.c

    # Optimized INT8 functions
    ${CMSIS_NN_PATH}/Source/ConvolutionFunctions/arm_convolve_HWC_q7_basic_nonsquare.c
    ${CMSIS_NN_PATH}/Source/ConvolutionFunctions/arm_convolve_HWC_q7_fast_nonsquare.c
    ${CMSIS_NN_PATH}/Source/NNSupportFunctions/arm_q7_to_q15_no_shift.c
    ${CMSIS_NN_PATH}/Source/ConvolutionFunctions/arm_nn_mat_mult_kernel_q7_q15.c
    ${CMSIS_NN_PATH}/Source/ConvolutionFunctions/arm_nn_mat_mult_kernel_q7_q15_reordered.c
    ${CMSIS_NN_PATH}/Source/ActivationFunctions/arm_relu_q7.c
    ${CMSIS_NN_PATH}/Source/FullyConnectedFunctions/arm_fully_connected_q7.c
    ${CMSIS_NN_PATH}/Source/NNSupportFunctions/arm_q7_to_q15_reordered_no_shift.c
  )

  target_compile_options(cmsis-nn PRIVATE
    ${LIBQUALIA_NEURALNETWORK_CFLAGS}
    "-include" "${MODEL_DIR}/include/defines.h"
  )

target_include_directories(cmsis-nn PUBLIC
    ${CMSIS_NN_PATH}/Include
    ${CMSIS_NN_PATH}/../DSP/Include
    ${CMSIS_NN_PATH}/../Core/Include
  )

  target_link_libraries(qualia-neuralnetwork PRIVATE
    cmsis-nn
  )
endif()

target_compile_features(qualia-neuralnetwork PRIVATE
  cxx_std_17
)

target_include_directories(qualia-neuralnetwork PRIVATE
  ${MODEL_DIR}/include
)
target_include_directories(qualia-neuralnetwork PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}
)
