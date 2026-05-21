cmake_minimum_required(VERSION 3.13)

project(NucleoL4R5ZI C CXX ASM)

set(Python3_FIND_STRATEGY "LOCATION")
set(Python3_FIND_UNVERSIONED_NAMES FIRST)
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

set(NucleoL4R5ZI_CFLAGS
  "-mcpu=cortex-m4"
  "-mfpu=fpv4-sp-d16"
  "-mfloat-abi=hard"
  "-mthumb"

  "-Wall"
  "-Wextra"
  "-Werror=double-promotion"

  "-fdata-sections"
  "-ffunction-sections"

   $<$<COMPILE_LANGUAGE:CXX>:-fno-exceptions>
   $<$<COMPILE_LANGUAGE:CXX>:-fno-rtti>

  "-Ofast"
)

set(CORE_CLOCK_48MHZ False CACHE BOOL "Set core frequency to 48MHz instead of 80MHz")

add_compile_options(
  ${NucleoL4R5ZI_CFLAGS}
)

add_link_options(
  ${NucleoL4R5ZI_CFLAGS}

  "-T" "${CMAKE_CURRENT_SOURCE_DIR}/STM32L4R5XX_FLASH.ld"
  "-z" "separate-code"
  "-Wl,--print-memory-usage"
)

add_subdirectory(${LIBQUALIA_NEURALNETWORK_SOURCE_DIR} ${CMAKE_CURRENT_BINARY_DIR}/libqualia-neuralnetwork)

add_executable(${PROJECT_NAME}
  Core/Src/main.c
  Core/Src/stm32l4xx_it.c
  Core/Src/stm32l4xx_hal_msp.c
  Drivers/STM32L4xx_HAL_Driver/Src/stm32l4xx_hal_uart.c
  Drivers/STM32L4xx_HAL_Driver/Src/stm32l4xx_hal_uart_ex.c
  Drivers/STM32L4xx_HAL_Driver/Src/stm32l4xx_hal.c
  Drivers/STM32L4xx_HAL_Driver/Src/stm32l4xx_hal_rcc.c
  Drivers/STM32L4xx_HAL_Driver/Src/stm32l4xx_hal_rcc_ex.c
  Drivers/STM32L4xx_HAL_Driver/Src/stm32l4xx_hal_flash.c
  Drivers/STM32L4xx_HAL_Driver/Src/stm32l4xx_hal_flash_ex.c
  Drivers/STM32L4xx_HAL_Driver/Src/stm32l4xx_hal_flash_ramfunc.c
  Drivers/STM32L4xx_HAL_Driver/Src/stm32l4xx_hal_gpio.c
  Drivers/STM32L4xx_HAL_Driver/Src/stm32l4xx_hal_i2c.c
  Drivers/STM32L4xx_HAL_Driver/Src/stm32l4xx_hal_i2c_ex.c
  Drivers/STM32L4xx_HAL_Driver/Src/stm32l4xx_hal_dma.c
  Drivers/STM32L4xx_HAL_Driver/Src/stm32l4xx_hal_dma_ex.c
  Drivers/STM32L4xx_HAL_Driver/Src/stm32l4xx_hal_pcd.c
  Drivers/STM32L4xx_HAL_Driver/Src/stm32l4xx_hal_pcd_ex.c
  Drivers/STM32L4xx_HAL_Driver/Src/stm32l4xx_hal_pwr.c
  Drivers/STM32L4xx_HAL_Driver/Src/stm32l4xx_hal_pwr_ex.c
  Drivers/STM32L4xx_HAL_Driver/Src/stm32l4xx_hal_cortex.c
  Drivers/STM32L4xx_HAL_Driver/Src/stm32l4xx_hal_exti.c
  Drivers/STM32L4xx_HAL_Driver/Src/stm32l4xx_ll_usb.c
  Core/Src/system_stm32l4xx.c
  startup_stm32l4r5xx.s
)

target_include_directories(${PROJECT_NAME} PUBLIC
  Core/Inc
  Drivers/STM32L4xx_HAL_Driver/Inc
  Drivers/STM32L4xx_HAL_Driver/Inc/Legacy
  Drivers/CMSIS/Device/ST/STM32L4xx/Include
  Drivers/CMSIS/Include
)

target_link_libraries(${PROJECT_NAME} PUBLIC
  qualia-neuralnetwork
)

target_compile_definitions(${PROJECT_NAME} PUBLIC
  "USE_HAL_DRIVER"
  "STM32L4R5xx"
)
