cmake_minimum_required(VERSION 3.13)

project(NucleoH7S3L8 C CXX ASM)

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

set(CMAKE_SYSTEM_PROCESSOR arm)

# which compilers to use for C and C++
set(CMAKE_C_COMPILER arm-none-eabi-gcc)
set(CMAKE_CXX_COMPILER arm-none-eabi-g++)
set(CMAKE_ASM_COMPILER arm-none-eabi-gcc)

SET(CMAKE_ASM_FLAGS "-x assembler-with-cpp")

# here is the target environment located
#set(CMAKE_FIND_ROOT_PATH  )

# adjust the default behaviour of the FIND_XXX() commands:
# search headers and libraries in the target environment, search 
# programs in the host environment
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)


set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)

set(NucleoH7S3L8_CFLAGS
  "-mcpu=cortex-m7"
  "-mfpu=fpv5-d16"
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

add_compile_options(
  ${NucleoH7S3L8_CFLAGS}
)

add_link_options(
  ${NucleoH7S3L8_CFLAGS}

  "-specs=nosys.specs"
  "-T" "${CMAKE_CURRENT_SOURCE_DIR}/Makefile/Boot/stm32h7s3xx_flash.ld"
  "-z" "separate-code"
  "-Wl,--print-memory-usage"
)

add_subdirectory(${LIBQUALIA_NEURALNETWORK_SOURCE_DIR} ${CMAKE_CURRENT_BINARY_DIR}/libqualia-neuralnetwork)

add_executable(${PROJECT_NAME}
  Boot/Core/Src/main.c
  Boot/Core/Src/stm32h7rsxx_it.c
  Boot/Core/Src/stm32h7rsxx_hal_msp.c
  Drivers/STM32H7RSxx_HAL_Driver/Src/stm32h7rsxx_hal_cortex.c
  Drivers/STM32H7RSxx_HAL_Driver/Src/stm32h7rsxx_hal_dma.c
  Drivers/STM32H7RSxx_HAL_Driver/Src/stm32h7rsxx_hal_dma_ex.c
  Drivers/STM32H7RSxx_HAL_Driver/Src/stm32h7rsxx_hal_rcc.c
  Drivers/STM32H7RSxx_HAL_Driver/Src/stm32h7rsxx_hal_rcc_ex.c
  Drivers/STM32H7RSxx_HAL_Driver/Src/stm32h7rsxx_hal_flash.c
  Drivers/STM32H7RSxx_HAL_Driver/Src/stm32h7rsxx_hal_flash_ex.c
  Drivers/STM32H7RSxx_HAL_Driver/Src/stm32h7rsxx_hal_gpio.c
  Drivers/STM32H7RSxx_HAL_Driver/Src/stm32h7rsxx_hal_pwr.c
  Drivers/STM32H7RSxx_HAL_Driver/Src/stm32h7rsxx_hal_pwr_ex.c
  Drivers/STM32H7RSxx_HAL_Driver/Src/stm32h7rsxx_hal.c
  Drivers/STM32H7RSxx_HAL_Driver/Src/stm32h7rsxx_hal_exti.c
  Drivers/STM32H7RSxx_HAL_Driver/Src/stm32h7rsxx_hal_tim.c
  Drivers/STM32H7RSxx_HAL_Driver/Src/stm32h7rsxx_hal_tim_ex.c
  Drivers/STM32H7RSxx_HAL_Driver/Src/stm32h7rsxx_hal_uart.c
  Drivers/STM32H7RSxx_HAL_Driver/Src/stm32h7rsxx_hal_uart_ex.c
  Boot/Core/Src/system_stm32h7rsxx.c
  Makefile/Boot/startup_stm32h7s3xx.s
)

target_include_directories(${PROJECT_NAME} PUBLIC
  Boot/Core/Inc
  Drivers/STM32H7RSxx_HAL_Driver/Inc
  Drivers/STM32H7RSxx_HAL_Driver/Inc/Legacy
  Drivers/CMSIS/Device/ST/STM32H7RSxx/Include
  Drivers/CMSIS/Include
)

target_link_libraries(${PROJECT_NAME} PUBLIC
  qualia-neuralnetwork
)

target_compile_definitions(${PROJECT_NAME} PUBLIC
  "USE_HAL_DRIVER"
  "STM32H7S3xx"
)

