cmake_minimum_required(VERSION 3.13)

project(NucleoL452REP C CXX ASM)

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

set(NucleoL452REP_CFLAGS
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

add_compile_options(
  ${NucleoL452REP_CFLAGS}
)

add_link_options(
  ${NucleoL452REP_CFLAGS}

  "-specs=nosys.specs"
  "-T" "${CMAKE_SOURCE_DIR}/STM32L452RETxP_FLASH.ld"
  "-z" "separate-code"
  "-Wl,--print-memory-usage"
)

add_subdirectory(../libqualia-neuralnetwork ${CMAKE_CURRENT_BINARY_DIR}/libqualia-neuralnetwork)

add_executable(${PROJECT_NAME}
  Core/Src/main.c
  Core/Src/stm32l4xx_it.c
  Core/Src/stm32l4xx_hal_msp.c
  Drivers/STM32L4xx_HAL_Driver/Src/stm32l4xx_hal_tim.c
  Drivers/STM32L4xx_HAL_Driver/Src/stm32l4xx_hal_tim_ex.c
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
  Drivers/STM32L4xx_HAL_Driver/Src/stm32l4xx_hal_pwr.c
  Drivers/STM32L4xx_HAL_Driver/Src/stm32l4xx_hal_pwr_ex.c
  Drivers/STM32L4xx_HAL_Driver/Src/stm32l4xx_hal_cortex.c
  Drivers/STM32L4xx_HAL_Driver/Src/stm32l4xx_hal_exti.c
  Core/Src/system_stm32l4xx.c
  startup_stm32l452xx.s
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
  "STM32L452xx"
)

