#******************************************************************************
#
# Makefile - Rules for building the libraries, examples and docs.
#
# Copyright (c) 2019, Ambiq Micro
# All rights reserved.
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
# 
# 1. Redistributions of source code must retain the above copyright notice,
# this list of conditions and the following disclaimer.
# 
# 2. Redistributions in binary form must reproduce the above copyright
# notice, this list of conditions and the following disclaimer in the
# documentation and/or other materials provided with the distribution.
# 
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from this
# software without specific prior written permission.
# 
# Third party software included in this distribution is subject to the
# additional license terms as defined in the /docs/licenses directory.
# 
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#
# This is part of revision v2.0.0 of the AmbiqSuite Development Package.
#
#******************************************************************************
TARGET := main
COMPILERNAME := gcc
PROJECT := ${TARGET}_gcc
OUT := bin

SHELL:=/bin/bash

#### Bootloader Configuration
SERIAL_PORT ?= /dev/ttyUSB0# Or perhaps a Unix-like machine might put '/dev/cu.usbserial-1410' 
ARTEMIS_LOADER_CMD =#for example: "C:\Users\you\~\artemis_uart_loader.exe"

#### Setup ####

TOOLCHAIN ?= arm-none-eabi
PART = apollo3
CPU = cortex-m4
FPU = fpv4-sp-d16
# Default to FPU hardware calling convention.  However, some customers and/or
# applications may need the software calling convention.
#FABI = softfp
FABI = hard

LINKER_FILE := ./linker.ld
STARTUP_FILE := ./startup_$(COMPILERNAME).c

#### Required Executables ####
CC = $(TOOLCHAIN)-gcc
CXX = $(TOOLCHAIN)-g++
GCC = $(TOOLCHAIN)-gcc
CPP = $(TOOLCHAIN)-cpp
LD = $(TOOLCHAIN)-ld
CP = $(TOOLCHAIN)-objcopy
OD = $(TOOLCHAIN)-objdump
RD = $(TOOLCHAIN)-readelf
AR = $(TOOLCHAIN)-gcc-ar
SIZE = $(TOOLCHAIN)-size
RM = $(shell which rm 2>/dev/null)

EXECUTABLES = CC LD CP OD AR RD SIZE GCC
K := $(foreach exec,$(EXECUTABLES),\
        $(if $(shell which $($(exec)) 2>/dev/null),,\
        $(info $(exec) not found on PATH ($($(exec))).)$(exec)))
$(if $(strip $(value K)),$(info Required Program(s) $(strip $(value K)) not found))

ifneq ($(strip $(value K)),)
all:
	$(info Tools $(TOOLCHAIN)-$(COMPILERNAME) not installed.)
else

DEFINES = -DPART_$(PART)
DEFINES+= -DAM_PACKAGE_BGA
DEFINES+= -DAM_PART_APOLLO3
# CMSIS-NN optimizations
ifeq ($(WITH_CMSIS_NN),1)
DEFINES+= -DWITH_CMSIS_NN
DEFINES+= -DARM_MATH_DSP
endif

INCLUDES = -I../src
INCLUDES+= -I../third_party/SparkFun_Apollo3_AmbiqSuite_BSPs/edge/bsp
INCLUDES+= -I$(SWROOT)/CMSIS/AmbiqMicro/Include
INCLUDES+= -I$(SWROOT)/mcu/apollo3
INCLUDES+= -I$(SWROOT)/CMSIS/ARM/Include
INCLUDES+= -I$(SWROOT)/utils
INCLUDES+= -I$(SWROOT)/devices
INCLUDES+= -I$(MODELDIR)
INCLUDES+= -I$(MODELDIR)/include
INCLUDES+= -I../../third_party/cmsis/CMSIS/NN/Include
INCLUDES+= -I../../third_party/cmsis/CMSIS/DSP/Include

VPATH = ../src

HAL_SRC = $(SWROOT)/mcu/apollo3/hal/am_hal_cachectrl.c
HAL_SRC += $(SWROOT)/mcu/apollo3/hal/am_hal_clkgen.c
HAL_SRC += $(SWROOT)/mcu/apollo3/hal/am_hal_interrupt.c
HAL_SRC += $(SWROOT)/mcu/apollo3/hal/am_hal_flash.c
HAL_SRC += $(SWROOT)/mcu/apollo3/hal/am_hal_gpio.c
HAL_SRC += $(SWROOT)/mcu/apollo3/hal/am_hal_itm.c
HAL_SRC += $(SWROOT)/mcu/apollo3/hal/am_hal_queue.c
HAL_SRC += $(SWROOT)/mcu/apollo3/hal/am_hal_pwrctrl.c
HAL_SRC += $(SWROOT)/mcu/apollo3/hal/am_hal_reset.c
HAL_SRC += $(SWROOT)/mcu/apollo3/hal/am_hal_rtc.c
HAL_SRC += $(SWROOT)/mcu/apollo3/hal/am_hal_uart.c

BSP_SRC = ../third_party/SparkFun_Apollo3_AmbiqSuite_BSPs/edge/bsp/am_bsp.c
BSP_SRC += ../third_party/SparkFun_Apollo3_AmbiqSuite_BSPs/edge/bsp/am_bsp_pins.c

# Optimized INT16 functions
CMSIS_SRC = ../../third_party/cmsis/CMSIS/NN/Source/ConvolutionFunctions/arm_convolve_HWC_q15_basic_nonsquare.c
CMSIS_SRC += ../../third_party/cmsis/CMSIS/NN/Source/ConvolutionFunctions/arm_convolve_HWC_q15_fast_nonsquare.c
CMSIS_SRC += ../../third_party/cmsis/CMSIS/NN/Source/ActivationFunctions/arm_relu_q15.c
CMSIS_SRC += ../../third_party/cmsis/CMSIS/NN/Source/FullyConnectedFunctions/arm_fully_connected_q15.c
#CMSIS_SRC += ../../third_party/cmsis/CMSIS/NN/Source/PoolingFunctions/arm_pool_q15_HWC_nonsquare.c

# Optimized INT8 functions
CMSIS_SRC += ../../third_party/cmsis/CMSIS/NN/Source/ConvolutionFunctions/arm_convolve_HWC_q7_basic_nonsquare.c
CMSIS_SRC += ../../third_party/cmsis/CMSIS/NN/Source/ConvolutionFunctions/arm_convolve_HWC_q7_fast_nonsquare.c
CMSIS_SRC += ../../third_party/cmsis/CMSIS/NN/Source/NNSupportFunctions/arm_q7_to_q15_no_shift.c
CMSIS_SRC += ../../third_party/cmsis/CMSIS/NN/Source/ConvolutionFunctions/arm_nn_mat_mult_kernel_q7_q15.c
CMSIS_SRC += ../../third_party/cmsis/CMSIS/NN/Source/ConvolutionFunctions/arm_nn_mat_mult_kernel_q7_q15_reordered.c
CMSIS_SRC += ../../third_party/cmsis/CMSIS/NN/Source/ActivationFunctions/arm_relu_q7.c
CMSIS_SRC += ../../third_party/cmsis/CMSIS/NN/Source/FullyConnectedFunctions/arm_fully_connected_q7.c
CMSIS_SRC += ../../third_party/cmsis/CMSIS/NN/Source/NNSupportFunctions/arm_q7_to_q15_reordered_no_shift.c
#CMSIS_SRC += ../../third_party/cmsis/CMSIS/NN/Source/PoolingFunctions/arm_pool_q7_HWC_nonsquare.c

SRC = startup_gcc.c
SRC += main.cpp
SRC += serial.cpp
SRC += $(SWROOT)/utils/am_util_stdio.c
SRC += $(MODELDIR)/model.c
SRC += $(HAL_SRC)
SRC += $(BSP_SRC)
# CMSIS-NN optimizations
ifeq ($(WITH_CMSIS_NN),1)
SRC += $(CMSIS_SRC)
endif

CSRC = $(filter %.c,$(SRC))
CXXSRC = $(filter %.cpp,$(SRC))
ASRC = $(filter %.s,$(SRC))

OBJS = $(CSRC:%.c=$(OUT)/%.o)
OBJS+= $(CXXSRC:%.cpp=$(OUT)/%.o)
OBJS+= $(ASRC:%.s=$(OUT)/%.o)

DEPS = $(CSRC:%.c=$(OUT)/%.d)
DEPS+= $(CXXSRC:%.cpp=$(OUT)/%.d)
DEPS+= $(ASRC:%.s=$(OUT)/%.d)

#LIBS = ../third_party/SparkFun_Apollo3_AmbiqSuite_BSPs/edge/bsp/gcc/bin/libam_bsp.a
#LIBS += $(SWROOT)/mcu/apollo3/hal/gcc/bin/libam_hal.a


## Common C/C++ flags
CFLAGS = -mthumb -mcpu=$(CPU) -mfpu=$(FPU) -mfloat-abi=$(FABI)
CFLAGS+= -ffunction-sections -fdata-sections
CFLAGS+= -MMD -MP -Wall -Wextra -Werror=double-promotion
#CFLAGS+= -fgraphite-identity -floop-nest-optimize -floop-parallelize-all
CFLAGS+= -Ofast
CFLAGS+= --specs=nano.specs
CFLAGS+= -mlong-calls #-DNN_CODE_IN_RAM
#CFLAGS+= -flto
CFLAGS+= $(DEFINES)
CFLAGS+= $(INCLUDES)
CFLAGS+= -save-temps=obj
CFLAGS+= -include"$(MODELDIR)/include/defines.h"

CXXFLAGS = $(CFLAGS)
## C++ flags
CXXFLAGS+= -std=c++20
CXXFLAGS+= -fno-exceptions -fno-rtti

## C flags
CLFAGS+= -std=c17

# Additional user specified CFLAGS
CFLAGS+=$(EXTRA_CFLAGS)
CXXFLAGS+=$(EXTRA_CXXFLAGS)

# Copy C++ flags to linker flags for possible LTO support
LFLAGS = $(CXXFLAGS)
## Link flags
LFLAGS+= -nostartfiles -static
LFLAGS+= --specs=nosys.specs
#LFLAGS+= --specs=nano.specs -u _printf_float
LFLAGS+= -Wl,--gc-sections,--entry,Reset_Handler,-Map,$(OUT)/$(TARGET).map
#LFLAGS+= -Wl,--start-group -lm -lc -lgcc $(LIBS) -Wl,--end-group
LFLAGS+= -Wl,--start-group $(LIBS) -Wl,--end-group
##LFLAGS+= -Wl,--wrap=malloc -Wl,--wrap=free #Prevent heap allocation as we have no heap
LFLAGS+= -Wl,--print-memory-usage

CPFLAGS = -Obinary

ODFLAGS = -S

#### Rules ####
all: directories $(OUT)/$(TARGET).bin

directories: $(OUT)

$(OUT):
	@mkdir -p $@

$(OUT)/%.o: %.c $(OUT)/%.d
	@echo " Compiling $(COMPILERNAME) $<" ;\
	$(CC) -c $(CFLAGS) $< -o $@

$(OUT)/%.o: %.cpp $(OUT)/%.d
	@echo " Compiling $(COMPILERNAME) $<" ;\
	$(CXX) -c $(CXXFLAGS) $< -o $@

$(OUT)/%.o: %.s $(OUT)/%.d
	@echo " Assembling $(COMPILERNAME) $<" ;\
	$(CC) -c $(CFLAGS) $< -o $@

$(OUT)/$(TARGET).axf: $(OBJS) $(LIBS)
	@echo " Linking $(COMPILERNAME) $@" ;\
	$(CXX) -Wl,-T,$(LINKER_FILE) -o $@ $(OBJS) $(LFLAGS)

$(OUT)/$(TARGET).bin: $(OUT)/$(TARGET).axf
	@echo " Copying $(COMPILERNAME) $@..." ;\
	$(CP) $(CPFLAGS) $< $@ ;\
	$(OD) $(ODFLAGS) $< > $(OUT)/$(TARGET).lst

clean:
	@echo "Cleaning..." ;\
	$(RM) -f $(OBJS) $(DEPS) \
	    $(OUT)/$(TARGET).bin $(OUT)/$(TARGET).axf \
	    $(OUT)/$(TARGET).lst $(OUT)/$(TARGET).map

$(OUT)/%.d:
	@mkdir -p $(dir $@)

../third_party/SparkFun_Edge_BSP/bsp/gcc/bin/libam_bsp.a:
	$(MAKE) -C ../third_party/SparkFun_Edge_BSP/bsp

$(SWROOT)/mcu/apollo3/hal/gcc/bin/libam_hal.a:
	$(MAKE) -C $(SWROOT)/mcu/apollo3/hal

# Automatically include any generated dependencies
-include $(DEPS)
endif

bootload:
	python3 ../third_party/SparkFun_Apollo3_AmbiqSuite_BSPs/common/tools_sfe/asb/asb.py --bin $(OUT)/$(TARGET).bin --load-address-blob 0x20000 --magic-num 0xCB -o $(OUT)/$(TARGET) --version 0x0 --load-address-wired 0xC000 -i 6 --options 0x1 -b 921600 -port $(SERIAL_PORT) -r 2 -v

.PHONY: all clean directories bootload
