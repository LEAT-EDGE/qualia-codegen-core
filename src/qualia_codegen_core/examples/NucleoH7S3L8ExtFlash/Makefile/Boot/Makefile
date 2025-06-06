##########################################################################################################################
# File automatically-generated by tool: [projectgenerator] version: [4.5.0-B34] date: [Sat Oct 19 19:03:35 CEST 2024]
##########################################################################################################################

# ------------------------------------------------
# Generic Makefile (based on gcc)
#
# ChangeLog :
#	2017-02-10 - Several enhancements + project update mode
#   2015-07-22 - first version
# ------------------------------------------------

######################################
# target
######################################
TARGET = NucleoH7S3L8ExtFlash_Boot


######################################
# building variables
######################################
# debug build?
DEBUG = 1
# optimization
OPT = -Og


#######################################
# paths
#######################################
# Build path
BUILD_DIR = build

######################################
# source
######################################
# C sources
C_SOURCES =  \
../../Boot/Core/Src/main.c \
../../Boot/Core/Src/extmem_manager.c \
../../Boot/Core/Src/stm32h7rsxx_it.c \
../../Boot/Core/Src/stm32h7rsxx_hal_msp.c \
../../Drivers/STM32H7RSxx_HAL_Driver/Src/stm32h7rsxx_hal_cortex.c \
../../Drivers/STM32H7RSxx_HAL_Driver/Src/stm32h7rsxx_hal_rcc.c \
../../Drivers/STM32H7RSxx_HAL_Driver/Src/stm32h7rsxx_hal_rcc_ex.c \
../../Drivers/STM32H7RSxx_HAL_Driver/Src/stm32h7rsxx_hal_flash.c \
../../Drivers/STM32H7RSxx_HAL_Driver/Src/stm32h7rsxx_hal_flash_ex.c \
../../Drivers/STM32H7RSxx_HAL_Driver/Src/stm32h7rsxx_hal_gpio.c \
../../Drivers/STM32H7RSxx_HAL_Driver/Src/stm32h7rsxx_hal_dma.c \
../../Drivers/STM32H7RSxx_HAL_Driver/Src/stm32h7rsxx_hal_dma_ex.c \
../../Drivers/STM32H7RSxx_HAL_Driver/Src/stm32h7rsxx_hal_pwr.c \
../../Drivers/STM32H7RSxx_HAL_Driver/Src/stm32h7rsxx_hal_pwr_ex.c \
../../Drivers/STM32H7RSxx_HAL_Driver/Src/stm32h7rsxx_hal.c \
../../Drivers/STM32H7RSxx_HAL_Driver/Src/stm32h7rsxx_hal_exti.c \
../../Drivers/STM32H7RSxx_HAL_Driver/Src/stm32h7rsxx_hal_tim.c \
../../Drivers/STM32H7RSxx_HAL_Driver/Src/stm32h7rsxx_hal_tim_ex.c \
../../Drivers/STM32H7RSxx_HAL_Driver/Src/stm32h7rsxx_hal_uart.c \
../../Drivers/STM32H7RSxx_HAL_Driver/Src/stm32h7rsxx_hal_uart_ex.c \
../../Drivers/STM32H7RSxx_HAL_Driver/Src/stm32h7rsxx_hal_xspi.c \
../../Boot/Core/Src/system_stm32h7rsxx.c \
../../Middlewares/ST/STM32_ExtMem_Manager/stm32_extmem.c \
../../Middlewares/ST/STM32_ExtMem_Manager/boot/stm32_boot_xip.c \
../../Middlewares/ST/STM32_ExtMem_Manager/sal/stm32_sal_xspi.c \
../../Middlewares/ST/STM32_ExtMem_Manager/sal/stm32_sal_sd.c \
../../Middlewares/ST/STM32_ExtMem_Manager/nor_sfdp/stm32_sfdp_data.c \
../../Middlewares/ST/STM32_ExtMem_Manager/nor_sfdp/stm32_sfdp_driver.c \
../../Middlewares/ST/STM32_ExtMem_Manager/psram/stm32_psram_driver.c \
../../Middlewares/ST/STM32_ExtMem_Manager/sdcard/stm32_sdcard_driver.c \
../../Middlewares/ST/STM32_ExtMem_Manager/user/stm32_user_driver.c \
../../Boot/Core/Src/sysmem.c \
../../Boot/Core/Src/syscalls.c  

# ASM sources
ASM_SOURCES =  \
startup_stm32h7s3xx.s

# ASM sources
ASMM_SOURCES = 


#######################################
# binaries
#######################################
PREFIX = arm-none-eabi-
# The gcc compiler bin path can be either defined in make command via GCC_PATH variable (> make GCC_PATH=xxx)
# either it can be added to the PATH environment variable.
ifdef GCC_PATH
CC = $(GCC_PATH)/$(PREFIX)gcc
AS = $(GCC_PATH)/$(PREFIX)gcc -x assembler-with-cpp
CP = $(GCC_PATH)/$(PREFIX)objcopy
SZ = $(GCC_PATH)/$(PREFIX)size
else
CC = $(PREFIX)gcc
AS = $(PREFIX)gcc -x assembler-with-cpp
CP = $(PREFIX)objcopy
SZ = $(PREFIX)size
endif
HEX = $(CP) -O ihex
BIN = $(CP) -O binary -S
 
#######################################
# CFLAGS
#######################################
# cpu
CPU = -mcpu=cortex-m7

# fpu
FPU = -mfpu=fpv5-d16

# float-abi
FLOAT-ABI = -mfloat-abi=hard

# mcu
MCU = $(CPU) -mthumb $(FPU) $(FLOAT-ABI)

# macros for gcc
# AS defines
AS_DEFS = 

# C defines
C_DEFS =  \
-DUSE_HAL_DRIVER \
-DSTM32H7S3xx


# AS includes
AS_INCLUDES = 

# C includes
C_INCLUDES =  \
-I../../Boot/Core/Inc \
-I../../Drivers/STM32H7RSxx_HAL_Driver/Inc \
-I../../Drivers/STM32H7RSxx_HAL_Driver/Inc/Legacy \
-I../../Middlewares/ST/STM32_ExtMem_Manager \
-I../../Middlewares/ST/STM32_ExtMem_Manager/boot \
-I../../Middlewares/ST/STM32_ExtMem_Manager/sal \
-I../../Middlewares/ST/STM32_ExtMem_Manager/nor_sfdp \
-I../../Middlewares/ST/STM32_ExtMem_Manager/psram \
-I../../Middlewares/ST/STM32_ExtMem_Manager/sdcard \
-I../../Middlewares/ST/STM32_ExtMem_Manager/user \
-I../../Drivers/CMSIS/Device/ST/STM32H7RSxx/Include \
-I../../Drivers/CMSIS/Include


# compile gcc flags
ASFLAGS = $(MCU) $(AS_DEFS) $(AS_INCLUDES) $(OPT) -Wall -fdata-sections -ffunction-sections

CFLAGS += $(MCU) $(C_DEFS) $(C_INCLUDES) $(OPT) -Wall -fdata-sections -ffunction-sections

ifeq ($(DEBUG), 1)
CFLAGS += -g -gdwarf-2
endif


# Generate dependency information
CFLAGS += -MMD -MP -MF"$(@:%.o=%.d)"


#######################################
# LDFLAGS
#######################################
# link script
LDSCRIPT = stm32h7s3xx_flash.ld

# libraries
LIBS = -lc -lm -lnosys 
LIBDIR = 
LDFLAGS = $(MCU) -specs=nano.specs -T$(LDSCRIPT) $(LIBDIR) $(LIBS) -Wl,-Map=$(BUILD_DIR)/$(TARGET).map,--cref -Wl,--gc-sections

# default action: build all
all: $(BUILD_DIR)/$(TARGET).elf $(BUILD_DIR)/$(TARGET).hex $(BUILD_DIR)/$(TARGET).bin


#######################################
# build the application
#######################################
# list of objects
OBJECTS = $(addprefix $(BUILD_DIR)/,$(notdir $(C_SOURCES:.c=.o)))
vpath %.c $(sort $(dir $(C_SOURCES)))
# list of ASM program objects
OBJECTS += $(addprefix $(BUILD_DIR)/,$(notdir $(ASM_SOURCES:.s=.o)))
vpath %.s $(sort $(dir $(ASM_SOURCES)))
OBJECTS += $(addprefix $(BUILD_DIR)/,$(notdir $(ASMM_SOURCES:.S=.o)))
vpath %.S $(sort $(dir $(ASMM_SOURCES)))

$(BUILD_DIR)/%.o: %.c Makefile | $(BUILD_DIR) 
	$(CC) -c $(CFLAGS) -Wa,-a,-ad,-alms=$(BUILD_DIR)/$(notdir $(<:.c=.lst)) $< -o $@

$(BUILD_DIR)/%.o: %.s Makefile | $(BUILD_DIR)
	$(AS) -c $(CFLAGS) $< -o $@
$(BUILD_DIR)/%.o: %.S Makefile | $(BUILD_DIR)
	$(AS) -c $(CFLAGS) $< -o $@

$(BUILD_DIR)/$(TARGET).elf: $(OBJECTS) Makefile
	$(CC) $(OBJECTS) $(LDFLAGS) -o $@
	$(SZ) $@

$(BUILD_DIR)/%.hex: $(BUILD_DIR)/%.elf | $(BUILD_DIR)
	$(HEX) $< $@
	
$(BUILD_DIR)/%.bin: $(BUILD_DIR)/%.elf | $(BUILD_DIR)
	$(BIN) $< $@	
	
$(BUILD_DIR):
	mkdir $@		

#######################################
# clean up
#######################################
clean:
	-rm -fR $(BUILD_DIR)
  
#######################################
# dependencies
#######################################
-include $(wildcard $(BUILD_DIR)/*.d)

# *** EOF ***