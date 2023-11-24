LOCAL_CFLAGS := -Ofast -ffunction-sections -fdata-sections
LOCAL_PATH := $(call my-dir)/qualia/src
include $(CLEAR_VARS)
LOCAL_MODULE    := qualia
LOCAL_SRC_FILES := main.cpp
include $(BUILD_EXECUTABLE)
