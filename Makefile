NVCC ?= nvcc
PKG_CONFIG ?= pkg-config
CUDA_ARCH ?= sm_75
TARGET := cube

SDL_CFLAGS := $(shell $(PKG_CONFIG) --cflags sdl3)
SDL_LIBS := $(shell $(PKG_CONFIG) --libs sdl3)

NVCCFLAGS ?= -O2
COMMON_FLAGS := -std=c++17 -arch=$(CUDA_ARCH) \
	-Xcompiler=-Wall,-Wextra $(SDL_CFLAGS)

SOURCES := cube.cu sdl.cpp
HEADERS := gpu_renderer.h

.PHONY: all run clean

all: $(TARGET)

$(TARGET): $(SOURCES) $(HEADERS)
	$(NVCC) $(COMMON_FLAGS) $(NVCCFLAGS) $(SOURCES) -o $@ $(SDL_LIBS)

run: $(TARGET)
	./$(TARGET)

clean:
	$(RM) $(TARGET) *.o
