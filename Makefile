CUDA_PATH       ?= /usr/local/cuda-9.0
CUDA_INC_PATH   ?= $(CUDA_PATH)/include
CUDA_BIN_PATH   ?= $(CUDA_PATH)/bin
CUDA_LIB_PATH  ?= $(CUDA_PATH)/lib64

NVCC            ?= $(CUDA_BIN_PATH)/nvcc
GCC             ?= g++

GENCODE_SM50    := -gencode arch=compute_50,code=sm_50 -gencode arch=compute_50,code=sm_50
GENCODE_FLAGS   := $(GENCODE_SM50)

LDFLAGS   := -L$(CUDA_LIB_PATH) 
CCFLAGS   := -m64

NVCCFLAGS := -m64 -x cu 


INCLUDES      := -I$(CUDA_INC_PATH) -I. -I.. -I$(CUDA_PATH)/samples/common/inc

all: build

build: exec



conn.o: runner.cpp
	$(NVCC) $(NVCCFLAGS) $(EXTRA_NVCCFLAGS) $(GENCODE_FLAGS) $(INCLUDES) -o $@ -c $<
	
exec: runner.cpp 
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -o $@ $+ $(LDFLAGS) $(EXTRA_LDFLAGS)

run: build
	./exec

clean:
	rm -f exec conn.o *.bin
