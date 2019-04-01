INC="../common/inc"
NVCCFLAGS=-I$(INC)
OMPFLAG=-fopenmp
CC=gcc
NVCC=nvcc
CCFLAGS=-g -Wall

all: cuda_reduction

cuda_reduction: cuda_reduction.cu
	$(NVCC) $(NVCCFLAGS) cuda_reduction.cu -o cuda_reduction

clean:
	rm cuda_reduction
