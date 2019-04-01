INC="../common/inc"
NVCCFLAGS=-I$(INC)
OMPFLAG=-fopenmp
CC=gcc
NVCC=nvcc
CCFLAGS=-g -Wall

all: gray2bw

gray2bw: gray2bw.cu
	$(NVCC) $(NVCCFLAGS) gray2bw.cu -o gray2bw

clean:
	rm gray2bw
