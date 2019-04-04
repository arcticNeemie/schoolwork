# General NVCC Makefile
CC	= /usr/local/cuda-10.0/bin/nvcc #Points to NVCC on lab computers
LDFLAGS = -L /usr/local/cuda-10.0/lib64 #Included Libraries
IFLAGS 	= -I/usr/local/cuda-10.0/samples/common/inc #Included sample Files

all: cuda_reduction

cuda_reduction: cuda_reduction.cu
	$(CC) cuda_reduction.cu $(LDFLAGS) $(IFLAGS) -c $<
	$(CC) cuda_reduction.o  $(LDFLAGS) $(IFLAGS) -o cuda_reduction

clean:
	$(RM) cuda_reduction *.o *.~
