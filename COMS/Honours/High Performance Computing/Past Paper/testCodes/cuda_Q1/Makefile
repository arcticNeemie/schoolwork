# General NVCC Makefile
CC	= /usr/local/cuda-10.0/bin/nvcc #Points to NVCC on lab computers
LDFLAGS = -L /usr/local/cuda-10.0/lib64 #Included Libraries
IFLAGS 	= -I/usr/local/cuda-10.0/samples/common/inc #Included sample Files

all: gray2bw

gray2bw: gray2bw.cu
	$(CC) gray2bw.cu $(LDFLAGS) $(IFLAGS) -c $<
	$(CC) gray2bw.o  $(LDFLAGS) $(IFLAGS) -o gray2bw

clean:
	$(RM) gray2bw *.o *.~
