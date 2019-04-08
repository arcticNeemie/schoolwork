CC	= /usr/local/cuda-10.0/bin/nvcc
LDFLAGS = -L /usr/local/cuda-10.0/lib64
IFLAGS 	= -I/usr/local/cuda-10.0/samples/common/inc

all: convolution simpleTexture

convolution: convolution.cu
	$(CC) convolution.cu $(LDFLAGS) $(IFLAGS) -c $<
	$(CC) convolution.o  $(LDFLAGS) $(IFLAGS) -o convolution

simpleTexture: simpleTexture.cu
	$(CC) simpleTexture.cu $(LDFLAGS) $(IFLAGS) -c $<
	$(CC) simpleTexture.o  $(LDFLAGS) $(IFLAGS) -o simpleTexture

clean:
	$(RM) convolution simpleTexture*.o *.~
