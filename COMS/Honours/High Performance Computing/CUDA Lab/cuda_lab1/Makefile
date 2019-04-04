# General NVCC Makefile
CC	= /usr/local/cuda-10.0/bin/nvcc #Points to NVCC on lab computers
LDFLAGS = -L /usr/local/cuda-10.0/lib64 #Included Libraries
IFLAGS 	= -I/usr/local/cuda-10.0/samples/common/inc #Included sample Files

all: double vector_addition matrix_op1

double: double.cu
	$(CC) double.cu $(LDFLAGS) $(IFLAGS) -c $<
	$(CC) double.o  $(LDFLAGS) $(IFLAGS) -o double

vector_addition: vector_addition.cu
	$(CC) vector_addition.cu $(LDFLAGS) $(IFLAGS) -c $<
	$(CC) vector_addition.o  $(LDFLAGS) $(IFLAGS) -o vector_addition

matrix_op1: matrix_op1.cu
	$(CC) matrix_op1.cu $(LDFLAGS) $(IFLAGS) -c $<
	$(CC) matrix_op1.o  $(LDFLAGS) $(IFLAGS) -o matrix_op1

# target_name: object_name.cu
	#$(CC) object_name.cu $(LDFLAGS) $(IFLAGS) -c $<
	#$(CC) object_name.o $(LDFLAGS) $(IFLAGS) -o target_name

clean:
	$(RM) double matrix_op1 vector_addition*.o *.~ #add target_name
