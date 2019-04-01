OMPFLAG=-fopenmp
CC=gcc
NVCC=nvcc
CCFLAGS=-g -Wall

all: linked

linked: linked.c
	$(CC) $(CCFLAGS) $(OMPFLAG) linked.c -o linked

clean:
	rm linked
