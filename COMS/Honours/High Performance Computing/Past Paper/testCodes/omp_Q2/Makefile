OMPFLAG=-fopenmp
CC=gcc
NVCC=nvcc
CCFLAGS=-g -Wall

all: mat_vect

mat_vect: mat_vect.c
	$(CC) $(CCFLAGS) $(OMPFLAG) mat_vect.c -o mat_vect

clean:
	rm mat_vect
