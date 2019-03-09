INC="./inc"
FLAGS=-I$(INC)
OMPFLAG=-fopenmp
CC=gcc
MATH= -lm

all: serial_kNN parallel_kNN.c

serial_kNN: serial_kNN.c
	$(CC) serial_kNN.c -o serial_kNN $(OMPFLAG) $(MATH)

parallel_kNN: parallel_kNN.c
	$(CC) parallel_kNN.c -o parallel_kNN $(OMPFLAG) $(MATH)

clean:
	rm serial_kNN parallel_kNN
