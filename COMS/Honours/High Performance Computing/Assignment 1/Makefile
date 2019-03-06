INC="./inc"
FLAGS=-I$(INC)
OMPFLAG=-fopenmp
CC=gcc
MATH= -lm

all: serial_kNN

serial_kNN: serial_kNN.c
	$(CC) serial_kNN.c -o serial_kNN $(MATH)

clean:
	rm serial_KNN
