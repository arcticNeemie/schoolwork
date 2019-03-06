INC="./inc"
FLAGS=-I$(INC)
OMPFLAG=-fopenmp
CC=gcc

all: serial_kNN

serial_KNN: serial_kNN.c
	$(CC) serial_kNN.c -o serial_kNN

clean:
	rm serial_KNN
