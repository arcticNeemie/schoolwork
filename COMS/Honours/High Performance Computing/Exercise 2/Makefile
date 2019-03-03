INC="./inc"
FLAGS=-I$(INC)
OMPFLAG=-fopenmp
CC=gcc

all: fib pi qsort worksharing1

fib: fib.c
	$(CC) fib.c -o fib $(OMPFLAG)

pi: pi.c
	$(CC) pi.c -o pi $(OMPFLAG)

qsort: qsort.c
	$(CC) qsort.c -o qsort $(OMPFLAG)

worksharing1: worksharing1.c
	$(CC) worksharing1.c -o worksharing1 $(OMPFLAG)

clean:
	rm fib pi qsort worksharing1
