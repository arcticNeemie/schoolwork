INC="./inc"
FLAGS=-I$(INC)
OMPFLAG=-fopenmp
CC=gcc
MATH=-lm

all: fib pi qsort worksharing1 myReduction sieve

fib: fib.c
	$(CC) fib.c -o fib $(OMPFLAG)

pi: pi.c
	$(CC) pi.c -o pi $(OMPFLAG)

qsort: qsort.c
	$(CC) qsort.c -o qsort $(OMPFLAG)

worksharing1: worksharing1.c
	$(CC) worksharing1.c -o worksharing1 $(OMPFLAG)

myReduction: myReduction.c
	$(CC) myReduction.c -o myReduction $(OMPFLAG)

sieve: sieve.c
	$(CC) sieve.c -o sieve $(OMPFLAG) $(MATH)

clean:
	rm fib pi qsort worksharing1 myReduction sieve
