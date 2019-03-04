/*
This program implements a parallel Seive of Eratosthenes
*/
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <math.h>
void usage(char prog_name[]);
int* sieve_serial(int n);
int* sieve_parallel(int n);
int countPrimes(int* primes, int n);

int main (int argc, char **argv)
{
  double start_time, run_time=0;

  if (argc != 3) {
		usage(argv[0]);
		exit (-1);
	}
  int n=atoi(argv[1]);
  int t=atoi(argv[2]);

  int* primes;

  omp_set_num_threads(t);
  //Serial
  start_time = omp_get_wtime();
  primes = sieve_serial(n);
  run_time = omp_get_wtime() - start_time;
  int x = countPrimes(primes,n);
  printf("\n Serial: Found %i primes in %f seconds\n",x,run_time);


}

int* sieve_serial(int n){
  int* primes = malloc(n * sizeof(int));
  //Initialize
  for(int k=0;k<n;k++){
    primes[k] = 1;
  }
  //Algorithm
  int i=0;
  while(i<=sqrt(n)){
    int j = i*i;
    while(j<=n){
      printf("i = %i and j=%i\n",i,j);
      //primes[j] = 0;
      j+=i;
    }
    i++;
  }
  return primes;
}

int countPrimes(int* primes, int n){
  int sum = 0;
  for(int i=0;i<n;i++){
    if(primes[i]==1){
      sum++;
    }
  }
  return sum;
}


/*--------------------------------------------------------------------
 * Function:    usage
 * Purpose:     Print command line for function
 * In arg:      prog_name
 */
void usage(char prog_name[]) {
   fprintf(stderr, "usage:  %s <length of sieve> <number of threads>\n", prog_name);
} /* usage */
