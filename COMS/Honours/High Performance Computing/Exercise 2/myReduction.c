/*
This program implements a parallel summation without the default reduction
*/
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
void usage(char prog_name[]);
int sum_serial(int num);
int sum_parallel(int num);
int sum_custom(int num);
#define NUM_THREADS 4

int main (int argc, char **argv)
{
  double start_time, run_time=0;
  int sum;

  if (argc != 3) {
		usage(argv[0]);
		exit (-1);
	}
  int num=atoi(argv[1]);
  int iter=atoi(argv[2]);

  //Serial
  start_time = omp_get_wtime();
  for(int j=0; j<iter; j++){
		sum = sum_serial(num);
	}
  run_time = omp_get_wtime() - start_time;
  printf("\n Serial: Summed to %i %i times in %f seconds\n",sum,iter,run_time);

  //Inbuilt function
  start_time = omp_get_wtime();
  for(int j=0; j<iter; j++){
		sum = sum_parallel(num);
	}
  run_time = omp_get_wtime() - start_time;
  printf("\n Inbuilt Parallel: Summed to %i %i times in %f seconds\n",sum,iter,run_time);

  //Custom parallel
  start_time = omp_get_wtime();
  for(int j=0; j<iter; j++){
		sum = sum_custom(num);
	}
  run_time = omp_get_wtime() - start_time;
  printf("\n Custom Parallel: Summed to %i %i times in %f seconds\n",sum,iter,run_time);
}

int sum_serial(int num){
  int sum = 0;
  for(int i=0;i<num;i++){
    sum += i;
  }
  return sum;
}

int sum_parallel(int num){
  int sum = 0;
  int i;
  omp_set_num_threads(NUM_THREADS);
  #pragma omp parallel for reduction(+:sum)
    for(i=0;i<num;i++){
      sum += i;
    }
  return sum;
}

int sum_custom(int num){
  int buffer = 16;
  int sum[buffer*NUM_THREADS], total = 0, id, i;
  //Initialize sum[]
  for(int k=0;k<NUM_THREADS;k++){
    sum[buffer*k] = 0;
  }
  omp_set_num_threads(NUM_THREADS);
  #pragma omp parallel for
    for(i=0;i<num;i++){
      id = omp_get_thread_num();
      sum[buffer*id] += i; //Take only every 8 entries to avoid cache coherency false sharing stuff
    }
  for(int j=0;j<NUM_THREADS;j++){
    total += sum[buffer*j];
  }
  return total;
}

/*--------------------------------------------------------------------
 * Function:    usage
 * Purpose:     Print command line for function
 * In arg:      prog_name
 */
void usage(char prog_name[]) {
   fprintf(stderr, "usage:  %s <number to sum to> <number of iterations>\n", prog_name);
} /* usage */
