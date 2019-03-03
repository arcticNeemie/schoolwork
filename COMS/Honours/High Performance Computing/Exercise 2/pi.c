/* 
This program will numerically compute the integral of
                  4/(1+x*x)				  
from 0 to 1.  The value of this integral is pi. 
It uses the timer from the OpenMP runtime library
*/
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
double compute_pi(double step);
double compute_piI(double step);
void usage(char prog_name[]);
static long num_steps = 1000000;
#define NUM_THREADS 4

int main (int argc, char **argv)
{
	double start_time, run_time=0, pi, step;
	double start_timeI, run_timeI=0, piI;
	int iter;

	if (argc != 2) {
		usage(argv[0]);
		exit (-1);
	}
	
	iter=atoi(argv[1]);
	step = 1.0/(double)num_steps;
	for(int i=0; i<iter; i++){
		start_time = omp_get_wtime();
		pi=compute_pi(step);
		run_time += omp_get_wtime() - start_time;

		start_timeI = omp_get_wtime();
		piI=compute_piI(step);
		run_timeI += omp_get_wtime() - start_timeI;
	}
	printf("\n pi with %ld steps is %f in %f seconds\n",num_steps,pi,run_time/iter);
	printf("\n pi with %ld steps is %f in %f seconds\n",num_steps,piI,run_timeI/iter);
}	  
double compute_pi(double step){
	double pi, x, sum=0.0;
	for (int i=1;i<= num_steps; i++){
		x = (i-0.5)*step;
		sum = sum + 4.0/(1.0+x*x);
	}
	pi = step * sum;
	return pi;
}

double compute_piI(double step){
	double pi = 0.0;
	double sum[NUM_THREADS];
	int nthreads;
	omp_set_num_threads(NUM_THREADS);
	#pragma omp parallel
	{
		int i, id, tthreads;
		double x;
		tthreads = omp_get_num_threads();
		id = omp_get_thread_num();
		if(id==0){
			nthreads = tthreads;
		}
		for(id=0,sum[id]=0.0;i<num_steps;i+=tthreads){
			x = (i-0.5)*step;
			sum[id] += 4.0/(1.0+x*x);
		}
	}
	for(int i=0;i<nthreads;i++){
		pi += sum[i]*step;
	}
	return pi;
}


/*--------------------------------------------------------------------
 * Function:    usage
 * Purpose:     Print command line for function
 * In arg:      prog_name
 */
void usage(char prog_name[]) {
   fprintf(stderr, "usage:  %s <number of times to run>\n", prog_name);
} /* usage */


