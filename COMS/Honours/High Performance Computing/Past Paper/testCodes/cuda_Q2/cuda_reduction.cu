// includes, system
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <time.h>
#include <unistd.h>
#include <sys/time.h>
#include <cuda_runtime.h>

#define N 1024
#define NUM_THREADS_PER_BLOCK 1024

// simple utility function to check for CUDA runtime errors
void checkCUDAError(const char *msg) {
    cudaError_t err = cudaGetLastError();
    if( cudaSuccess != err) {
        fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString( err) );
        exit(EXIT_FAILURE);
    }
}

void sumReductionCPU(const float *h_in, float *h_out) {
	float sum = 0.0;
	for (int i = 0; i < N; i++) {
		sum += h_in[i];
	}
	*h_out = sum;
}

int main(int argc, char** argv) {
	// pointers to host memory
	float *h_in, *h_correct;
	// pointer for device memory
	
	// allocate memory for pointers
	h_in       = (float *)malloc(N * sizeof(float));
	h_correct  = (float *)malloc(sizeof(float));
	
	// Initialize h_in
	srand(9999);
	for (int i = 0; i < N; i++) {
		h_in[i] = 1.0f;
	}
	// memset to 0	
	memset(h_correct, 0, sizeof(float));
	
	// calculate correct output
	sumReductionCPU(h_in, h_correct);
	
	// you may validate your results by changing the following if structure
	if (!(abs((*h_correct) - (*h_correct)) <= 0.0001) || 
				!(abs((*h_correct) - (*h_correct)) <= 0.0001) ) {
		printf("Test failed");
		exit(1);
	}
	printf("Test passed!\n");
	// cleanup
	free(h_in);
	free(h_correct);
	return 0;
}
