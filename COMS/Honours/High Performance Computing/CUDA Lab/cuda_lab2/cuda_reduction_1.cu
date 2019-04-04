// includes, system
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <ctime>
#include <cuda_runtime.h>
#include "../common/helper_cuda.h"
#include "../common/helper_functions.h"

#define N 1024
#define NUM_THREADS_PER_BLOCK 1024

// simple utility function to check for CUDA runtime errors
/*void checkCUDAError(const char *msg) {
    cudaError_t err = cudaGetLastError();
    if( cudaSuccess != err) {
        fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString( err) );
        exit(EXIT_FAILURE);
    }
}*/

static void HandleError( cudaError_t err,
                         const char *file,
                         int line ) {
    if (err != cudaSuccess) {
        printf( "%s in %s at line %d\n", cudaGetErrorString( err ),
                file, line );
        exit( EXIT_FAILURE );
    }
}
#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))

void sumReductionCPU(const float *h_in, float *h_out) {
	float sum = 0.0;
	for (int i = 0; i < N; i++) {
		sum += h_in[i];
	}
	*h_out = sum;
}

__global__ void sumReductionGPU_A(const float *g_in, float *g_out) {

}


__global__ void sumReductionGPU_B(const float *g_in, float *g_out) {

}


int main(int argc, char** argv) {
	// pointers to host memory
	float *h_in, *h_out_A, *h_out_B, *h_correct;
	// pointer for device memory
	float *d_in, *d_out_A, *d_out_B;
	StopWatchInterface *hTimer = NULL;
	double dAvgSecs, hAvgSecs;
	// allocate memory for pointers
	h_in       = (float *)malloc(N * sizeof(float));
	h_out_A      = (float *)malloc(sizeof(float));
	h_out_B      = (float *)malloc(sizeof(float));
	h_correct  = (float *)malloc(sizeof(float));
	// allocate memory for device pointers
	cudaMalloc((void **) &d_in,  N * sizeof(float));
	cudaMalloc((void **) &d_out_A, sizeof(float));
	cudaMalloc((void **) &d_out_B, sizeof(float));

	sdkCreateTimer(&hTimer);

	// Initialize h_in
	srand(2019);
	for (int i = 0; i < N; i++) {
		h_in[i] = rand() % 512;
	}
	// memset to 0
	memset(h_out_A, 0, sizeof(float));
	memset(h_out_B, 0, sizeof(float));
	memset(h_correct, 0, sizeof(float));
	HANDLE_ERROR(cudaMemset(d_in, 0, N * sizeof(float)));
	HANDLE_ERROR(cudaMemset(d_out_A, 0, sizeof(float)));
	HANDLE_ERROR(cudaMemset(d_out_B, 0, sizeof(float)));
	// calculate correct output
	hAvgSecs = clock();
	sumReductionCPU(h_in, h_correct);
	hAvgSecs = clock() - hAvgSecs;
	printf("Host time: %0.5f sec \n", hAvgSecs/1e+9);

	cudaDeviceSynchronize();
	sdkResetTimer(&hTimer);
	sdkStartTimer(&hTimer);
	// transfer data to GPU
	HANDLE_ERROR(cudaMemcpy(d_in, h_in, N * sizeof(float), cudaMemcpyHostToDevice));
	// execute kernel A
	sumReductionGPU_A<<< N / NUM_THREADS_PER_BLOCK, NUM_THREADS_PER_BLOCK >>>(d_in, d_out_A);
	// transfer data from GPU
	HANDLE_ERROR(cudaMemcpy(h_out_A, d_out_A, sizeof(float), cudaMemcpyDeviceToHost));

	cudaDeviceSynchronize();
	sdkStopTimer(&hTimer);
	dAvgSecs = 1.0e-3 * (double)sdkGetTimerValue(&hTimer);
	printf("Kernel A time (average) : %.5f sec\n", dAvgSecs);

	HANDLE_ERROR(cudaDeviceSynchronize());
	sdkResetTimer(&hTimer);
	sdkStartTimer(&hTimer);
	// execute kernel B
	sumReductionGPU_B<<< N / NUM_THREADS_PER_BLOCK, NUM_THREADS_PER_BLOCK >>>(d_in, d_out_B);
	// transfer data from GPU
	HANDLE_ERROR(cudaMemcpy(h_out_B, d_out_B, sizeof(float), cudaMemcpyDeviceToHost));
	cudaDeviceSynchronize();
	sdkStopTimer(&hTimer);
	dAvgSecs = 1.0e-3 * (double)sdkGetTimerValue(&hTimer);
	printf("Kernel B time (average) : %.5f sec\n", dAvgSecs);
	// cleanup
	free(h_in);
	free(h_out_A);
	free(h_out_B);
	free(h_correct);
	HANDLE_ERROR(cudaFree(d_in));
	HANDLE_ERROR(cudaFree(d_out_A));
	HANDLE_ERROR(cudaFree(d_out_B));
	cudaDeviceReset();
	// validate data
	if (!(abs((*h_correct) - (*h_out_A)) <= 0.0001) ||
				!(abs((*h_correct) - (*h_out_B)) <= 0.0001) ) {
		printf("Test failed (h_correct [%f], h_out_A [%f], h_out_B [%f])!\n",
					*h_correct, *h_out_A, *h_out_B);
		exit(1);
	}
	printf("Test passed!\n");
	exit(EXIT_SUCCESS);
}
