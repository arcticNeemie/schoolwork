// includes, system
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <cuda_runtime.h>


#define N (1 << 22)

// Simple utility function to check for CUDA runtime errors
void checkCUDAError(const char *msg) {
    cudaError_t err = cudaGetLastError();
    if( cudaSuccess != err) {
        fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString( err) );
        exit(EXIT_FAILURE);
    }
}

void multiplyCPU(const float *h_v1, const float *h_v2, float *h_out) {
    for (int i = 0; i < N; i++) {
        h_out[i] = h_v1[i] * h_v2[i];
    }
}

void expensiveFunctionCPU(const float *h_v1, const float *h_v2, float *h_out) {
    for (int i = 0; i < N; i++) {
        float a = h_v1[i], b = h_v2[i];
        h_out[i] = (a * b) * (sqrt(a + b) + sqrt(a) + sqrt(b - a) + sqrt(b));
    }
}

__global__ void multiplyGPU(const float *g_v1, const float *g_v2, float *g_out) {
    // YOUR CODE GOES HERE
    // Implement GPU version of multiplyCPU
    // (a)
    unsigned int i = threadIdx.x + blockDim.x*blockIdx.x;

    if(i<N){
      g_out[i] = g_v1[i]*g_v2[i];
    }

}

__global__ void expensiveFunctionGPU(const float *g_v1, const float *g_v2, float *g_out) {
    // YOUR CODE GOES HERE
    // Implement GPU version of expensiveFunctionCPU
    // (c)
    unsigned int i = threadIdx.x + blockDim.x*blockIdx.x;

    if(i<N){
      float a = g_v1[i], b = g_v2[i];
      g_out[i] = (a * b) * (sqrt(a + b) + sqrt(a) + sqrt(b - a) + sqrt(b));
    }

}

///////////////////////////////////////////////////////////////////////////////
// Program main
///////////////////////////////////////////////////////////////////////////////
int main( int argc, char** argv) {
    ///////////////////// LOTS OF INITIALIZATION CODE ////////////////////////
    // pointers to host vectors v1 and v1
    float *h_v1, *h_v2;
    // pointers to host output for multiply and expensive
    float *h_multiply_correct, *h_expensive_correct;
    // pointers to store results from gpu functions on the host
    float *h_multiply_out, *h_expensive_out;

    // pointer for device vectors v1 and v2
    float *d_v1, *d_v2;
    // pointers to device output
    float *d_multiply_out, *d_expensive_out;



    // Number of test iterations to use for timing
    int testIterations = 3;

    // allocate memory for pointers
    h_v1 = (float *)malloc(N * sizeof(float));
    h_v2 = (float *)malloc(N * sizeof(float));
    h_multiply_correct  = (float *)malloc(N * sizeof(float));
    h_expensive_correct = (float *)malloc(N * sizeof(float));
    h_multiply_out  = (float *)malloc(N * sizeof(float));
    h_expensive_out = (float *)malloc(N * sizeof(float));

    // allocate memory for device pointers
    cudaMalloc( (void **) &d_v1, N * sizeof(float));
    cudaMalloc( (void **) &d_v2, N * sizeof(float));
    cudaMalloc( (void **) &d_multiply_out, N *sizeof(float));
    cudaMalloc( (void **) &d_expensive_out, N *sizeof(float));

    // Initialize v1 and v2
    for (int i = 0; i < N; i++) {
        h_v1[i] = 1.0 * (i % 10000);
        h_v2[i] = 2 * h_v1[i];
    }

    // CPU functions over testIterations
    for (int i = 0; i < testIterations; i++) {
        // zero memory for outputs
        memset(h_multiply_correct,  0, N * sizeof(float));
        memset(h_expensive_correct,  0, N * sizeof(float));

				// run cpu kernel
        multiplyCPU(h_v1, h_v2, h_multiply_correct);

        // run cpu kernel
        expensiveFunctionCPU(h_v1, h_v2, h_expensive_correct);
    }

    //////////////////////// INSERT CODE IN THIS SECTION /////////////////////
    // GPU functions over testIterations
    for (int i = 0; i < testIterations; i++) {
        // zero output memory
        memset(h_multiply_out, 0, N * sizeof(float));
        memset(h_expensive_out, 0, N * sizeof(float));
        cudaMemset(d_multiply_out, 0, N * sizeof(float));
        cudaMemset(d_expensive_out, 0, N * sizeof(float));
        // zero input memory
        cudaMemset(d_v1, 0, N * sizeof(float));
        cudaMemset(d_v2, 0, N * sizeof(float));

        // transfer data to GPU
        cudaMemcpy(d_v1, h_v1, N * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_v2, h_v2, N * sizeof(float), cudaMemcpyHostToDevice);

        // YOUR CODE GOES HERE
        // Execute multiply kernel
        // (b)
        const size_t block_size = 256;
        size_t grid_size = N / block_size;

        // time the kernel launches using CUDA events
        cudaEvent_t launch_begin, launch_end;
        cudaEventCreate(&launch_begin);
        cudaEventCreate(&launch_end);
        // record a CUDA event immediately before and after the kernel launch
        cudaEventRecord(launch_begin,0);
        // launch the kernel
        multiplyGPU<<<grid_size,block_size>>>(d_v1,d_v2,d_multiply_out);
        cudaEventRecord(launch_end,0);
        cudaEventSynchronize(launch_end);
      	// measure the time (ms) spent in the kernel
        float time = 0;
      	cudaEventElapsedTime(&time, launch_begin, launch_end);

        printf("Completed multiplication in %f ms\n",time);

        // transfer data from GPU
        cudaMemcpy(h_multiply_out, d_multiply_out, N * sizeof(float), cudaMemcpyDeviceToHost);


        // Check for any CUDA errors
        checkCUDAError("multiplyGPU");

        // zero input memory
        cudaMemset(d_v1, 0, N * sizeof(float));
        cudaMemset(d_v2, 0, N * sizeof(float));

        // transfer data to GPU
        cudaMemcpy(d_v1, h_v1, N * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_v2, h_v2, N * sizeof(float), cudaMemcpyHostToDevice);

        // YOUR CODE GOES HERE
        // Execute expensive function kernel
        // (d)
        expensiveFunctionGPU<<<grid_size,block_size>>>(d_v1,d_v2,d_expensive_out);


        // transfer data from GPU
        cudaMemcpy(h_expensive_out, d_expensive_out, N * sizeof(float), cudaMemcpyDeviceToHost);

        // Check for any CUDA errors
        checkCUDAError("expensiveFunctionGPU");
    }


    /////////////////////////////// VALIDATION ///////////////////////////////

    // check if output from gpu kernels is correct
    for (int i = 0; i < N; i++) {
        if (!(abs(h_multiply_out[i] - h_multiply_correct[i]) <= 0.0001)) {
            printf("Test failed (h_multiply_out[%d]:%f != h_multiply_correct[%d]:%f)!\n",
                   i, h_multiply_out[i], i, h_multiply_correct[i]);
            exit(1);
        }
        if (!(abs(h_expensive_out[i] - h_expensive_correct[i]) <= 0.00001 * abs(h_expensive_correct[i]))) {
            printf("Test failed (h_expensive_out[%d]:%f != h_expensive_correct[%d]:%f)!\n",
                   i, h_expensive_out[i], i, h_expensive_correct[i]);
            exit(1);
        }
    }

    printf("Test passed!\n");

    //////////////////////////////// CLEANUP /////////////////////////////////
    // free host memory
    free(h_v1);
    free(h_v2);
    free(h_multiply_correct);
    free(h_expensive_correct);
    free(h_multiply_out);
    free(h_expensive_out);

    // free device memory
    cudaFree(d_v1);
    cudaFree(d_v2);
    cudaFree(d_multiply_out);
    cudaFree(d_expensive_out);
    return 0;
}
