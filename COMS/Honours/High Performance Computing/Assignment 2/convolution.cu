/*
 *
 * This program takes an input grayscale image and applies a sepcified filter
 * using image convolution
 */

// Includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#ifdef _WIN32
#  define WINDOWS_LEAN_AND_MEAN
#  define NOMINMAX
#  include <windows.h>
#endif

// Includes CUDA
#include <cuda_runtime.h>

// Utilities and timing functions
#include "helper_functions.h"    // includes cuda.h and cuda_runtime_api.h

// CUDA helper functions
#include "helper_cuda.h"         // helper functions for CUDA error check

#define MAX_EPSILON_ERROR 5e-3f

//Image files
const char *imageFilename = "lena_bw.pgm";

//Functions
void printImage(float* hData, int width, int height);

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv)
{
    printf("Starting execution\n");
    //Load Image
    printf("Loading image: %s\n",imageFilename);
    float *hData = NULL;
    unsigned int width, height;
    char *imagePath = sdkFindFilePath(imageFilename, argv[0]);
    if (imagePath == NULL){
        printf("Unable to source image file: %s\n", imageFilename);
        exit(EXIT_FAILURE);
    }
    sdkLoadPGM(imagePath, &hData, &width, &height);
    unsigned int size = width * height * sizeof(float);
    printf("Loaded '%s', %d x %d pixels\n", imageFilename, width, height);
    
}

////////////////////////////////////////////////////////////////////////////////
// Utility Functions
////////////////////////////////////////////////////////////////////////////////
void printImage(float* hData, int width, int height){
    for(int i=0;i<height;i++){
        for(int j=0;j<width;j++){
            printf("%f",hData[i*height+j]);
        }
        printf("\n");
    }
}
