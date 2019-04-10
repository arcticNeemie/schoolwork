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
#define FILTERSIZE 3

//Image files
const char *imageFilename = "lena_bw.pgm";

//Functions
void printImage(float* hData, int width, int height);
void saveImage(float* dData,char* imagePath,int filter,int width, int height);
void convolveCPU(float* dData, float*hData, float* filter, int width, int height);

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv)
{
    printf("Starting execution\n");
    //Load Image
    printf("Loading image: %s\n",imageFilename);
    float *hData = NULL; //Input
    unsigned int width, height;
    char *imagePath = sdkFindFilePath(imageFilename, argv[0]);
    if (imagePath == NULL){
        printf("Unable to source image file: %s\n", imageFilename);
        exit(EXIT_FAILURE);
    }
    sdkLoadPGM(imagePath, &hData, &width, &height);
    unsigned int size = width * height * sizeof(float);
    printf("Loaded '%s', %d x %d pixels\n", imageFilename, width, height);

    //Define Filter
    float averagingFilter[] = {1.0/9,1.0/9,1.0/9,1.0/9,1.0/9,1.0/9,1.0/9,1.0/9,1.0/9}; //Averaging Filter
    float sharpeningFilter[] = {-1,-1,-1,-1,9,-1,-1,-1,-1}; //Sharpening Filter
    float sobelFilter[] = {-1,0,1,-2,0,2,-1,0,1};

    //Apply serial convolution
    float *dDataAverage = (float*) malloc(size); //Output
    convolveCPU(dDataAverage,hData,averagingFilter,width,height);

    float *dDataSharpen = (float*) malloc(size); //Output
    convolveCPU(dDataSharpen,hData,sharpeningFilter,width,height);

    float *dDataSobel = (float*) malloc(size); //Output
    convolveCPU(dDataSobel,hData,sobelFilter,width,height);
    // Write result to file
    saveImage(dDataAverage,imagePath,1,width,height);



}

////////////////////////////////////////////////////////////////////////////////
// Convolutions
////////////////////////////////////////////////////////////////////////////////

//Serial convolution
void convolveCPU(float* dData, float*hData, float* filter, int width, int height){
  float sum;
  int adjust = FILTERSIZE/2; //Integer division should floor
  int x1, y1;
  for(int x=0;x<height;x++){
    for(int y=0;y<width;y++){
      sum = 0;
      for(int s=0;s<FILTERSIZE;s++){
        for(int t=0;t<FILTERSIZE;t++){
          x1 = x-s+adjust;
          y1 = y-t+adjust;
          if(x1>=0 && x1<height && y1>=0 && y1<width){
              sum += hData[x1*width+y1]*filter[s*FILTERSIZE+t];
          }
        }
      }
      dData[x*width+y] = sum;
    }
  }
}

////////////////////////////////////////////////////////////////////////////////
// Utility Functions
////////////////////////////////////////////////////////////////////////////////

//Print out the image as a matrix
void printImage(float* hData, int width, int height){
    for(int i=0;i<height;i++){
        for(int j=0;j<width;j++){
            printf("%f",hData[i*height+j]);
        }
        printf("\n");
    }
}

//Save image to file
void saveImage(float* dData,char* imagePath,int filter,int width, int height){
  char outputFilename[1024];
  strcpy(outputFilename, imagePath);
  strcpy(outputFilename + strlen(imagePath) - 3, "_out1.pgm");
  sdkSavePGM(outputFilename, dData, width, height);
  printf("Wrote '%s'\n", outputFilename);
}
