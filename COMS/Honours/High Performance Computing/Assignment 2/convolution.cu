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
const char *sobelName = "_sobel_";
const char *sharpenName = "_sharpen_";
const char *averageName = "_average_";

//Functions
void printImage(float* hData, int width, int height);
void saveImage(float* dData,char* imagePath,const char* filter,int width, int height, float time);
void convolveCPU(float* dData, float*hData, float* filter, int width, int height);
void applySerialConvolution(float* hData, float* filter, char* imagePath, const char* name, int width, int height, unsigned int size);

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv)
{
    const char* imageFilename;
    if(argc>1){
      imageFilename = argv[1];
    }
    else{
      imageFilename = "lena_bw.pgm";
    }
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
    printf("Loaded '%s', %d x %d pixels\n\n", imageFilename, width, height);

    //Define Filter
    float averagingFilter[] = {1.0/9,1.0/9,1.0/9,1.0/9,1.0/9,1.0/9,1.0/9,1.0/9,1.0/9}; //Averaging Filter
    float sharpeningFilter[] = {-1,-1,-1,-1,9,-1,-1,-1,-1}; //Sharpening Filter
    float sobelFilter[] = {-1,0,1,-2,0,2,-1,0,1}; //Sobel Filter

    //Apply serial convolution
    printf("Beginning serial convolution...\n");
    applySerialConvolution(hData,averagingFilter,imagePath,averageName,width,height,size);
    applySerialConvolution(hData,sharpeningFilter,imagePath,sharpenName,width,height,size);
    applySerialConvolution(hData,sobelFilter,imagePath,sobelName,width,height,size);
    printf("Finished serial convolution!\n\n");

    //Apply naive parallelization implementation
    //TODO

    //Apply shared memory implementation
    //TODO

    //Apply constant memory implementation
    //TODO

    //Apply texture memory implementation
    //TODO

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
void saveImage(float* dData,char* imagePath,const char* filter,int width, int height, float time){
  char outputFilename[1024];
  char* sub = (char*) malloc(strlen(filter)+strlen("out"));
  strcpy(sub,filter);
  strcat(sub,"serial_out");
  int offset = strlen(imagePath)/sizeof(char) - 4;
  strncpy(outputFilename,imagePath,offset);
  outputFilename[offset] = '\0';
  strcat(outputFilename,sub);
  strcat(outputFilename,imagePath+offset);
  sdkSavePGM(outputFilename, dData, width, height);
  printf("Convolved in serial in %f s, saved to '%s'\n", time, outputFilename);
}

//Apply a filter in serial, time it and save result
void applySerialConvolution(float* hData, float* filter, char* imagePath, const char* name, int width, int height, unsigned int size){
  float *dData = (float*) malloc(size); //Output
  StopWatchInterface *timer = NULL;
  sdkCreateTimer(&timer);
  sdkStartTimer(&timer);
  convolveCPU(dData,hData,filter,width,height);
  sdkStopTimer(&timer);
  float time = sdkGetTimerValue(&timer) / 1000.0f;
  sdkDeleteTimer(&timer);
  saveImage(dData,imagePath,name,width,height,time);
}
