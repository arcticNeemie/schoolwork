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
const char *sobelName = "sobel";
const char *sharpenName = "sharpen";
const char *averageName = "average";

//Functions
void printImage(float* hData, int width, int height);
void saveImage(float* dData,char* imagePath,const char* filter,
    const char* type, int width, int height, float time);
void printDivider();
float* readCustomFilter(int filtersize, const char* filename);

void convolveCPU(float* dData, float*hData, float* filter,
    int width, int height, int filtersize);

void compare(const char* filterName, float* oldImage, float* newImage, int width, int height, float time);

float* applySerialConvolution(float* hData, float* filter, char* imagePath,
    const char* name, int width, int height, unsigned int size, int filtersize);
void applyNaiveParallelConvolution(float* oldImage,float* hData, float* filter, char* imagePath,
    const char* name, int width, int height, unsigned int size, int filtersize);
void applyConstantMemoryConvolution(float* oldImage,float* hData, float* filter, char* imagePath,
    const char* name, int width, int height, unsigned int size, int filtersize);

////////////////////////////////////////////////////////////////////////////////
// Convolutions
////////////////////////////////////////////////////////////////////////////////

//Serial convolution
void convolveCPU(float* dData, float*hData, float* filter, int width, int height, int filtersize){
  float sum;
  int adjust = filtersize/2; //Integer division should floor
  int x1, y1;
  for(int x=0;x<height;x++){
    for(int y=0;y<width;y++){
      sum = 0;
      for(int s=0;s<filtersize;s++){
        for(int t=0;t<filtersize;t++){
          x1 = x-s+adjust;
          y1 = y-t+adjust;
          if(x1>=0 && x1<height && y1>=0 && y1<width){
              sum += hData[x1*width+y1]*filter[s*filtersize+t];
          }
        }
      }
      if(sum>1){
          sum = 1;
      }
      else if(sum<0){
          sum = 0;
      }
      dData[x*width+y] = sum;
    }
  }
}

__global__ void convolveGPUNaive(float* dData,float* hData,float* filter,int width,int height, int filtersize){
  unsigned int x = threadIdx.x + blockDim.x*blockIdx.x;
  unsigned int y = threadIdx.y + blockDim.y*blockIdx.y;

  int adjust = filtersize/2;
  int x1,y1;
  if(x<height && y<width){
    float sum = 0;
    for(int s=0;s<filtersize;s++){
      for(int t=0;t<filtersize;t++){
        x1 = x-s+adjust;
        y1 = y-t+adjust;
        if((x1>=0) && (x1<height) && (y1>=0) && (y1<width)){
          if(x1*width+y1<width*height && s*filtersize+t<filtersize*filtersize){
            sum += hData[x1*width+y1]*filter[s*filtersize+t];
          }
          else{
            printf("Hello\n");
          }

        }
      }
    }
    if(sum>1){
        sum = 1;
    }
    else if(sum<0){
        sum = 0;
    }
    dData[x*width+y] = sum;
  }
}

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv)
{
    const char* imageFilename;
    int customFilter = 0;
    int filtersize;
    if(argc>1){
      imageFilename = argv[1];
      if(argc>3){
        customFilter = 1;
      }
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

    if(customFilter == 0){
      //Define Filter
      filtersize = 3;
      float averagingFilter[] = {1.0/9,1.0/9,1.0/9,1.0/9,1.0/9,1.0/9,1.0/9,1.0/9,1.0/9}; //Averaging Filter
      float sharpeningFilter[] = {-1,-1,-1,-1,9,-1,-1,-1,-1}; //Sharpening Filter
      float sobelFilter[] = {-1,0,1,-2,0,2,-1,0,1}; //Sobel Filter

      //Apply serial convolution
      printf("Beginning serial convolution...\n\n");
      float* refAverage = applySerialConvolution(hData,averagingFilter,imagePath,averageName,width,height,size,filtersize);
      float* refSharpen = applySerialConvolution(hData,sharpeningFilter,imagePath,sharpenName,width,height,size,filtersize);
      float* refSobel = applySerialConvolution(hData,sobelFilter,imagePath,sobelName,width,height,size,filtersize);
      printf("\nFinished serial convolution!");
      printDivider();

      //Apply naive parallelization implementation
      printf("Beginning naive parallel convolution...\n\n");
      applyNaiveParallelConvolution(refAverage,hData,averagingFilter,imagePath,"averaging",width,height,size,filtersize);
      applyNaiveParallelConvolution(refSharpen,hData,sharpeningFilter,imagePath,"sharpening",width,height,size,filtersize);
      applyNaiveParallelConvolution(refSobel,hData,sobelFilter,imagePath,"Sobel",width,height,size,filtersize);
      printf("Finished naive parallel convolution!");
      printDivider();

      //Apply shared memory implementation
      //TODO

      //Apply constant memory implementation
      //TODO
      printf("Beginning constant memory parallel convolution...\n\n");
      applyConstantMemoryConvolution(refAverage,hData,averagingFilter,imagePath,"averaging",width,height,size,filtersize);
      printf("Finished constant memory parallel convolution!");

      //Apply texture memory implementation
      //TODO
    }
    else{
      filtersize = atoi(argv[2]);
      const char* filterName = argv[3];
      //Load custom filter from file
      float* filter = readCustomFilter(filtersize,filterName);
      //Apply serial convolution
      printf("Beginning serial convolution...\n\n");
      float* refCustom = applySerialConvolution(hData,filter,imagePath,filterName,width,height,size,filtersize);
      printf("\nFinished serial convolution!");
      printDivider();

      //Apply naive parallelization implementation
      printf("Beginning naive parallel convolution...\n\n");
      applyNaiveParallelConvolution(refCustom,hData,filter,imagePath,filterName,width,height,size,filtersize);
      printf("Finished naive parallel convolution!");
      printDivider();

      //Apply constant memory implementation
      printf("Beginning constant memory parallel convolution...\n\n");
      applyConstantMemoryConvolution(refCustom,hData,averagingFilter,imagePath,filterName,width,height,size,filtersize);
      printf("Finished constant memory parallel convolution!");
      printDivider();

    }

    //Free
    free(imagePath);

}

////////////////////////////////////////////////////////////////////////////////
// Testing
////////////////////////////////////////////////////////////////////////////////

//Compare two images and print an accuracy
void compare(const char* filterName, float* oldImage, float* newImage, int width, int height, float time){
  //Compare data
  float accurate = 0;
  for(int i=0;i<height*width;i++){
    if(abs(newImage[i]-oldImage[i])<MAX_EPSILON_ERROR){
      accurate++;
    }
  }
  //Print Accuracy
  printf("Convolved image using %s filter in %f seconds!\n",filterName,time);
  printf("Compared image to serial implementation: accuracy = %f percent\n\n",(100.0*accurate)/(width*height));
}

////////////////////////////////////////////////////////////////////////////////
// Utility Functions
////////////////////////////////////////////////////////////////////////////////

//Print out the image as a matrix (for testing purposes)
void printImage(float* hData, int width, int height){
    for(int i=0;i<height;i++){
        for(int j=0;j<width;j++){
            printf("%f",hData[i*height+j]);
        }
        printf("\n");
    }
}

//Save image to file
void saveImage(float* dData,char* imagePath,const char* filter, const char* type, int width, int height, float time){
  char outputFilename[1024];
  char* sub = (char*) malloc(strlen(filter)+strlen("out")+2*sizeof(char));
  strcpy(sub,"_");
  strcat(sub,filter);
  strcat(sub,"_");
  strcat(sub,type);
  strcat(sub,"_out");
  int offset = strlen(imagePath)/sizeof(char) - 4;
  strncpy(outputFilename,imagePath,offset);
  outputFilename[offset] = '\0';
  strcat(outputFilename,sub);
  strcat(outputFilename,imagePath+offset);
  sdkSavePGM(outputFilename, dData, width, height);
  printf("Convolved in serial in %f s, saved to '%s'\n", time, outputFilename);
  free(sub);
}

//Used to divide output in the terminal
void printDivider(){
  printf("\n\n=========================\n\n");
}

//Read in from file
float* readCustomFilter(int filtersize, const char* filename){
  FILE* f;
  char* newFilename = (char*)malloc(strlen(filename));
  strncpy(newFilename,filename,strlen(filename));
  newFilename[strlen(filename)/sizeof(char)] = '\0';
  strcat(newFilename,".txt");
  f = fopen(newFilename,"r");
  if(f==NULL){
    printf("Error reading custom filter %s\n",newFilename);
    exit(-1);
  }

  char* line = (char*)malloc(1024*sizeof(char));

  float* filter = (float*) malloc(filtersize*filtersize*sizeof(float));
  for(int i=0;i<filtersize*filtersize;i++){
    fgets(line,1024,f);
    filter[i] = atof(line);
  }

  free(line);
  free(newFilename);

  fclose(f);
  return filter;
}

////////////////////////////////////////////////////////////////////////////////
// Application Functions
////////////////////////////////////////////////////////////////////////////////

//Apply a filter in serial, time it and save result
float* applySerialConvolution(float* hData, float* filter, char* imagePath, const char* name, int width, int height, unsigned int size, int filtersize){
  const char* type = "serial";
  float *dData = (float*) malloc(size); //Output
  StopWatchInterface *timer = NULL;
  sdkCreateTimer(&timer);
  sdkStartTimer(&timer);
  convolveCPU(dData,hData,filter,width,height,filtersize);
  sdkStopTimer(&timer);
  float time = sdkGetTimerValue(&timer) / 1000.0f;
  sdkDeleteTimer(&timer);
  saveImage(dData,imagePath,name,type,width,height,time);
  return dData;
}

//Apply a filter in the naive parallel approach, time it and compare against serial version
void applyNaiveParallelConvolution(float* oldImage,float* hData, float* filter, char* imagePath, const char* name, int width, int height, unsigned int size, int filtersize){
  //int devID = findCudaDevice(argc, (const char **) argv);
  // Allocate device memory for result
  float *dData = NULL;
  // Allocate device memory and copy image data
  checkCudaErrors(cudaMalloc((void **) &dData, size));
  checkCudaErrors(cudaMemcpy(dData,hData,size,cudaMemcpyHostToDevice));

  int fsize = filtersize*filtersize*sizeof(float);
  float *dFilter = NULL;
  checkCudaErrors(cudaMalloc((void **) &dFilter, fsize));
  checkCudaErrors(cudaMemcpy(dFilter,filter,fsize,cudaMemcpyHostToDevice));

  float *dOutput = NULL;
  checkCudaErrors(cudaMalloc((void **) &dOutput, size));

  dim3 dimBlock(8, 8, 1);
  dim3 dimGrid(height / dimBlock.x, width / dimBlock.y, 1);
  checkCudaErrors(cudaDeviceSynchronize());
  //Time
  StopWatchInterface *timer = NULL;
  sdkCreateTimer(&timer);
  sdkStartTimer(&timer);
  //Execute kernel
  convolveGPUNaive<<<dimGrid, dimBlock>>>(dOutput,dData,dFilter,width,height,filtersize);
  // Check if kernel execution generated an error
  getLastCudaError("Kernel execution failed");

  checkCudaErrors(cudaDeviceSynchronize());
  sdkStopTimer(&timer);
  float time = sdkGetTimerValue(&timer)/1000.0f;
  //const char* type = "naive";

  // Allocate mem for the result on host side
  float* hOutput = (float*) malloc(size);
  checkCudaErrors(cudaMemcpy(hOutput,dOutput,size,cudaMemcpyDeviceToHost));

  compare(name,oldImage, hOutput, width, height, time);
  //saveImage(hOutput,imagePath,name,type,width,height,time);
  sdkDeleteTimer(&timer);

  free(hOutput);
  checkCudaErrors(cudaFree(dData));
  checkCudaErrors(cudaFree(dFilter));
  checkCudaErrors(cudaFree(dOutput));
  cudaDeviceReset();
}

//Apply a filter in the constant memory parallel approach, time it and compare against serial version
void applyConstantMemoryConvolution(float* oldImage,float* hData, float* filter, char* imagePath, const char* name, int width, int height, unsigned int size, int filtersize){
  //int devID = findCudaDevice(argc, (const char **) argv);
  // Allocate device memory for result
  float* dData;
  checkCudaErrors(cudaMalloc((void**) &dData, size));

  __constant__ float* dData;
  checkCudaErrors(cudaMemcpyToSymbol(dData,hData,size));

  int fsize = filtersize*filtersize*sizeof(float);
  float *dFilter = NULL;
  checkCudaErrors(cudaMalloc((void **) &dFilter, fsize));
  checkCudaErrors(cudaMemcpy(dFilter,filter,fsize,cudaMemcpyHostToDevice));

  float *dOutput = NULL;
  checkCudaErrors(cudaMalloc((void **) &dOutput, size));

  dim3 dimBlock(8, 8, 1);
  dim3 dimGrid(height / dimBlock.x, width / dimBlock.y, 1);
  checkCudaErrors(cudaDeviceSynchronize());
  //Time
  StopWatchInterface *timer = NULL;
  sdkCreateTimer(&timer);
  sdkStartTimer(&timer);
  //Execute kernel
  convolveGPUNaive<<<dimGrid, dimBlock>>>(dOutput,dData,dFilter,width,height,filtersize);
  // Check if kernel execution generated an error
  getLastCudaError("Kernel execution failed");

  checkCudaErrors(cudaDeviceSynchronize());
  sdkStopTimer(&timer);
  float time = sdkGetTimerValue(&timer)/1000.0f;
  //const char* type = "naive";

  // Allocate mem for the result on host side
  float* hOutput = (float*) malloc(size);
  checkCudaErrors(cudaMemcpy(hOutput,dOutput,size,cudaMemcpyDeviceToHost));

  compare(name,oldImage, hOutput, width, height, time);
  //saveImage(hOutput,imagePath,name,type,width,height,time);
  sdkDeleteTimer(&timer);

  free(hOutput);
  checkCudaErrors(cudaFree(dData));
  checkCudaErrors(cudaFree(dFilter));
  checkCudaErrors(cudaFree(dOutput));
  cudaDeviceReset();
}
