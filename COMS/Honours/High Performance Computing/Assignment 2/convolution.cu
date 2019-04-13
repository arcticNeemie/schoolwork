/*
 *
 * This program takes an input grayscale image and applies a sepcified filter
 * using image convolution, applying various parallelization approaches on the GPU
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
#define CONST_FILTERSIZE 5
#define TILE_WIDTH 16

//Constant memory
__device__ __constant__ float constantFilter[CONST_FILTERSIZE*CONST_FILTERSIZE];

// Texture reference for 2D float texture
texture<float, 2, cudaReadModeElementType> tex;

//Functions
void printImage(float* hData, int width, int height);
void saveImage(float* dData,char* imagePath,const char* filter,
    const char* type, int width, int height, float time);
void printDivider();
float* readCustomFilter(const char* filename);
float* padArray(float* array, int pad, int width, int height);
float* createAverageFilter();
float* createSharpenFilter();
float* createSobelFilter();

void convolveCPU(float* dData, float*hData, float* filter,
    int width, int height);

void compare(const char* filterName, float* oldImage, float* newImage, int width, int height, float time);

float* applySerialConvolution(float* hData, float* filter, char* imagePath,
    const char* name, int width, int height, unsigned int size);
void applyNaiveParallelConvolution(float* oldImage,float* hData, float* filter, char* imagePath,
    const char* name, int width, int height, unsigned int size);
void applyConstantMemoryConvolution(float* oldImage,float* hData, float* filter, char* imagePath,
    const char* name, int width, int height, unsigned int size);
void applyTextureMemoryConvolution(float* oldImage,float* hData, float* filter, char* imagePath,
    const char* name, int width, int height, unsigned int size);
void applySharedMemoryConvolution(float* oldImage,float* hData, float* filter, char* imagePath,
    const char* name, int width, int height, unsigned int size);

////////////////////////////////////////////////////////////////////////////////
// Convolutions
////////////////////////////////////////////////////////////////////////////////

//Serial convolution
void convolveCPU(float* dData, float*hData, float* filter, int width, int height){
  float sum;
  int adjust = CONST_FILTERSIZE/2; //Integer division should floor
  int x1, y1;
  for(int x=0;x<height;x++){
    for(int y=0;y<width;y++){
      sum = 0;
      for(int s=0;s<CONST_FILTERSIZE;s++){
        for(int t=0;t<CONST_FILTERSIZE;t++){
          x1 = x-s+adjust;
          y1 = y-t+adjust;
          if(x1>=0 && x1<height && y1>=0 && y1<width){
              sum += hData[x1*width+y1]*filter[s*CONST_FILTERSIZE+t];
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

//Naive Parallel convolution
__global__ void convolveGPUNaive(float* dData,float* hData,float* filter,int width,int height){
  unsigned int x = threadIdx.x + blockDim.x*blockIdx.x;
  unsigned int y = threadIdx.y + blockDim.y*blockIdx.y;

  int adjust = CONST_FILTERSIZE/2;
  int x1,y1;
  if(x<height && y<width){
    float sum = 0;
    for(int s=0;s<CONST_FILTERSIZE;s++){
      for(int t=0;t<CONST_FILTERSIZE;t++){
        x1 = x-s+adjust;
        y1 = y-t+adjust;
        if((x1>=0) && (x1<height) && (y1>=0) && (y1<width)){
          sum += hData[x1*width+y1]*filter[s*CONST_FILTERSIZE+t];
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

//Constant convolution
__global__ void convolveGPUConstant(float* dData, float* image, int width, int height){
  unsigned int x = threadIdx.x + blockDim.x*blockIdx.x;
  unsigned int y = threadIdx.y + blockDim.y*blockIdx.y;

  int adjust = CONST_FILTERSIZE/2;
  int x1,y1;
  if(x<height && y<width){
    float sum = 0;
    for(int s=0;s<CONST_FILTERSIZE;s++){
      for(int t=0;t<CONST_FILTERSIZE;t++){
        x1 = x-s+adjust;
        y1 = y-t+adjust;
        if((x1>=0) && (x1<height) && (y1>=0) && (y1<width)){
          sum += image[x1*width+y1]*constantFilter[s*CONST_FILTERSIZE+t];
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

//Texture convolution
__global__ void convolveGPUTexture(float* dData,float* hData,float* filter,int width,int height){
  unsigned int x = threadIdx.x + blockDim.x*blockIdx.x;
  unsigned int y = threadIdx.y + blockDim.y*blockIdx.y;
  int adjust = CONST_FILTERSIZE/2;
  int x1,y1;
  float coords = 0.5;
  if(x<height && y<width){
    float sum = 0;
    for(int s=0;s<CONST_FILTERSIZE;s++){
      for(int t=0;t<CONST_FILTERSIZE;t++){
        x1 = x-s+adjust;
        y1 = y-t+adjust;
        if((x1>=0) && (x1<height) && (y1>=0) && (y1<width)){
            sum += tex2D(tex,(y1+coords)/(float)width,(x1+coords)/(float)height)*filter[s*CONST_FILTERSIZE+t];
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

//Shared memory convolution
__global__ void convolveGPUShared(float* dData,float* hData,float* filter,int width,int height){
  unsigned int bx = blockIdx.x;
  unsigned int by = blockIdx.y;
  unsigned int tx = threadIdx.x;
  unsigned int ty = threadIdx.y;

  unsigned int row = tx + TILE_WIDTH*bx;
  unsigned int col = ty + TILE_WIDTH*by;

  const int adjust = CONST_FILTERSIZE/2;
  const int BIGTILE = TILE_WIDTH + 2*adjust;
  __shared__ float image[BIGTILE][BIGTILE];
  //Initialize Image
  int imPos = tx*TILE_WIDTH + ty;
  int tileY = imPos % BIGTILE; //Shared memory index
  int tileX = imPos / BIGTILE; //Shared memory index
  int x = bx*TILE_WIDTH + tileX; //Convert to image coordinates
  int y = by*TILE_WIDTH + tileY; //Convert to image coordinates
  if(x>=0 && x<height && y>=0 && y<width){ //Check if out of bounds
    //printf("tileX = %i, tileY = %i\n",tileX,tileY);
    image[tileX][tileY] = hData[x*width+y];
  }
  else{
    image[tileX][tileY] = 0;
  }


  imPos = tx*TILE_WIDTH + ty + TILE_WIDTH*TILE_WIDTH;
  tileY = imPos % BIGTILE; //Shared memory index
  tileX = imPos / BIGTILE; //Shared memory index
  x = bx*TILE_WIDTH + tileX; //Convert to image coordinates
  y = by*TILE_WIDTH + tileY; //Convert to image coordinates
  if(tileY < BIGTILE && tileX < BIGTILE){ //Check if out of bounds during second pass
    if(x>=0 && x<height && y>=0 && y<width){ //Check if out of bounds
      //printf("tileX = %i, tileY = %i\n",tileX,tileY);
      image[tileX][tileY] = hData[x*width+y];
    }
    else{
      image[tileX][tileY] = 0;
    }
  }

  __syncthreads();
  //Calculations
  if(row<height && col<width){
    float sum = 0;
    int x1,y1;
    if(tx<adjust || tx >= BIGTILE-adjust || ty<adjust || ty >= BIGTILE-adjust){ //Border tile
      for(int s=0;s<CONST_FILTERSIZE;s++){
        for(int t=0;t<CONST_FILTERSIZE;t++){
          x1 = row-s+adjust;
          y1 = col-t+adjust;
          if((x1>=0) && (x1<height) && (y1>=0) && (y1<width)){
            sum += hData[x1*width+y1]*filter[s*CONST_FILTERSIZE+t];
          }
        }
      }
    }
    else if(ty<BIGTILE && tx<BIGTILE){ //Inside tile
      for(int s=0;s<CONST_FILTERSIZE;s++){
        for(int t=0;t<CONST_FILTERSIZE;t++){
          x1 = tx-s+adjust;
          y1 = ty-t+adjust;
          if((x1>=0) && (x1<BIGTILE) && (y1>=0) && (y1<BIGTILE)){
            sum += image[x1][y1]*filter[s*CONST_FILTERSIZE+t];
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
    dData[row*width+col] = sum;
  }
}

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv){

    const char* imageFilename;
    const char* filterType;
    int filterNum;
    if(argc>2){
      imageFilename = argv[1];
      filterNum = atoi(argv[2]);
    }
    else{
      imageFilename = "lena.pgm";
      filterNum = 0;
    }
    printDivider();
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
    printf("Filter size: %i x %i\n",CONST_FILTERSIZE,CONST_FILTERSIZE);

    float* filter;
    if(filterNum==0){
      filter = createAverageFilter();
      filterType = "average";
    }
    else if(filterNum==1){
      filter = createSharpenFilter();
      filterType = "sharpen";
    }
    else if(filterNum==2){
      filter = createSobelFilter();
      filterType = "Sobel";
    }
    else{
      printf("Error: filter name not valid\n");
      exit(-1);
    }
    //Serial
    printDivider();
    printf("Beginning serial implementation...\n");
    float* refImage = applySerialConvolution(hData, filter, imagePath,filterType, width, height, size);
    printf("Finished serial implementation...");

    //Naive
    printDivider();
    printf("Beginning naive parallel implementation...\n");
    applyNaiveParallelConvolution(refImage, hData, filter, imagePath,filterType, width, height, size);
    printf("Finished naive parallel implementation...");

    //Shared
    printDivider();
    printf("Beginning shared memory implementation...\n");
    applySharedMemoryConvolution(refImage, hData, filter, imagePath,filterType, width, height, size);
    printf("Finished shared memory implementation...");

    //Constant
    printDivider();
    printf("Beginning constant memory implementation...\n");
    applyConstantMemoryConvolution(refImage, hData, filter, imagePath,filterType, width, height, size);
    printf("Finished constant memory implementation...");

    //Texture
    printDivider();
    printf("Beginning texture memory implementation...\n");
    applyTextureMemoryConvolution(refImage, hData, filter, imagePath,filterType, width, height, size);
    printf("Finished texture memory implementation...");

    printDivider();
    printf("Finished all tests!\n");

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

//Create an averaging filter
float* createAverageFilter(){
  float* filter = (float*) malloc(CONST_FILTERSIZE*CONST_FILTERSIZE*sizeof(float));
  for(int i=0;i<CONST_FILTERSIZE*CONST_FILTERSIZE;i++){
    filter[i] = 1.0/(CONST_FILTERSIZE*CONST_FILTERSIZE);
  }
  return filter;
}

float* createSharpenFilter(){
  float* filter = (float*) malloc(CONST_FILTERSIZE*CONST_FILTERSIZE*sizeof(float));
  for(int i=0;i<CONST_FILTERSIZE*CONST_FILTERSIZE;i++){
    filter[i] = -1;
  }
  int adjust = CONST_FILTERSIZE/2;
  filter[adjust*CONST_FILTERSIZE+adjust] = CONST_FILTERSIZE*CONST_FILTERSIZE;
  return filter;
}

float* createSobelFilter(){
  if(CONST_FILTERSIZE!=3){
    printf("Error: Sobel filter can only be of size 3 x 3\n");
    exit(-1);
  }
  else{
    float* filter = (float*) malloc(CONST_FILTERSIZE*CONST_FILTERSIZE*sizeof(float));
    filter[0] = -1;
    filter[1] = 0;
    filter[2] = 1;
    filter[3] = -2;
    filter[4] = 0;
    filter[5] = 2;
    filter[6] = -1;
    filter[7] = 0;
    filter[8] = 1;
    return filter;
  }
}

//Print out the image as a matrix (for testing purposes)
void printImage(float* hData, int width, int height){
    for(int i=0;i<height;i++){
        for(int j=0;j<width;j++){
            printf("%f ",hData[i*height+j]);
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
float* readCustomFilter(const char* filename){
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

  float* filter = (float*) malloc(CONST_FILTERSIZE*CONST_FILTERSIZE*sizeof(float));
  for(int i=0;i<CONST_FILTERSIZE*CONST_FILTERSIZE;i++){
    fgets(line,1024,f);
    filter[i] = atof(line);
  }

  free(line);
  free(newFilename);

  fclose(f);
  return filter;
}

//Pads an array with 0s
float* padArray(float* array, int pad, int width, int height){
  int newWidth = width + 2*pad;
  int newHeight = height + 2*pad;
  float* newArray = (float*) malloc(newWidth*newHeight*sizeof(float));
  for(int i=0;i<newHeight;i++){
    for(int j=0;j<newWidth;j++){
      if(i==0 || i==newHeight-1){
        newArray[i*newWidth+j] = 0;
      }
      else if(j==0 || j==newWidth-1){
        newArray[i*newWidth+j] = 0;
      }
      else{
        newArray[i*newWidth+j] = array[(i-1)*width+(j-1)];
      }
    }
  }
  return newArray;
}

//Removes padding
float* unPad(float* array, int pad, int width, int height){
  float* newArray = (float*) malloc(width*height*sizeof(float));
  for(int i=0;i<height;i++){
    for(int j=0;j<width;j++){
      newArray[i*width+j] = array[(i+pad)*(width+2*pad)+(j+pad)];
    }
  }
  return newArray;
}

////////////////////////////////////////////////////////////////////////////////
// Application Functions
////////////////////////////////////////////////////////////////////////////////

//Apply a filter in serial, time it and save result
float* applySerialConvolution(float* hData, float* filter, char* imagePath, const char* name, int width, int height, unsigned int size){
  const char* type = "serial";
  float *dData = (float*) malloc(size); //Output
  StopWatchInterface *timer = NULL;
  sdkCreateTimer(&timer);
  sdkStartTimer(&timer);
  convolveCPU(dData,hData,filter,width,height);
  sdkStopTimer(&timer);
  float time = sdkGetTimerValue(&timer) / 1000.0f;
  sdkDeleteTimer(&timer);
  saveImage(dData,imagePath,name,type,width,height,time);
  return dData;
}

//Apply a filter in the naive parallel approach, time it and compare against serial version
void applyNaiveParallelConvolution(float* oldImage,float* hData, float* filter, char* imagePath, const char* name, int width, int height, unsigned int size){
  //int devID = findCudaDevice(argc, (const char **) argv);
  // Allocate device memory for result
  float *dData = NULL;
  // Allocate device memory and copy image data
  checkCudaErrors(cudaMalloc((void **) &dData, size));
  checkCudaErrors(cudaMemcpy(dData,hData,size,cudaMemcpyHostToDevice));

  int fsize = CONST_FILTERSIZE*CONST_FILTERSIZE*sizeof(float);
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
  convolveGPUNaive<<<dimGrid, dimBlock>>>(dOutput,dData,dFilter,width,height);
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
void applyConstantMemoryConvolution(float* oldImage,float* hData, float* filter, char* imagePath, const char* name, int width, int height, unsigned int size){
  //int devID = findCudaDevice(argc, (const char **) argv);
  // Allocate device memory for result
  float *dData = NULL;
  // Allocate device memory and copy image data
  checkCudaErrors(cudaMalloc((void **) &dData, size));
  checkCudaErrors(cudaMemcpy(dData,hData,size,cudaMemcpyHostToDevice));

  int fsize = CONST_FILTERSIZE*CONST_FILTERSIZE*sizeof(float);
  float my_filter[CONST_FILTERSIZE*CONST_FILTERSIZE];
  for(int i=0;i<CONST_FILTERSIZE*CONST_FILTERSIZE;i++){
    my_filter[i] = filter[i];
  }
  checkCudaErrors(cudaMemcpyToSymbol(constantFilter,my_filter,fsize));

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
  convolveGPUConstant<<<dimGrid, dimBlock>>>(dOutput,dData,width,height);
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
  checkCudaErrors(cudaFree(dOutput));
  cudaDeviceReset();
}

//Apply a filter in the texture memory parallel approach, time it and compare against serial version
void applyTextureMemoryConvolution(float* oldImage,float* hData, float* filter, char* imagePath, const char* name, int width, int height, unsigned int size){
  // Allocate device memory for result
  float *dData = NULL;
  checkCudaErrors(cudaMalloc((void **) &dData, size));

  //Allocate device memory for filter
  int fsize = CONST_FILTERSIZE*CONST_FILTERSIZE*sizeof(float);
  float *dFilter = NULL;
  checkCudaErrors(cudaMalloc((void **) &dFilter, fsize));
  checkCudaErrors(cudaMemcpy(dFilter,filter,fsize,cudaMemcpyHostToDevice));

  // Allocate array and copy image data
  cudaChannelFormatDesc channelDesc =
      cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
  cudaArray *cuArray;
  checkCudaErrors(cudaMallocArray(&cuArray,&channelDesc,width,height));
  checkCudaErrors(cudaMemcpyToArray(cuArray,0,0,hData,size,cudaMemcpyHostToDevice));
  // Set texture parameters
  tex.addressMode[0] = cudaAddressModeWrap;
  tex.addressMode[1] = cudaAddressModeWrap;
  tex.filterMode = cudaFilterModeLinear;
  tex.normalized = true;    // access with normalized texture coordinates
  // Bind the array to the texture
  checkCudaErrors(cudaBindTextureToArray(tex, cuArray, channelDesc));
  //Grid and block stuff
  dim3 dimBlock(8, 8, 1);
  dim3 dimGrid(height / dimBlock.x, width / dimBlock.y, 1);

  checkCudaErrors(cudaDeviceSynchronize());
  StopWatchInterface *timer = NULL;
  sdkCreateTimer(&timer);
  sdkStartTimer(&timer);

  // Execute the kernel
  convolveGPUTexture<<<dimGrid, dimBlock>>>(dData,hData,dFilter,width,height);

  // Check if kernel execution generated an error
  getLastCudaError("Kernel execution failed");

  checkCudaErrors(cudaDeviceSynchronize());
  sdkStopTimer(&timer);
  float time = sdkGetTimerValue(&timer)/1000.0f;

  // Allocate mem for the result on host side
  float *hOutputData = (float *) malloc(size);
  // copy result from device to host
  checkCudaErrors(cudaMemcpy(hOutputData,dData,size,cudaMemcpyDeviceToHost));

  compare(name,oldImage, hOutputData, width, height, time);
  //saveImage(hOutputData,imagePath,name,"texture",width,height,time);
  sdkDeleteTimer(&timer);

  free(hOutputData);

  //Free
  checkCudaErrors(cudaFree(dData));
  checkCudaErrors(cudaFreeArray(cuArray));

  cudaDeviceReset();
}

//Apply a filter in the shared memory parallel approach, time it and compare against serial version
void applySharedMemoryConvolution(float* oldImage,float* hData, float* filter, char* imagePath, const char* name, int width, int height, unsigned int size){
  //int adjust = CONST_FILTERSIZE/2;
  //int padSize = (width+2*adjust)*(height+2*adjust)*sizeof(float);
  //float* padHData = padArray(hData,adjust,width,height);
  //Image
  float *dData = NULL;
  checkCudaErrors(cudaMalloc((void **) &dData, size));
  checkCudaErrors(cudaMemcpy(dData,hData,size,cudaMemcpyHostToDevice));

  //Filter
  int fsize = CONST_FILTERSIZE*CONST_FILTERSIZE*sizeof(float);
  float *dFilter = NULL;
  checkCudaErrors(cudaMalloc((void **) &dFilter, fsize));
  checkCudaErrors(cudaMemcpy(dFilter,filter,fsize,cudaMemcpyHostToDevice));

  //Output
  float *dOutput = NULL;
  checkCudaErrors(cudaMalloc((void **) &dOutput, size));

  //Block
  dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);
  dim3 dimGrid(ceil(height / TILE_WIDTH), ceil(width / TILE_WIDTH), 1);
  //printf("dimBlock: %i,%i,%i\n",dimBlock.x,dimBlock.y,dimBlock.z);
  //printf("dimGrid: %i,%i,%i\n",dimGrid.x,dimGrid.y,dimGrid.z);
  checkCudaErrors(cudaDeviceSynchronize());

  //Time
  StopWatchInterface *timer = NULL;
  sdkCreateTimer(&timer);
  sdkStartTimer(&timer);
  //Execute kernel
  convolveGPUShared<<<dimGrid, dimBlock>>>(dOutput,dData,dFilter,width,height);
  // Check if kernel execution generated an error
  getLastCudaError("Kernel execution failed");

  checkCudaErrors(cudaDeviceSynchronize());
  sdkStopTimer(&timer);
  float time = sdkGetTimerValue(&timer)/1000.0f;
  //const char* type = "naive";

  // Allocate mem for the result on host side
  float* hOutput = (float*) malloc(size);
  checkCudaErrors(cudaMemcpy(hOutput,dOutput,size,cudaMemcpyDeviceToHost));

  //float* finalImage = unPad(hOutput,adjust,width,height);

  compare(name,oldImage, hOutput, width, height, time);
  //saveImage(hOutput,imagePath,name,"shared",width,height,time);
  sdkDeleteTimer(&timer);

  free(hOutput);
  checkCudaErrors(cudaFree(dData));
  checkCudaErrors(cudaFree(dFilter));
  checkCudaErrors(cudaFree(dOutput));
  cudaDeviceReset();

}
