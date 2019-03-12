/*

Parallel kNN using sections directive OpenMP

*/

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>
#include <math.h>
#include <string.h>

//Preprocess
double** initDoubleStarStar(int r,int c);
double** readInArray(char filename[], int type);
//Main Algorithm
int** parallel_kNN(double** P, double** Q, int k);
//Distance metrics
double euclid(double* x, double* y);
double manhattan(double* x, double* y);
//Sorts
void myQsort(int* indices, double* array, int low, int high);
int partition(int* indices, double* array,int low, int high);

void bubble(int* indices, double* array, int size);

void myMergesort(int* indices, double* array, int size);
void myMsort(int* indices, int* indices2, double* array, double* b, int low, int high);
void merge(int* indices, int* indices2, double* array, double* b, int low, int mid, int high);
//Misc
void swapD(double* array,int i, int j);
void swapI(int* array,int i, int j);
void usage(char prog_name[]);

int maxline = 20;
int m,n,d;
int myDist,mySort;
int CUT_OFF = 1000;
int NUM_THREADS = 4;

int main(int argc,char **argv){
  //Check if enough args
  if (argc != 9) {
		usage(argv[0]);
		exit (-1);
	}

  char* pfile = argv[1];
  char* qfile = argv[2];
  int k = atoi(argv[3]);
  m = atoi(argv[4]);
  n = atoi(argv[5]);
  d = atoi(argv[6]);
  myDist = atoi(argv[7]);   //0 = euclid, 1 = manhattan
  mySort = atoi(argv[8]);   //0 = quicksort, 1 = mergesort, 2 = bubblesort

  double** P;
  double** Q;

  //Get arrays
  P = readInArray(pfile,0);
  Q = readInArray(qfile,1);

  omp_set_num_threads(NUM_THREADS);
  int** kNN = parallel_kNN(P,Q,k);

  //Test
  /*
  printf("\n");
  for(int j=0;j<k;j++){
    printf("%i ",kNN[0][j]);
  }
  printf("\n");
  */


  //Cleanup
  free(kNN);
  free(P);
  free(Q);
  exit(0);

}

//Takes in P, Q and k and returns the indices of P which are the k-nearest to each qi
int** parallel_kNN(double** P, double** Q, int k){
  double start_time1, run_time1, start_time, run_time;
  //Calculate Distances
  double** dist = initDoubleStarStar(n,m);
  int i,j;
  start_time = omp_get_wtime();
  #pragma omp parallel for private(i,j) collapse(2)
  for(int i=0;i<n;i++){
    for(int j=0;j<m;j++){
      if(myDist==0){
        dist[i][j] = euclid(Q[i],P[j]);
      }
      else{
        dist[i][j] = manhattan(Q[i],P[j]);
      }
    }
  }
  run_time = omp_get_wtime() - start_time;


  //Initialize index array for sorting
  int** indices;
  indices = (int**) malloc(n * sizeof(int*));
  for(int i=0;i<n;i++){
    indices[i] = (int*) malloc(m * sizeof(int));
    for(int j=0;j<m;j++){
      indices[i][j] = j;
    }
  }



  //Sort
  start_time1 = omp_get_wtime();
  for(int i=0;i<n;i++){
    if(mySort==0){
      #pragma omp parallel
      {
        #pragma omp single
        {
          myQsort(indices[i],dist[i],0,m);
        }
      }
    }
    else if(mySort==1){
      myMergesort(indices[i],dist[i],m);
    }
    else{
      //printf("Here %i\n",i);
      bubble(indices[i],dist[i],m);
    }
  }
  run_time1 = omp_get_wtime() - start_time1;



  //Test
  /*
  for(int i=1;i<=k;i++){
    printf("%f ",dist[0][i]);
  }
  printf("\n");
  */


  free(dist);

  printf("Runtime for m = %i, n = %i, d = %i, k = %i\n",m,n,d,k);
  if(myDist==0){
    printf("Euclidean distance - ");
  }
  else{
    printf("Manhattan distance - ");
  }
  if(mySort==0){
    printf("Quicksort\n");
  }
  else if(mySort==1){
    printf("Mergesort\n");
  }
  else{
    printf("Bubblesort\n");
  }
  printf("Number of threads: %i\n",NUM_THREADS);
  printf("Distance: %f seconds\n",run_time);
  printf("Sorting: %f seconds\n",run_time1);
  printf("Total: %f seconds\n\n\n",run_time+run_time1);

  //Pick k nearest indices:
  int** kIndices = (int**) malloc(n * sizeof(int*));
  for(int i=0;i<n;i++){
    kIndices[i] = (int*) malloc(k * sizeof(int));
    for(int j=0;j<k;j++){
      kIndices[i][j] = indices[i][j];
    }
  }

  free(indices);
  return kIndices;
}

/**
*
*
*   Sorts
*
*/

//Quicksort
void myQsort(int* indices, double* array, int low, int high){
  if(low<high){
    double pivot = partition(indices, array, low, high);
    if(high-low<CUT_OFF){
      myQsort(indices,array, low, pivot - 1);
      myQsort(indices,array, pivot+1, high);
    }
    else{
      #pragma omp task
        myQsort(indices,array, low, pivot - 1);
      #pragma omp task
        myQsort(indices,array, pivot+1, high);
    }
  }
}

//Partition for quicksort
int partition(int* indices, double* array,int low, int high){
  double pivot = array[high];
  int i = low - 1;

  for (int j = low; j <= high - 1; j++){
        if (array[j] <= pivot){
            i++;
            swapD(array,i,j);
            swapI(indices,i,j);
        }
  }
  swapD(array,i+1,high);
  swapI(indices,i+1,high);
  return i+1;
}

//Bubblesort
void bubble(int* indices, double* array, int size){
  //Odd-Even sort
  for(int i=0;i<size;i++){
    int j0 = i % 2;
    #pragma omp parallel for shared(array,indices)
    for(int j=j0;j<size-1;j+=2){
      if(array[j]>array[j+1]){
        swapD(array,j,j+1);
        swapI(indices,j,j+1);
      }
    }
  }
}

//Mergesort parent
void myMergesort(int* indices, double* array, int size){
  double* b = (double*)malloc(size*sizeof(double));
  int* indices2 = (int*)malloc(size*sizeof(int));
  myMsort(indices,indices2,array,b,0,size);

  free(b);
  free(indices2);

}

//Merge sort
void myMsort(int* indices, int* indices2, double* array, double* b, int low, int high){
  int mid;

   if(low < high) {
      mid = (low + high) / 2;
      if(high-low<CUT_OFF){
        myMsort(indices, indices2, array, b, low, mid);
        myMsort(indices, indices2, array, b, mid+1, high);
      }
      else{
        #pragma omp parallel
        {
          #pragma omp single
          {
            #pragma omp task
              myMsort(indices, indices2, array, b, low, mid);
            #pragma omp task
              myMsort(indices, indices2, array, b, mid+1, high);
          }
        }
      }
      merge(indices, indices2, array, b, low, mid, high);
   } else {
      return;
   }

}

//Merge for mergesort
void merge(int* indices, int* indices2, double* array, double* b, int low, int mid, int high){
  int l1 = low;
  int l2 = mid + 1;
  int i;

   for(i = low; l1 <= mid && l2 <= high; i++) {
      if(array[l1] <= array[l2]){
         b[i] = array[l1];
         indices2[i] = indices[l1];
         l1++;
      }
      else{
         b[i] = array[l2];
         indices2[i] = indices[l2];
         l2++;
      }
   }

   while(l1 <= mid){
      b[i] = array[l1];
      indices2[i] = indices[l1];
      i++;
      l1++;
    }

   while(l2 <= high){
      b[i] = array[l2];
      indices2[i] = indices[l2];
      i++;
      l2++;
    }

    for(i = low; i <= high; i++){
      array[i] = b[i];
      indices[i] = indices2[i];
    }
}


/**
*
*
*   Distance Metrics
*
*/

//Computes the euclidean distance
double euclid(double* x, double* y){
  double sum = 0.0;
  for(int i=0;i<d;i++){
    sum+=(x[i]-y[i])*(x[i]-y[i]);
  }
  return sqrt(sum);
}

//Computes the Manhattan Distance
double manhattan(double* x, double* y){
  double sum = 0.0;
  double diff;
  for(int i=0;i<d;i++){
    diff = x[i]-y[i];
    if(diff<0){
      diff = -1*diff;
    }
    sum+=diff;
  }
  return sum;
}

/**
*
*
*   Utility
*
*/

//Reads in the file and returns it as a double**
double** readInArray(char filename[], int type){
  FILE* f;
  char ch;
  f = fopen(filename,"r");
  if(f==NULL){
    printf("Error reading file!");
    exit(-1);
  }

  int r;
  if(type==0){
    r = m;
  }
  else{
    r = n;
  }

  double ** buf = initDoubleStarStar(r,d);
  char* line = malloc(maxline*sizeof(char));
  for(int i=0;i<r;i++){
    for(int j=0;j<d;j++){
      fgets(line,maxline,f);
      buf[i][j] = atof(line);
    }
  }

  free(line);

  fclose(f);
  return buf;
}


//All-purpose function for initializing a double** r x c
double** initDoubleStarStar(int r,int c){
  double ** buf;
  buf = (double**) malloc(r * sizeof(double*));
  for(int i=0;i<r;i++){
    buf[i] = (double*) malloc(c * sizeof(double));
  }

  return buf;
}

//Swaps two members of a double array
void swapD(double* array,int i, int j){
  double temp = array[i];
  array[i] = array[j];
  array[j] = temp;
}

//Swaps two members of an int array
void swapI(int* array,int i, int j){
  int temp = array[i];
  array[i] = array[j];
  array[j] = temp;
}

/*--------------------------------------------------------------------
 * Function:    usage
 * Purpose:     Print command line for function
 * In arg:      prog_name
 */
void usage(char prog_name[]) {
   fprintf(stderr, "usage:  %s <input text file> <query text file> <value of k> <number of P> <number of Q> <dimension>\n", prog_name);
} /* usage */
