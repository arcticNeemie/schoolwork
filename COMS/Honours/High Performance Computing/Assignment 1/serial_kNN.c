#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>
#include <math.h>
#include <string.h>

//Preprocess
double** initDoubleStarStar(int r,int c);
double** readInArray(char filename[]);
//Main Algorithm
int** serial_kNN(double** P, double** Q, int k);
//Distance metrics
double euclid(double* x, double* y);
double manhattan(double* x, double* y);
//Sorts
void myQsort(int* indices, double* array, int low, int high);
int partition(int* indices, double* array,int low, int high);

void bubble(int* indices, double* array, int size);
//Misc
void swapD(double* array,int i, int j);
void swapI(int* array,int i, int j);
void usage(char prog_name[]);

int maxline = 20;
int m,n,d;

int main(int argc,char **argv){
  //Check if enough args
  if (argc != 7) {
		usage(argv[0]);
		exit (-1);
	}

  char* pfile = argv[1];
  char* qfile = argv[2];
  int k = atoi(argv[3]);
  m = atoi(argv[4]);
  n = atoi(argv[5]);
  d = atoi(argv[6]);

  double** P;
  double** Q;

  //Get arrays
  P = readInArray(pfile);
  Q = readInArray(qfile);

  int** kNN = serial_kNN(P,Q,k);


  printf("\n");
  for(int j=0;j<k;j++){
    printf("%i ",kNN[0][j]);
  }
  printf("\n");


  //Cleanup
  free(kNN);
  free(P);
  free(Q);
  exit(0);

}

//Takes in P, Q and k and returns the indices of P which are the k-nearest to each qi
int** serial_kNN(double** P, double** Q, int k){
  //Calculate Distances
  double** dist = initDoubleStarStar(n,m);
  for(int i=0;i<n;i++){
    for(int j=0;j<m;j++){
      dist[i][j] = euclid(Q[i],P[j]);
    }
  }

  //Apply sorting
  int** indices;
  indices = (int**) malloc(n * sizeof(int*));
  for(int i=0;i<n;i++){
    indices[i] = (int*) malloc(m * sizeof(int));
    for(int j=0;j<m;j++){
      indices[i][j] = j;
    }
  }

  for(int i=0;i<n;i++){
    bubble(indices[i],dist[i],m);
  }

  for(int i=1;i<=k;i++){
    printf("%f ",dist[0][i]);
  }
  printf("\n");


  free(dist);

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
    myQsort(indices,array, low, pivot - 1);
    myQsort(indices,array, pivot+1, high);
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
  for(int i = 0;i<size-1;i++){
    for (int j = 0; j < size-i-1; j++){
      if(array[j]>array[j+1]){
        swapD(array,j,j+1);
        swapI(indices,j,j+1);
      }
    }
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
  for(int i=0;i<d;i++){
    sum+=abs(x[i]-y[i]);
  }
}

/**
*
*
*   Utility
*
*/


//Reads in the file and returns it as a double**
double** readInArray(char filename[]){
  FILE* f;
  char ch;
  f = fopen(filename,"r");
  if(f==NULL){
    printf("Error reading file!");
    exit(-1);
  }

  char rs[maxline], cs[maxline];
  fgets(rs,maxline,f);
  fgets(cs,maxline,f);
  int r = atoi(rs);
  int c = atoi(cs);

  double ** buf = initDoubleStarStar(r,c);
  char* line = malloc(maxline*sizeof(char));
  for(int i=0;i<r;i++){
    for(int j=0;j<c;j++){
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
