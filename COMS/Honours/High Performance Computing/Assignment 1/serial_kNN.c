#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>
#include <math.h>
#include <string.h>

double** readInArray(char filename[]);
int* serial_kNN(double** P, double** Q, int k);
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


}


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

  double ** buf;
  char* line = malloc(maxline*sizeof(char));
  buf = (double**) malloc(r * sizeof(double*));
  for(int i=0;i<r;i++){
    buf[i] = (double*) malloc(c * sizeof(double));
  }
  for(int i=0;i<r;i++){
    for(int j=0;j<c;j++){
      //printf("Hello\n");
      fgets(line,maxline,f);
      //printf("Hello2\n");
      buf[i][j] = atof(line);
    }
  }

  free(line);

  fclose(f);
  return buf;
}


/*--------------------------------------------------------------------
 * Function:    usage
 * Purpose:     Print command line for function
 * In arg:      prog_name
 */
void usage(char prog_name[]) {
   fprintf(stderr, "usage:  %s <input text file> <query text file> <value of k> <number of P> <number of Q> <dimension>\n", prog_name);
} /* usage */
