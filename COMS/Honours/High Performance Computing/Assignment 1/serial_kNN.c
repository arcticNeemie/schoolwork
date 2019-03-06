#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>
#include <math.h>
#include <string.h>

double** readInArray(char filename[]);
int* getParams(char pfilename[],char qfilename[]);
int* serial_kNN(double** P, double** Q, int k);
void usage(char prog_name[]);

int maxline = 20;
int m,n,d;

int main(int argc,char **argv){
  //Check if enough args
  if (argc != 4) {
    //printf("%i\n",argc);
		usage(argv[0]);
		exit (-1);
	}

  char* pfile = argv[1];
  char* qfile = argv[2];
  int k = atoi(argv[3]);

  double** P;
  double** Q;

  //Get arrays
  P = readInArray(pfile);
  Q = readInArray(qfile);

  //Get parameters
  int* params = getParams(pfile,qfile);
  m = params[0];
  n = params[1];
  d = params[2];

  /*
  for(int i=0;i<m;i++){
    for(int j=0;j<d;j++){
      printf("%f,",P[i][j]);
    }
    printf("\n");
  }
  */

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

  char rows[r*c][maxline];

  //Read in lines
  for(int i=0;i<r*c;i++){
    fgets(rows[i],maxline,f);
  }
  fclose(f);

  double ** buf;
  buf = (double**) malloc(r * sizeof(double*));
  for(int i=0;i<r;i++){
    buf[i] = (double*) malloc(c * sizeof(double));
  }
  for(int i=0;i<r;i++){
    for(int j=0;j<c;j++){
      buf[i][j] = atof(rows[i+j]);
    }
  }

  return buf;
}

int* getParams(char pfilename[],char qfilename[]){
  FILE* p;
  FILE* q;
  p = fopen(pfilename,"r");
  q = fopen(qfilename,"r");

  char ps[maxline], qs[maxline], ds[maxline];
  fgets(ps,maxline,p);
  fgets(qs,maxline,q);
  fgets(ds,maxline,p);
  int m = atoi(ps);
  int n = atoi(qs);
  int d = atoi(ds);

  int* params;

  params[0] = m;
  params[1] = n;
  params[2] = d;

  return params;
}


/*--------------------------------------------------------------------
 * Function:    usage
 * Purpose:     Print command line for function
 * In arg:      prog_name
 */
void usage(char prog_name[]) {
   fprintf(stderr, "usage:  %s <input text file> <query text file> <value of k>\n", prog_name);
} /* usage */
