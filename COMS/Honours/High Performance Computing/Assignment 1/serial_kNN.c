#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>
#include <math.h>
#include <string.h>

double* readInArray(char filename[]);
int* serial_KNN(double* glug);
void usage(char prog_name[]);

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

  readInArray(pfile);

}


//Reads in the file and returns it as an array
double* readInArray(char filename[]){
  FILE* f;
  char ch;
  f = fopen(filename,"r");
  if(f==NULL){
    printf("Error reading p.txt!");
    exit(-1);
  }

  //TODO
  int r = atoi(fgets(f));
  int c = atoi(fgets(f));

  int maxline = 20;

  char rows[r*c][maxline];

  //Read in lines
  for(int i=0;i<r*c;i++){
    fgets(rows[i],maxline,f);
  }
  fclose(f);

  double array[r][c];
  for(int i=0;i<r;i++){
    for(int j=0;j<c;j++){
      array[i][j] = atof(rows[i+j]);
      printf("%f\n",array[i][j]);
    }
  }



  double* car;
  return car;

}


/*--------------------------------------------------------------------
 * Function:    usage
 * Purpose:     Print command line for function
 * In arg:      prog_name
 */
void usage(char prog_name[]) {
   fprintf(stderr, "usage:  %s <input text file> <query text file> <value of k>\n", prog_name);
} /* usage */
