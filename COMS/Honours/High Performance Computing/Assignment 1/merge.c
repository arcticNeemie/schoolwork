/*

//Mergesort parent
void myMergesort(int* indices, double* array){
  int size = sizeof(array)/sizeof(array[0])-1;
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
      myMsort(indices, indices2, array, b, low, mid);
      myMsort(indices, indices2, array, b, mid+1, high);
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

*/
