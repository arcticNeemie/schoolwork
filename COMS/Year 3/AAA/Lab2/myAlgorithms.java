import java.util.ArrayList;

public class myAlgorithms{
  public static int mergeCount;

  //
  //  Linear Search - returns the index of key in the ArrayList arr. If key is
  //  not in the list, returns -1
  //
  public static int linearSearch(int key, ArrayList<Integer> arr){
    for (int i=0;i<arr.size();i++){
      if(arr.get(i)==key){
        return i;
      }
    }
    return -1;
  }

  //
  //  Bubble Sort - sorts the ArrayList arr. Algorithm does not terminate early
  //
  public static ArrayList<Integer> bubbleNoEscape(ArrayList<Integer> arr){
    for(int i=arr.size()-1;i>0;i--){
      for(int j=0;j<i;j++){
        if(arr.get(j)>arr.get(j+1)){
          int temp = arr.get(j);
          arr.set(j,arr.get(j+1));
          arr.set(j+1,temp);
        }
      }
    }
    return arr;
  }

  //
  //  Bubble Sort - sorts the ArrayList arr. Algorithm does not terminate early
  //              - returns the number of key comparisons
  //
  public static int bubbleNoEscapeCountComparisons(ArrayList<Integer> arr){
    int count = 0;
    for(int i=arr.size()-1;i>0;i--){
      for(int j=0;j<i;j++){
        if(arr.get(j)>arr.get(j+1)){
          int temp = arr.get(j);
          arr.set(j,arr.get(j+1));
          arr.set(j+1,temp);
        }
        count++;
      }
    }
    return count;
  }

  //
  //  Bubble Sort - sorts the ArrayList arr. Algorithm does not terminate early
  //              - returns the number of swaps
  //
  public static int bubbleNoEscapeCountSwaps(ArrayList<Integer> arr){
    int count = 0;
    for(int i=arr.size()-1;i>0;i--){
      for(int j=0;j<i;j++){
        if(arr.get(j)>arr.get(j+1)){
          int temp = arr.get(j);
          arr.set(j,arr.get(j+1));
          arr.set(j+1,temp);
          count++;
        }
      }
    }
    return count;
  }

  //
  //  Bubble Sort - sorts the ArrayList arr. Algorithm may terminate early if
  //  list is sorted
  //
  public static ArrayList<Integer> bubbleEscape(ArrayList<Integer> arr){
    int i = arr.size()-1;
    boolean sorting = true;
    while(i>0 && sorting){
      sorting = false;
      for(int j=0;j<i;j++){
        if(arr.get(j)>arr.get(j+1)){
          int temp = arr.get(j);
          arr.set(j,arr.get(j+1));
          arr.set(j+1,temp);
          sorting = true;
        }
      }
      i--;
    }
    return arr;
  }

  //
  //  Bubble Sort - sorts the ArrayList arr. Algorithm may terminate early if
  //  list is sorted. Returns number of key comparisons
  //
  public static int bubbleEscapeCountComparisons(ArrayList<Integer> arr){
    int i = arr.size()-1;
    int count = 0;
    boolean sorting = true;
    while(i>0 && sorting){
      sorting = false;
      for(int j=0;j<i;j++){
        if(arr.get(j)>arr.get(j+1)){
          int temp = arr.get(j);
          arr.set(j,arr.get(j+1));
          arr.set(j+1,temp);
          sorting = true;
        }
        count++;
      }
      i--;
    }
    return count;
  }

  //
  //  Merge Sort - sorts the ArrayList arr
  //
  public static ArrayList<Integer> mergeSort(ArrayList<Integer> arr){
    int left = 0;
    int right = arr.size()-1;
    return myMerge(arr,left,right);
  }

  //
  //  Merge Sort - sorts the ArrayList arr. Returns number of merges
  //
  public static int mergeCount(ArrayList<Integer> arr){
    int left = 0;
    int right = arr.size()-1;
    mergeCount = 0;
    myMerge(arr,left,right);
    return mergeCount;
  }

  //
  //  Recursive merge sort function called by mergeSort. Left and right
  //  are poisitions in the array, intitially 0 and n-1
  //
  public static ArrayList<Integer> myMerge(ArrayList<Integer> arr, int left, int right){
    if(right-left>0){
      int mid = (int)Math.floor((left+right)/2);
      myMerge(arr,left,mid);
      myMerge(arr,mid+1,right);
      int[] temp = new int[arr.size()];
      for(int i=mid;i>=left;i--){
        temp[i]=arr.get(i);
      }
      for(int j=mid+1;j<=right;j++){
        temp[right+mid+1-j]=arr.get(j);
      }
      int i = left;
      int j = right;
      for(int k=left;k<=right;k++){
        if(temp[i]<temp[j]){
          arr.set(k,temp[i]);
          i++;
        }
        else{
          arr.set(k,temp[j]);
          j--;
        }
      }
    }
    mergeCount++;
    return arr;
  }
}
