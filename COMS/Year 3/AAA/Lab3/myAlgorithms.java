import java.util.ArrayList;

public class myAlgorithms{

  //Note - for change, convention is to have coinSets arranged in order from smallest to biggest

  public static int[] minConfig;
  public static int minCoinCount;
  public static int[] minConfigS;
  public static int minWait;

  //
  //  Greedy Making Change - returns the least-coin change of a given number,
  //  using a given coin set. Does so with a greedy approach
  //
  public static int[] greedyChange(int[] coinSet, int value){
    int runningTotal = value;
    int[] minConfigGreedy = new int[coinSet.length];
    for(int i=coinSet.length-1;i>=0;i--){
      int divides = runningTotal/coinSet[i];
      minConfigGreedy[i] = divides;
      runningTotal = value-divides;
    }
    return minConfigGreedy;
  }

  //
  //  Brute Force Making Change - returns the least-coin change of a given number,
  //  using a given coin set. Does so by brute forcing every possible combination
  //  Eugh...
  //
  public static int[] bruteForceChange(int[] coinSet, int value){
    minConfig = new int[coinSet.length];
    int[] currentCombination = new int[coinSet.length];
    for(int i=0;i<coinSet.length;i++){
      minConfig[i] = 0;
      currentCombination[i] = 0;
    }
    minCoinCount = value*2;
    myBruteChange(coinSet,value,0,currentCombination,value);
    return minConfig;
  }

  //
  //  Brute Force - actual algorithm
  //
  public static void myBruteChange(int[] coinSet, int value, int index, int[] currentCombination, int originalValue){
    if(index==coinSet.length){
      int sum = 0;
      int coinCount = 0;
      for(int j=0;j<coinSet.length;j++){
        sum += currentCombination[j]*coinSet[j];
        coinCount += currentCombination[j];
      }
      if(sum==originalValue && coinCount<minCoinCount){
        System.arraycopy(currentCombination,0,minConfig,0,currentCombination.length);
        minCoinCount = coinCount;
      }
    }
    else{
      for(int i=0;i<=value;i++){
        currentCombination[index]=i;
        myBruteChange(coinSet,value-i,index+1,currentCombination,originalValue);
      }
    }

  }

  //
  //  Brute Force Optimal Service
  //
  public static int[] bruteService(ArrayList<Integer> t){
    int n = t.size();
    minConfigS = new int[n];
    int[] currentCombination = new int[n];
    int max = 0;
    for(int i=0;i<n;i++){
      if(t.get(i)>max){
        max = t.get(i);
      }
      currentCombination[i] = i;
    }
    minWait = max*n*100000;
    myBruteService(t,0,currentCombination);
    return minConfigS;
  }

  //
  //  Brute Force Optimal Service - Actual Algorithm
  //
  public static void myBruteService(ArrayList<Integer> t,int index, int[] currentCombination){
    int n = t.size();
    if(index==n){
      //System.out.println("Here");
      int wait = 0;
      int[] used = new int[n];
      boolean unique = true;
      for(int i=0;i<n;i++){
        used[i] = 0;
      }
      for(int i=0;i<n-1;i++){
        wait += (n-currentCombination[i]+1)*t.get(i);
        if(used[i]==0){
          used[i] = 1;
        }
        else{
          unique = false;
          //System.out.println("Not unique");
          break;
        }
      }
      if(unique){
        //System.out.println("Unique with wait time = "+wait+" min="+minWait);
      }
      if(wait<minWait && unique){
        System.arraycopy(currentCombination,0,minConfigS,0,currentCombination.length);
        minWait = wait;
        System.out.println("Found new combination with wait time = "+wait);
        printArray(currentCombination);
      }
    }
    else{
      for(int j=0;j<n;j++){
        currentCombination[j] = j;
        //System.out.println("Index = "+index+" and value = "+j);
        int newIndex = index + 1;
        myBruteService(t,newIndex,currentCombination);
      }
    }
  }


  //
  //Prints out an array, comma separated
  //
  public static void printArray(int[] arr){
    String text = Integer.toString(arr[0]);
    for(int i=1;i<arr.length;i++){
      text += ","+Integer.toString(arr[i]);
    }
    System.out.println(text);
  }

}
