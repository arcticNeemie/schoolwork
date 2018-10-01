import java.util.Collections;
import java.util.ArrayList;

public class arrayMaker{
  //
  //  Make a random ArrayList of size n containing randomly sorted numbers from 0 until n-1
  //
  public static ArrayList<Integer> randomList(int n){
    ArrayList<Integer> myArray = ascendingList(n);
    Collections.shuffle(myArray);
    return myArray;
  }

  //
  //  Make a random ArrayList of size n containing randomly sorted numbers from 1 until n
  //
  public static ArrayList<Integer> randomList1(int n){
    ArrayList<Integer> myArray = ascendingList1(n);
    Collections.shuffle(myArray);
    return myArray;
  }

  //
  //  Prints the elements of an ArrayList arr
  //
  public static void printArrayList(ArrayList<Integer> arr){
    System.out.println(arr);
  }
  //
  //  Creates an ArrayList of size n with numbers ranging from 0 until n-1, in ascending order
  //
  public static ArrayList<Integer> ascendingList(int n){
    ArrayList<Integer> myArray = new ArrayList<>();
    for(int i=0;i<n;i++){
      myArray.add(i);
    }
    return myArray;
  }

  //
  //  Creates an ArrayList of size n with numbers ranging from 1 until n, in ascending order
  //
  public static ArrayList<Integer> ascendingList1(int n){
    ArrayList<Integer> myArray = new ArrayList<>();
    for(int i=1;i<=n;i++){
      myArray.add(i);
    }
    return myArray;
  }

  //
  //  Inserts an integer, key, into position pos in the ArrayList arr. If the
  //  key is already in the list, it will replace the original instance of the
  //  key with the number the key replaces
  //
  public static ArrayList<Integer> insertIntoList(int key, int pos, ArrayList<Integer> arr){
    int n = arr.size();
    if(arr.contains(key)){
      int i = arr.indexOf(key);
      int temp = arr.get(pos);
      arr.set(i,temp);
    }
    arr.set(pos,key);
    return arr;
  }

  //
  //  Returns a random integer between Min and Max, inclusive
  //
  public static int randBetween(int Min, int Max){
    return (int)Math.random()*(Max - Min + 1) + Min;
  }
}
