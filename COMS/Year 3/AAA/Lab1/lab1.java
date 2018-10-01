import java.util.Collections;
import java.util.ArrayList;
import java.io.PrintWriter;

public class lab1{

  static int startTest = 1000;
  static int increment = 1000;
  static int endTest = 1000000;

  public static void main(String args[]){
    //experiment(0,false); //Best
    //experiment(1,false); //Worst
    experiment(0.25,false); //25%
    //experiment(0.5,false); //50%
    //experiment(0.75,false); //75%
    //experiment(0,true); //Random
  }

  public static void experiment(double percent, boolean random){
    ArrayList<Integer> arr;
    ArrayList<Long> durs = new ArrayList<>();
    for(int n = startTest; n <= endTest; n = n + increment){
      int key = randBetween(0,n-1);
      if(random){
        arr = randomList(n);
      }
      else{
        arr = insertIntoList(key,(int)(percent*n),randomList(n));
      }
      long start = System.nanoTime();
      int index = linearSearch(key, arr);
      long duration = System.nanoTime()-start;
      durs.add(duration);
      System.out.println("n = "+n);
    }
    writeToCSV(durs,"linearSearch.csv");
  }

  public static ArrayList<Integer> randomList(int n){
    ArrayList<Integer> myArray = ascendingList(n);
    Collections.shuffle(myArray);
    return myArray;
  }

  public static void printArrayList(ArrayList<Integer> arr){
    for (int i=0;i<arr.size();i++){
      System.out.println(arr.get(i));
    }
  }

  public static ArrayList<Integer> ascendingList(int n){
    ArrayList<Integer> myArray = new ArrayList<>();
    for(int i=0;i<n;i++){
      myArray.add(i);
    }
    return myArray;
  }

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

  public static int linearSearch(int key, ArrayList<Integer> arr){
    for (int i=0;i<arr.size();i++){
      if(arr.get(i)==key){
        return i;
      }
    }
    return -1;
  }

  public static int randBetween(int Min, int Max){
    return (int)Math.random()*(Max - Min + 1) + Min;
  }

  public static void writeToCSV(ArrayList<Long> durs, String name){
    try{
      PrintWriter writer = new PrintWriter(name, "UTF-8");
      writer.println("input,time");
      for (int i=0;i<durs.size();i++){
        writer.println(startTest + increment*i+","+durs.get(i));
      }
      writer.close();
    }
    catch(Exception e){
        System.out.println("Woops");
    }
  }
}
