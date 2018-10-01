import java.util.ArrayList;
import java.io.PrintWriter;

public class lab2{
  //
  //  Class Variables
  //
  static int startTest = 100;    //  Initial n
  static int increment = 1;    //  Number by which n increments
  static int endTest = 1000;     //  Final n

  //
  //  Main - used to run experiments
  //
  public static void main(String args[]){
    experiment1();
    experiment2();
    experiment3();
    experiment4();
  }

  //
  //  Experiment 1 - testing bubble sort (no escape) against increasing values
  //  of n, counting the number of key comparisons
  //
  public static void experiment1(){
    ArrayList<Integer> arr;
    ArrayList<Integer> counts = new ArrayList<>();
    int count;
    for(int i=startTest;i<=endTest;i=i+increment){
       arr = arrayMaker.randomList(i);
       count = myAlgorithms.bubbleNoEscapeCountComparisons(arr);
       counts.add(count);
       System.out.println("n = "+i);
    }
    writeToCSVCount(counts,"experiment1.csv");
  }

  //
  //  Experiment 2 - testing bubble sort (no escape) against increasing values
  //  of n, counting the number of swaps
  //
  public static void experiment2(){
    ArrayList<Integer> arr;
    ArrayList<Integer> counts = new ArrayList<>();
    int count;
    for(int i=startTest;i<=endTest;i=i+increment){
       arr = arrayMaker.randomList(i);
       count = myAlgorithms.bubbleNoEscapeCountSwaps(arr);
       counts.add(count);
       System.out.println("n = "+i);
    }
    writeToCSVCount(counts,"experiment2.csv");
  }

  //
  //  Experiment 3 - testing bubble sort (with escape) against increasing values
  //  of n, counting the number of comparisons
  //
  public static void experiment3(){
    ArrayList<Integer> arr;
    ArrayList<Integer> counts = new ArrayList<>();
    int count;
    for(int i=startTest;i<=endTest;i=i+increment){
       arr = arrayMaker.randomList(i);
       count = myAlgorithms.bubbleEscapeCountComparisons(arr);
       counts.add(count);
       System.out.println("n = "+i);
    }
    writeToCSVCount(counts,"experiment3.csv");
  }

  //
  //  Experiment 4 - testing merge sort against increasing values
  //  of n, counting the number of merges
  //
  public static void experiment4(){
    ArrayList<Integer> arr;
    ArrayList<Integer> counts = new ArrayList<>();
    int count;
    for(int i=startTest;i<=endTest;i=i+increment){
       arr = arrayMaker.randomList(i);
       count = myAlgorithms.mergeCount(arr);
       counts.add(count);
       System.out.println("n = "+i);
    }
    writeToCSVCount(counts,"experiment4.csv");
  }

  //
  //  Saves the contents of durs to a file.
  //  This is used for time based plots
  //
  public static void writeToCSVTime(ArrayList<Long> durs, String name){
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

  //
  //  Saves the contents of counts to a file.
  //  This is used for count based plots
  //
  public static void writeToCSVCount(ArrayList<Integer> counts, String name){
    try{
      PrintWriter writer = new PrintWriter(name, "UTF-8");
      writer.println("input,time");
      for (int i=0;i<counts.size();i++){
        writer.println((startTest+increment*i) + ","+counts.get(i));
      }
      writer.close();
    }
    catch(Exception e){
        System.out.println("Woops");
    }
  }


}
