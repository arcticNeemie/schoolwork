import java.util.ArrayList;
import java.io.PrintWriter;

public class lab3{
  //
  //  Class Variables
  //
  static int startTest = 100;    //  Initial n
  static int increment = 100;    //  Number by which n increments
  static int endTest = 3000;     //  Final n

  static int startTestSlow = 2;    //  Initial n
  static int incrementSlow = 2;    //  Number by which n increments
  static int endTestSlow = 8;     //  Final n

  //
  //  Main - used to run experiments
  //
  public static void main(String args[]){
    experiment1a();
    //experiment2a();
    //experiment2b();
  }

  //
  //  Experiment 1a - testing brute force optimal service
  //
  public static void experiment1a(){
    //TODO Fix this
    ArrayList<Long> durs = new ArrayList<>();
    for(int i=startTestSlow;i<=endTestSlow;i=i+incrementSlow){
       ArrayList<Integer> t = arrayMaker.randomList1(i);
       for(int j=0;j<t.size();j++){
         System.out.print("["+t.get(j)+"]");
       }
       System.out.print("\n");
       long start = System.nanoTime();
       int[] minConfig = myAlgorithms.bruteService(t);
       long duration = System.nanoTime()-start;
       durs.add(duration);
       System.out.println("Brute Force Service: n = "+i);
       myAlgorithms.printArray(minConfig);

    }
    writeToCSVTime(durs,"experiment1_bruteForce.csv");
  }

  //
  //  Experiment 2a - testing greedy making change
  //
  public static void experiment2a(){
    ArrayList<Long> durs = new ArrayList<>();
    int[] coinSet = new int[]{1,2,5};
    for(int i=startTest;i<=endTest*100;i=i+increment){
       long start = System.nanoTime();
       int[] minConfig = myAlgorithms.greedyChange(coinSet,i);
       long duration = System.nanoTime()-start;
       durs.add(duration);
       System.out.println("Greedy Change: n = "+i);

    }
    writeToCSVTime(durs,"experiment2_greedy.csv");
  }

  //
  //  Experiment 2b - testing brute force making change
  //
  public static void experiment2b(){
    ArrayList<Long> durs = new ArrayList<>();
    int[] coinSet = new int[]{1,2,5};
    for(int i=startTest;i<=endTest;i=i+increment){
       long start = System.nanoTime();
       int[] minConfig = myAlgorithms.bruteForceChange(coinSet,i);
       long duration = System.nanoTime()-start;
       durs.add(duration);
       System.out.println("Brute Force Change: n = "+i);

    }
    writeToCSVTime(durs,"experiment2_bruteForce.csv");
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
