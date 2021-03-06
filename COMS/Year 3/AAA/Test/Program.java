import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Date;

public class Program {
	
	static int start = 1000;
	static int inc = 1000;
	static int end = 150000;
	
	public static void main(String args[]){
		ArrayList<Long> dursB = new ArrayList<>();
		ArrayList<Long> dursW = new ArrayList<>();
		ArrayList<Long> dursA = new ArrayList<>();
		for(int n=start;n<=end;n=n+inc){
			//Best
			int[] arr = ascendingList(n);
			long startTime = System.currentTimeMillis();
			int[] sol = algorithm(arr,n);
			long elapsedTime = (new Date()).getTime() - startTime;
			dursB.add(elapsedTime);
			//Worst
			int[] arr2 = descendingList(n);
			long startTime2 = System.currentTimeMillis();
			int[] sol2 = algorithm(arr2,n);
			long elapsedTime2 = (new Date()).getTime() - startTime2;
			dursW.add(elapsedTime2);
			//Average
			int[] arr3 = randomArray(n);
			long startTime3 = System.currentTimeMillis();
			int[] sol3 = algorithm(arr3,n);
			long elapsedTime3 = (new Date()).getTime() - startTime3;
			dursA.add(elapsedTime3);
			print("n="+n);
		}
		writeToCSV(dursB,"bestCase.csv");
		writeToCSV(dursW,"worstCase.csv");
		writeToCSV(dursA,"averageCase.csv");
		
	}
	
	public static int[] algorithm(int[] A,int n){
		for(int i=1;i<n;i++){
			int x = A[i];
			int a = 0;
			int b = i;
			while(a<b){
				int c = (a+b)/2;
				if(x<A[c]){
					b = c;
				}
				else{
					a = c+1;
				}
			}
			for(int j=i;j>a;j--){
				A[j] = A[j-1];
			}
			A[a] = x;
			//print(A);
		}
		return A;
	}
	
	public static void print(int[] arr){
		String x = "";
		for(int i=0;i<arr.length;i++){
			x+=arr[i]+",";
		}
		System.out.println(x);
	}
	
	public static void print(int x){
		System.out.println(x);
	}
	
	public static void print(String x){
		System.out.println(x);
	}
	
	public static int[] randomArray(int n){
		int[] arr = ascendingList(n);
		return shuffleArray(arr);
	}
	
	public static int[] shuffleArray(int[] arr){
		ArrayList<Integer> alist = new ArrayList<>();
		for(int i=0;i<arr.length;i++){
			alist.add(arr[i]);
		}
		Collections.shuffle(alist);
		for(int i=0;i<arr.length;i++){
			arr[i] = alist.get(i);
		}
		return arr;
	}
	
	public static int[] ascendingList(int n){
	   	int[] arr = new int[n];
	    for(int i=0;i<n;i++){
	      arr[i] = i;
	    }
	    return arr;
	}
	
	public static int[] descendingList(int n){
	   	int[] arr = new int[n];
	    for(int i=0;i<n;i++){
	      arr[i] = n-1-i;
	    }
	    return arr;
	}
	
	public static void writeToCSV(ArrayList<Long> durs, String name){
		try{
		  PrintWriter writer = new PrintWriter(name, "UTF-8");
		  writer.println("input,time");
		  for (int i=0;i<durs.size();i++){
		    writer.println(start + inc*i+","+durs.get(i));
		  }
		  writer.close();
		}
		catch(Exception e){
			print("Woops");
		}
	}
}
