import java.util.Date;
import java.util.Random;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.lang.reflect.Array;

public class lab4 {
	//Sets up which values of n will be tested
	static int startN = 10;
	static int incrementN = 100;
	static int endN = 15000;
	
	//Main method used to execute experiments
	public static void main(String args[]){
		println("Executing experiments...");
		for(int i=1;i<=4;i++){
			experiment(i);
		}
	}
	
	/*
	 * 
	 * Experiment
	 * 
	 */
	
	//Runs experiment ex
	public static void experiment(int ex){
		ArrayList<Long> durs = new ArrayList<>();
		String filename = "bob.csv";
		for(int n=startN;n<endN;n=n+incrementN){
			Point[] set = genSet(n);
			double minD;
			switch(ex){
				case 1:
					long startTime1 = System.currentTimeMillis();
					minD = closestPairBruteForce(set);
					long elapsedTime1 = (new Date()).getTime() - startTime1;
					durs.add(elapsedTime1);
					filename = "experiment1.csv";
					break;
				case 2:
					set = sortPointSet(set);
					long startTime2 = System.currentTimeMillis();
					minD = closestPair0(set);
					long elapsedTime2 = (new Date()).getTime() - startTime2;
					durs.add(elapsedTime2);
					filename = "experiment2.csv";
					break;
				case 3:
					set = sortPointSet(set);
					long startTime3 = System.currentTimeMillis();
					minD = closestPair1(set);
					long elapsedTime3 = (new Date()).getTime() - startTime3;
					durs.add(elapsedTime3);
					filename = "experiment3.csv";
					break;
				case 4:
					set = sortPointSet(set);
					long startTime4 = System.currentTimeMillis();
					minD = closestPair2(set);
					long elapsedTime4 = (new Date()).getTime() - startTime4;
					durs.add(elapsedTime4);
					filename = "experiment4.csv";
					break;
					
			}
			println("Experiment "+ex+": n = "+n);
		}
		writeToCSV(durs,filename);
		println("Experiment "+ex+" successful!");
	}
	
	/*
	 * 
	 * Actual algorithms
	 * 
	 */
	
	//Finds the closest pair by brute force - looks at every pairwise distance and compares
	public static double closestPairBruteForce(Point[] set){
		int n = set.length;
		if(n<2){
			return -1;
		}
		else{
			double d = dist(set[0],set[1]);
			for(int i=0;i<n;i++){
				for(int j=i+1;j<n;j++){
					double tempD = dist(set[i],set[j]);
					if(tempD<d){
						d = tempD;
					}
				}
			}
			return d;
		}
	}
	
	//Implements Algorithm 3
	public static double closestPair0(Point[] set){
		int n= set.length;
		if(n == 2){
			return dist(set[0],set[1]);
		}
		else if(n == 3){
			return closestPairBruteForce(set);
		}
		else{
			//Split in 2
			Point[] set1 = Arrays.copyOfRange(set, 0, n/2);
			Point[] set2 = Arrays.copyOfRange(set, n/2, n);
			//println(set1.length);
			//Calculate distances in subsets
			double d1 = closestPair0(set1);
			double d2 = closestPair0(set2);
			double d = Math.min(d1, d2);
			//Find the subset of points in the strip
			double midLine = (set2[0].getX()-set1[set1.length-1].getX())/2.0;
			double stripR = midLine+d;
			double stripL = midLine-d;
			ArrayList<Point> strip = new ArrayList<>();
			for(int i=set1.length-1;i>=0;i--){
				if(set1[i].getX()>=stripL){
					strip.add(set1[i]);
				}
				else{
					break;
				}
			}
			for(int i=0;i<set2.length;i++){
				if(set2[i].getX()<=stripR){
					strip.add(set2[i]);
				}
				else{
					break;
				}
			}
			Point[] stripSet = strip.toArray(new Point[strip.size()]);
			//Compare distances within strip
			double stripD = closestPairBruteForce(stripSet);
			if(stripD<d){
				d = stripD;
			}
			return d;
		}
	}
	
	//Implements Algorithm 4
		public static double closestPair1(Point[] set){
			int n= set.length;
			if(n == 2){
				return dist(set[0],set[1]);
			}
			else if(n == 3){
				return closestPairBruteForce(set);
			}
			else{
				//Split in 2
				Point[] set1 = Arrays.copyOfRange(set, 0, n/2);
				Point[] set2 = Arrays.copyOfRange(set, n/2, n);
				//println(set1.length);
				//Calculate distances in subsets
				double d1 = closestPair1(set1);
				double d2 = closestPair1(set2);
				double d = Math.min(d1, d2);
				//Find the subset of points in the strip
				double midLine = (set2[0].getX()-set1[set1.length-1].getX())/2.0;
				double stripR = midLine+d;
				double stripL = midLine-d;
				ArrayList<Point> strip = new ArrayList<>();
				for(int i=set1.length-1;i>=0;i--){
					if(set1[i].getX()>=stripL){
						strip.add(set1[i]);
					}
					else{
						break;
					}
				}
				for(int i=0;i<set2.length;i++){
					if(set2[i].getX()<=stripR){
						strip.add(set2[i]);
					}
					else{
						break;
					}
				}
				Point[] stripSet = strip.toArray(new Point[strip.size()]);
				stripSet = sortPointSetY(stripSet);
				//Compare distances within radii
				for(int i=0;i<stripSet.length-1;i++){
					int j=1;
					while(i+j<stripSet.length && j<=7){
						double tempD = dist(stripSet[i],stripSet[i+j]);
						if(tempD<d){
							d = tempD;
						}
						j++;
					}
				}
				return d;
			}
		}
		
		//Implements Algorithm 5
				public static double closestPair2(Point[] set){
					int n= set.length;
					if(n == 2){
						set = sortPointSetY(set);
						return dist(set[0],set[1]);
					}
					else if(n == 3){
						set = sortPointSetY(set);
						return closestPairBruteForce(set);
					}
					else{
						//Split in 2
						Point[] set1 = Arrays.copyOfRange(set, 0, n/2);
						Point[] set2 = Arrays.copyOfRange(set, n/2, n);
						//println(set1.length);
						//Calculate distances in subsets
						double d1 = closestPair2(set1);
						double d2 = closestPair2(set2);
						set = sortPointSet(joinArrayGeneric(set1,set2));
						double d = Math.min(d1, d2);
						//Find the subset of points in the strip
						set1 = sortPointSet(set1);
						set2 = sortPointSet(set2);
						double midLine = (set2[0].getX()-set1[set1.length-1].getX())/2.0;
						double stripR = midLine+d;
						double stripL = midLine-d;
						ArrayList<Point> strip = new ArrayList<>();
						for(int i=0;i<set.length;i++){
							if(set[i].getX()<stripR && set[i].getX()>stripL){
								strip.add(set[i]);
							}
						}
						Point[] stripSet = strip.toArray(new Point[strip.size()]);
						//Compare distances within radii
						for(int i=0;i<stripSet.length-1;i++){
							int j=1;
							while(i+j<stripSet.length && j<=7){
								double tempD = dist(stripSet[i],stripSet[i+j]);
								if(tempD<d){
									d = tempD;
								}
								j++;
							}
						}
						return d;
					}
				}
	
	/*
	 * 
	 * Helpful
	 * 
	 */
	
	//Find the Euclidean distance between the two input points
	public static double dist(Point a, Point b){
		return Math.sqrt(Math.pow(a.x-b.x,2)+Math.pow(a.y-b.y,2));
	}
	
	//Generate a random set of n points in the square [0 10 0 10]
	public static Point[] genSet(int n){
		Point[] set = new Point[n];
		for(int i=0;i<n;i++){
			double x = randDouble(0,10);
			double y = randDouble(0,10);
			Point p = new Point(x,y);
			set[i] = p;
		}
		return set;
	}
	
	//Generate a random double between min and max
	public static double randDouble(double min, double max){
		Random r = new Random();
		return min + (max - min) * r.nextDouble();
	}
	
	//Writes the contents of an arraylist<long> to a csv alongside corresponding n values
	public static void writeToCSV(ArrayList<Long> durs, String name){
		try{
		  PrintWriter writer = new PrintWriter(name, "UTF-8");
		  writer.println("input,time");
		  for (int i=0;i<durs.size();i++){
		    writer.println(startN + incrementN*i+","+durs.get(i));
		  }
		  writer.close();
		}
		catch(Exception e){
			println("Woops");
		}
	}
	
	//Sort an array of Points based on X, and then Y
	public static Point[] sortPointSet(Point[] set){
		Arrays.sort(set,
		    Comparator.comparingDouble(Point::getX)
		              .thenComparingDouble(Point::getY)
	    );
		return set;
	}
	
	//Sort an array of Points based on Y  only
		public static Point[] sortPointSetY(Point[] set){
			Arrays.sort(set,
			    Comparator.comparingDouble(Point::getX)
		    );
			return set;
		}
	
	//Print a set of points
	public static void print(Point[] set){
		for(int i=0;i<set.length;i++){
			println(set[i].getX()+" : "+set[i].getY());
		}
	}
	
	//Print a string on a new line
	public static void println(String s){
		System.out.println(s);
	}
	
	//Print an int on a new line
	public static void println(int i){
		System.out.println(i);
	}
		
	static <T> T[] joinArrayGeneric(T[]... arrays) {
        int length = 0;
        for (T[] array : arrays) {
            length += array.length;
        }

        //T[] result = new T[length];
        final T[] result = (T[]) Array.newInstance(arrays[0].getClass().getComponentType(), length);

        int offset = 0;
        for (T[] array : arrays) {
            System.arraycopy(array, 0, result, offset, array.length);
            offset += array.length;
        }

        return result;
    }
}
