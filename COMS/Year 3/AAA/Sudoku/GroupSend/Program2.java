import java.util.*;
import java.io.File;
import java.io.PrintWriter;

public class Program2 {
	public static void main(String[] args) throws Exception{
		// we need to read in the input of the sudoku board
		/*
		Scanner in = new Scanner(System.in);
		print("Enter the number of the input to be run -> ");
		print("\n");
		String file = "input_files/input";
		file = file + in.nextLine().replace(" ", "");
		file = file + ".txt";
		*/
		String file = "sudoku.txt";

		Timer timer = new Timer();

		File input = new File(file);
		Scanner reader = new Scanner(input); // setting the file to be read through Scanner

		ArrayList<Block[][]> grid = readSudoku(reader);

		ArrayList<Integer> comparisons = new ArrayList<>();
		ArrayList<Integer> orders = new ArrayList<>();

		int makeJiaHaoHappy = 0;

		for(Block[][] sudoku : grid){
			orders.add(Order(sudoku));

			int count = solve(sudoku);

			comparisons.add(count);

			makeJiaHaoHappy++;
			System.out.println(makeJiaHaoHappy);

		}

		writeToCSV(comparisons,orders);

		reader.close();
	}

	// checks to see if a number is in the same 3x3 square we want to add a number in
	public static boolean checkSquare(int i, int j, int number, Block[][] sudoku) { // i row, j column
		int y = i % 9; // row
		int x = j % 9; // column

		i = i - (y % 3);
		j = j - (x % 3);
		boolean numberExists = false;

		for(int k = i; k < i + 3; k++) {
			for(int l = j; l < j + 3; l++) {
				if(sudoku[k][l].number == number) {
					numberExists = true;
				}
			}
		}
		return numberExists;
	}
	// checks to see if the same number appears in the row we want add a number
	public static boolean checkRow(int i, int number, Block[][] sudoku) {
		boolean numberExists = false;
		for(int j = 0; j < sudoku.length; j++) {
			if(sudoku[i][j].number == number) {
				numberExists = true;
			}
		}
		return numberExists;
	}
	// checks to see if the same number appears in the column we want to add a number
	public static boolean checkColumn(int j, int number, Block[][] sudoku) {
		boolean numberExists = false;
		for(int i = 0; i < sudoku.length; i++) {
			if(sudoku[i][j].number == number) {
				numberExists = true;
			}
		}
		return numberExists;
	}
	// function which checks if there are any 0's left on the board
	public static boolean isComplete(Block[][] sudoku) {
		boolean complete = true;
		for(int i = 0; i < sudoku.length; i++) {
			for(int j = 0; j < sudoku.length; j++) {
				if(sudoku[i][j].number == 0) {
					complete = false;
				}
			}
		}
		return complete;
	}
	// override functions for printing, makes life easier
	public static void print(String n) {
		System.out.print(n);
	}
	public static void print(int n) {
		System.out.print(n);
	}
	public static void print(Block[][] sudoku) {
		for(int i = 0 ; i < 9; i++) {
			for(int j = 0; j < 9; j++) {
				print(sudoku[i][j].number + " ");
			}
			print("\n");
		}
	}
	public static void print(long n) {
		System.out.print(n);
	}
	public static int Order(Block[][] sudoku) {
		int order = 0;
		for(int i = 0; i < sudoku.length; i++) {
			for (int j = 0; j < sudoku.length; j++) {
				if(sudoku[i][j].number == 0) {
					order++;
				}
			}
		}
		return order;
	}

	public static int solve(Block[][] sudoku){
		Stack<Block> stack = new Stack<>(); // initializing stack for backtracking algorithm
				// Backtracking Algorithm
		int number = 1; // we want to start by trying to put a 1 in the available position
		int i = 0;
		int j = 0;

		int comp = 0;

		while(!isComplete(sudoku)) {
			if(j % 9 == 0 && j != 0) { // iterations working
				i++;
				j = 0;
				comp++;
			}
			Block current = sudoku[i][j];
			if(current.number == 0) {
				comp++;
				if((checkSquare(i, j, number, sudoku) == false) &&
						(checkRow(i, number, sudoku) == false) &&
						(checkColumn(j, number, sudoku) == false) &&
						(number < sudoku.length + 1)) { // we want to keep the number between 1 - 9 for the last case
					// number !> 9, must not be in same row, column, square
					current.number = number;
					stack.push(current);
					number = 0;
					j++;
					comp++;
				}else {
					comp++;
					// reached the step where we need to backtrack
					if(number > sudoku.length - 1) {
						// number here should be 9, as number would increment 1 - 9
						if(!stack.isEmpty()) {
							comp++;
							current.number = 0;
							current = stack.pop();
							i = current.i; // resetting i and j to update current
							j = current.j;
							number = current.number; // setting number to the number that was popped off
							current.number = 0; // setting popped off number to 0 as what was previously added is not correct
						}else {
							// if the stack is empty the program will terminate, should not be empty
							break;
						}
					}
				}
			}else {
				comp++;
				j++;
				number = 0;
			}
			number++;
//			print(sudoku);
//			print("\n");
		}
		return comp;
	}

	public static ArrayList<Block[][]> readSudoku(Scanner reader){
		ArrayList<Block[][]> grids = new ArrayList<Block[][]>();
		Block[][] sudoku = new Block[9][9];

		int count = 0;
		int superCount = 0;
		while(reader.hasNextLine()) { // assigning input to an array
			String line = reader.nextLine();
			if(!line.contains("Grid")){
				for(int i = 0; i < line.length(); i++) {
					sudoku[count][i] = new Block(count, i, Character.getNumericValue(line.charAt(i)));
				}
				count++;
			}
			else{
				if(superCount!=0){
					grids.add(sudoku);
				}
				count = 0;
				sudoku = new Block[9][9]; // we only should be expecting 9 lines
			}
			superCount++;
		}
		grids.add(sudoku);
		return grids;
	}

	public static void writeToCSV(ArrayList<Integer> durs, ArrayList<Integer> orders){
    try{
      PrintWriter writer = new PrintWriter("experiment3.csv", "UTF-8");
      writer.println("input,time");
      for (int i=0;i<durs.size();i++){
        writer.println(orders.get(i)+","+durs.get(i));
      }
      writer.close();
    }
    catch(Exception e){
        System.out.println("Woops");
    }
  }
}

/*

String[] temp = reader.nextLine().split(" ");
for(int i = 0; i < temp.length; i++) {
	sudoku[count][i] = new Block(count, i, Integer.parseInt(temp[i]));
}
count++;

*/
