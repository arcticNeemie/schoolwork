import java.util.*;
import java.io.File;
import java.io.PrintWriter;

public class sudokuGenerator{
  public static void main(String args[]) throws Exception{
    String file = "solutions.txt";
    File input = new File(file);
		Scanner reader = new Scanner(input); // setting the file to be read through Scanner
    ArrayList<Block[][]> grid = readSudoku(reader);
    Random rand = new Random();
    int num = 0;
    for(Block[][] sudoku : grid){
      for(int order = 1; order<64; order++){
        int x = rand.nextInt(9);
        int y = rand.nextInt(9);
        while(sudoku[x][y].number == 0){
          x = rand.nextInt(9);
          y = rand.nextInt(9);
        }
        sudoku[x][y].number = 0;
        print("Grid "+num+"\n");
        print(sudoku);
        num++;
      }
    }

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
				print(sudoku[i][j].number + "");
			}
			print("\n");
		}
	}
	public static void print(long n) {
		System.out.print(n);
	}
}
