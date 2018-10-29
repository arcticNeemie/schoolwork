import java.util.Scanner;
import java.util.ArrayList;

public class Program {
	public static void main(String args[]){
		Scanner in = new Scanner(System.in);
		int vno = in.nextInt();
		ArrayList<String> instructions = new ArrayList<String>();
		while(true){
			String ins = in.nextLine();
			if(ins.equals("-1")){
				break;
			}
			else{
				instructions.add(ins);
			}
		}
		in.close();
		int[][] adjMat = new int[vno][vno];
		for(int i=0;i<vno;i++){
			for(int j=0;j<vno;j++){
				adjMat[i][j] = 0;
			}
		}
		for (int i=1; i<instructions.size(); i++){
			int[] vert = splitIns(instructions.get(i));
			adjMat[vert[0]][vert[1]] = 1;
			adjMat[vert[1]][vert[0]] = 1;
		}
		printBoard(adjMat);
	}
	
	public static int[] splitIns(String ins){
		String[] pair = ins.split(",");
		int[] verts = new int[2];
		verts[0] = Integer.parseInt(pair[0]);
		verts[1] = Integer.parseInt(pair[1]);
		return verts;
	}
	
	public static int getDegree(int[][] adjMat, int vertex){
		int degree = 0;
		for(int i=0;i<adjMat.length;i++){
			degree += adjMat[i][vertex];
		}
		return degree;
	}
	
	public static void printBoard(int[][] matrix){
		int rows = matrix.length;
		int cols = matrix[0].length;
		for (int j=0;j<cols;j++){
			for(int i=0;i<rows;i++){
				System.out.print(matrix[i][j]);
				if(i!=rows-1){
					System.out.print(" ");
				}
			}
			System.out.println();
		}
	}
	
}
