
import java.util.Scanner;

public class Program {
	
	public static void main(String args[]){
		Scanner in = new Scanner(System.in);
		String instructions = in.nextLine();
		String[] ins = instructions.split(" ");
		int snakeNo = Integer.parseInt(ins[0]);
		int rows = Integer.parseInt(ins[1]);
		int cols = Integer.parseInt(ins[2]);
		//Zero Matrix
				int[][] matrix = new int[rows][cols];
				for (int j=0;j<cols;j++){
					for(int i=0;i<rows;i++){
						matrix[i][j]=0;
					}
				}
		String snake;
		for(int i=1;i<=snakeNo;i++){
			 snake = in.nextLine();
			matrix = drawSnake(snake,i,matrix);
		}
		
		
		//Print matrix
		printBoard(matrix);
	}
	
	public static int[][] drawSnake(String bends, int no, int[][] matrix){
		String[] coords = bends.split(" ");
		for(int i=0;i<coords.length-1;i++){
			matrix = drawLine(matrix,coords[i],coords[i+1],no);
		}
		return matrix;
	}
	
	public static int[][] drawLine(int[][] matrix, String c1, String c2, int no){
		String[] p1 = c1.split(",");
		String[] p2 = c2.split(",");
		int minx,maxx,miny,maxy;
		if(Integer.parseInt(p1[0])<Integer.parseInt(p2[0])){
			minx=Integer.parseInt(p1[0]);
			maxx=Integer.parseInt(p2[0]);
		}
		else{
			minx=Integer.parseInt(p2[0]);
			maxx=Integer.parseInt(p1[0]);
		}
		if(Integer.parseInt(p1[1])<Integer.parseInt(p2[1])){
			miny=Integer.parseInt(p1[1]);
			maxy=Integer.parseInt(p2[1]);
		}
		else{
			miny=Integer.parseInt(p2[1]);
			maxy=Integer.parseInt(p1[1]);
		}
		for(int i=minx;i<=maxx;i++){
			for(int j=miny;j<=maxy;j++){
				matrix[i][j] = no;
			}
		}
		return matrix;
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
