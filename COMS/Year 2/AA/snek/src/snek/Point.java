package snek;

public class Point {
	//Coordinates
        int x;
	int y;
	
        //Constructor from coords
	public Point(int x, int y) {
		super();
		this.x = x;
		this.y = y;
	}
	
        //Constructor from string, with separator specified
	public Point(String s, String sep){
		String[] coords = s.split(sep);
		this.x = Integer.parseInt(coords[0]);
		this.y = Integer.parseInt(coords[1]);
	}

	public int getX() {
		return x;
	}
	public void setX(int x) {
		this.x = x;
	}
	public int getY() {
		return y;
	}
	public void setY(int y) {
		this.y = y;
	}
        
        public boolean onBoard(int rows, int cols){
            return (x>=0 && y>=0 && x<rows && y<cols);
        }
        
        public boolean equals(Point p){
            return (this.x == p.x && this.y==p.y);
        }
	
        //Check if point is above the snake's head
	public boolean above(Point other){
		return y<other.y;
	}
	
        //Check if point is below the snake's head
	public boolean below(Point other){
		return y>other.y;
	}
	
}
