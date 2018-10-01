
public class Block {
	int i;
	int j;
	int number;
	
	public void setI(int n) {
		i = n;
	}
	public int getI() {
		return i;
	}
	public void setJ(int n) {
		j = n;
	}
	public int getJ() {
		return j;
	}
	public void setNumber(int n) {
		number = n;
	}
	public int getNumber() {
		return number;
	}
	public Block(int x, int y, int num) { // constructor
		i = x; // row
		j = y; // column
		number = num; // number in the current i, j position
	}
}
