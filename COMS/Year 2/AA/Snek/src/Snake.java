import java.util.ArrayList;

public class Snake {
	ArrayList<SnakePoint> coords;
	
	public Snake(String snakeLine){
		coords = new ArrayList<SnakePoint>();
		String[] vals = snakeLine.split(" ");
		if(vals[0].equals("alive")){
			for(int i=3;i<vals.length;i++){
				SnakePoint sp = new SnakePoint(vals[i],",");
				coords.add(sp);
			}
		}
	}
	
	public SnakePoint head(){
		return coords.get(0);
	}
}
