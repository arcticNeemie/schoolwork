import java.util.LinkedList;

public class Vertex {
	
	int vertexNumber;
	int colour;
	LinkedList<Vertex> adjacencies;
	Vertex parent;
	boolean marked;
	
	public Vertex(int n){
		adjacencies = new LinkedList<Vertex>();
		this.vertexNumber = n;
		this.parent = null;
		this.marked = false;
		this.colour = -1;
	}
	
	public void addAdjacency(Vertex v){
		adjacencies.add(v);
	}
	
	public boolean isAdjacent(Vertex v){
		return adjacencies.contains(v);
	}
	
	public int getDegree(){
		return adjacencies.size();
	}
	
	public void setColour(int degree){
		int used[] = new int[degree];
		//Initialize
		for(int i=0;i<degree;i++){
			used[i] = 0;
		}
		//Set used
		for(int j=0;j<adjacencies.size();j++){
			if(adjacencies.get(j).colour != -1){
				int c = adjacencies.get(j).colour;
				used[c] = 1;
			}
		}
		//Find first zero value
		for(int k=0;k<degree;k++){
			if(used[k]==0){
				this.colour = k;
				break;
			}
		}
	}
}