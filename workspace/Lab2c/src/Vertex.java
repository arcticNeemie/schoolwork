import java.util.LinkedList;

public class Vertex {
	
	int vertexNumber;
	int colour;
	LinkedList<Vertex> adjacencies;
	
	public Vertex(int n){
		adjacencies = new LinkedList<Vertex>();
		this.vertexNumber = n;
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
}