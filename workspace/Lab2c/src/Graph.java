import java.util.ArrayList;
import java.util.Scanner;

public class Graph {
	
	ArrayList<Vertex> vertices;
	
	public Graph(){
		vertices = new ArrayList<Vertex>();
		Scanner in = new Scanner(System.in);
		String num = in.nextLine();
		int n = Integer.parseInt(num);
		for(int i=0;i<n;i++){
			this.addVertex();
		}
		while(true){
			String edge = in.nextLine();
			if(edge.equals("-1")){
				break;
			}
			else{
				String[] verts = edge.split(",");
				int n1 = Integer.parseInt(verts[0]);
				int n2 = Integer.parseInt(verts[1]);
				addEdge(n1,n2);
			}
		}
		in.close();
		for (int i=0;i<vertices.size();i++){
			System.out.println(i + ":" + getVertex(i).getDegree());
		}
	}
	
	
	public void addVertex(){
		int s = this.vertices.size();
		Vertex v = new Vertex(s);
		this.vertices.add(v);
	}
	
	public Vertex getVertex(int n){
		return vertices.get(n);
	}
	
	public void addEdge(int n1, int n2){
		vertices.get(n1).addAdjacency(vertices.get(n2));
		vertices.get(n2).addAdjacency(vertices.get(n1));
	}

}
