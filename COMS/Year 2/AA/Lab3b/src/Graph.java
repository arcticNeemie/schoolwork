
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
		
		while(getBiggestVertex() != -1){
			vertices.get(getBiggestVertex()).setColour(n);
		}
		//Print colour
		for(int k=0;k<n;k++){
			System.out.println(k+":"+vertices.get(k).colour);
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
	
	public ArrayList<Vertex> dfs(Vertex x, ArrayList<Vertex> tree){
		x.marked = true;
		tree.add(x);
		for(int i=0;i<x.getDegree();i++){
				Vertex d = x.adjacencies.get(i);
				if(!d.marked){
					d.parent = x;
					tree = dfs(d,tree);
				}
		}
		return tree;
	}
	
	public ArrayList<Vertex> sortTree(ArrayList<Vertex> tree){
		//Bubble sort... :(
		for(int i=0;i<tree.size()-1;i++){
			for(int j=i;j<tree.size();j++){
				if(tree.get(j).vertexNumber<tree.get(i).vertexNumber){
					Vertex temp = tree.get(j);
					tree.set(j,tree.get(i));
					tree.set(i, temp);
				}
			}
		}
		return tree;
	}
	
	public int getBiggestVertex(){
		int i =0;
		for(i=0;i<vertices.size();i++){
			if(vertices.get(i).colour == -1){
				break;
			}
		}
		if(i == vertices.size()){
			return -1;
		}
		else{
			Vertex b = vertices.get(i);
			Vertex curr;
			for(int j=i+1;j<vertices.size();j++){
				curr = vertices.get(j);
				if((curr.colour == -1) && curr.getDegree()>b.getDegree()){
					b = curr;
				}
			}
			return b.vertexNumber;
		}
		
	}

}