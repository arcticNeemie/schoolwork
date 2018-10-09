/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package snek;

import java.util.ArrayList;
import java.util.LinkedList;
import java.util.Queue;

/**
 *
 * @author Tamlin
 */
public class Board {
    int[][] board;
    int rows, cols;
    /*
    0 = empty
    1 = me body
    2 = me head
    3 = enemy head
    4 = enemy body
    5 = red apple
    6 = blue apple
    7 = enemy radius
    */
    
    public Board(int rows,int cols){
        this.rows = rows;
        this.cols = cols;
        board = new int[rows][cols];
        for (int i=0;i<rows;i++){
            for (int j=0;j<cols;j++){
                board[i][j] = 0;
            }
        }
    }
    
    public void drawPoint(Point p, int no){
        if(!onBoard(p)){
            return;
        }
        board[p.x][p.y] = no;
    }
    
    public void drawEnemyHead(Point p){
        if(!onBoard(p)){
            return;
        }
        int x = p.x;
        int y = p.y;
        board[x][y] = 3;
        if(onBoard(x+1,y)&&isFree(x+1,y)){
            board[x+1][y]=7;
        }
        if(onBoard(x-1,y)&&isFree(x-1,y)){
            board[x-1][y]=7;
        }
        if(onBoard(x,y+1)&&isFree(x,y+1)){
            board[x][y+1]=7;
        }
        if(onBoard(x,y-1)&&isFree(x,y-1)){
            board[x][y-1]=7;
        }
    }
    
    public boolean onBoard(Point p){
            return (p.x>=0 && p.y>=0 && p.x<rows && p.y<cols);
        }
    
    public boolean onBoard(int x, int y){
            return (x>=0 && y>=0 && x<rows && y<cols);
    }
    
    public int[] wander(Point head){
        int[] move = new int[2];
        int x = head.x;
        int y = head.y;
        if(isFree(x+1,y)){
            move[0] = x+1;
            move[1] = y;
            return move;
        }
        if(isFree(x-1,y)){
            move[0] = x-1;
            move[1] = y;
            return move;
        }
        if(isFree(x,y-1)){
            move[0] = x;
            move[1] = y-1;
            return move;
        }
        if(isFree(x,y+1)){
            move[0] = x;
            move[1] = y+1;
            return move;
        }
        else{
            return null;
        }
    }
    
    //Draw line between two points on the board
    public void drawLine(Point p1, Point p2, int no){
		if(!onBoard(p1)||!onBoard(p2)){
                    return;
                }
                int minx,maxx,miny,maxy;
		if(p1.x<p2.x){
			minx=p1.x;
			maxx=p2.x;
		}
		else{
			minx=p2.x;
			maxx=p1.x;
		}
		if(p1.y<p2.y){
			miny=p1.y;
			maxy=p2.y;
		}
		else{
			miny=p2.y;
			maxy=p1.y;
		}
		for(int i=minx;i<=maxx;i++){
			for(int j=miny;j<=maxy;j++){
				this.board[i][j] = no;
			}
		}
    }
    
    //Draw a snek onto the board
    public void drawSnake(Snek snek, int no){
		for(int i=0;i<snek.coords.size()-1;i++){
			drawLine(snek.coords.get(i),snek.coords.get(i+1),no);
		}
                
	}
    
    public boolean isFree(int x, int y){
        if(onBoard(x,y)){
            int p = board[x][y];
            if(p ==0 || p==5 || p==6){
                return true;
            }
            return false;
        }
        return false;
    }
    
    public boolean isGoal(int x){
        return (x==5 || x==6);
    }
    
    //BFS
    /*
    bfs[0] = x coord of next move
    bfs[1] = y coord of next move
    bfs[2] = distance to apple
    bfs[3] = type of apple
    */
    public int[] BFS(Point root){
        int[] bfs = new int[2];
        if(!onBoard(root)){
            bfs[0] = -1;
            bfs[1] = -1;
            return bfs;
        }
        //Init stuff
        Queue<Point> q = new LinkedList<Point>();
        Point[][] parent = new Point[rows][cols];
        boolean[][] marked = new boolean[rows][cols];
        //Init parents and marked
        for (int i=0;i<rows;i++){
            for (int j=0;j<cols;j++){
                parent[i][j] = null;
                marked[i][j] = false;
            }
        }
        q.add(root);
        marked[root.x][root.y] = true;
        Point goal = new Point(-1,-1);
        boolean found = false;
        while(!q.isEmpty()){
            //System.out.println("log BFS");
            Point w = q.poll();
            int x = w.x;
            int y = w.y;
            Point p;
            //Look down
            if(isFree(x,y+1)&&!marked[x][y+1]){
                marked[x][y+1] = true;
                parent[x][y+1] = w;
                if(isGoal(board[x][y+1])){
                   found = true;
                   goal = new Point(x,y+1);
                   break;
                }
                p = new Point(x,y+1);
                q.add(p);
            }
            //Look up
            if(isFree(x,y-1)&&!marked[x][y-1]){
                marked[x][y-1] = true;
                parent[x][y-1] = w;
                if(isGoal(board[x][y-1])){
                   found = true;
                   goal = new Point(x,y-1);
                   break;
                }
                p = new Point(x,y-1);
                q.add(p);
            }
            //Look left
            if(isFree(x-1,y)&&!marked[x-1][y]){
                marked[x-1][y] = true;
                parent[x-1][y] = w;
                if(isGoal(board[x-1][y])){
                   found = true;
                   goal = new Point(x-1,y);
                   break;
                }
                p = new Point(x-1,y);
                q.add(p);
            }
            //Look right
            if(isFree(x+1,y)&&!marked[x+1][y]){
                marked[x+1][y] = true;
                parent[x+1][y] = w;
                if(isGoal(board[x+1][y])){
                   found = true;
                   goal = new Point(x+1,y);
                   break;
                }
                p = new Point(x+1,y);
                q.add(p);
            }
        }
        if(found){
            Point curr = goal;
            while(!parent[curr.x][curr.y].equals(root)){
                curr = parent[curr.x][curr.y];
            }
            bfs[0] = curr.x;
            bfs[1] = curr.y;
        }
        else{
            bfs[0] = -1;
            bfs[1] = -1;
            return bfs;
        }
        return bfs;
    }
}