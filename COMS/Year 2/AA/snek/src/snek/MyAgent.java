package snek;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.Random;
import za.ac.wits.snake.DevelopmentAgent;

public class MyAgent extends DevelopmentAgent {

    public static void main(String args[]) throws IOException {
        MyAgent agent = new MyAgent();
        MyAgent.start(agent, args);
    }

    @Override
    public void run() {
        try (BufferedReader br = new BufferedReader(new InputStreamReader(System.in))) {
            String initString = br.readLine();
            String[] temp = initString.split(" ");
            int nSnakes = Integer.parseInt(temp[0]);
            while (true) {
                String line = br.readLine();
                if (line.contains("Game Over")) {
                    break;
                }
                String apple1 = line;
                String apple2 = br.readLine();
                //do stuff with apples
                Point rApple = new Point(apple2," ");
                Point iApple = new Point(apple1," ");
                
                //Board
                Board board = new Board(50,50);
                
                board.drawPoint(rApple, 5);
                board.drawPoint(iApple, 6);

                int[] bfs;
                
                Snek me = null;
                Point head = null;
                int mySnakeNum = Integer.parseInt(br.readLine());
                for (int i = 0; i < nSnakes; i++) {
                    String snakeLine = br.readLine();
                    if (i == mySnakeNum) {
                        //hey! That's me :)
                        me = new Snek(snakeLine);
                        head = me.head();
                        board.drawSnake(me, 1);
                        board.drawPoint(head, 2);
                    }
                    else{
                        //Other snakes
                        Snek them = new Snek(snakeLine);
                        board.drawSnake(them, 4);
                        board.drawEnemyHead(them.head());
                    }
                }
                //BFS
                bfs = board.BFS(head);
                if(isValidBFS(bfs)){
                    makeMove(head,bfs[0],bfs[1]);
                }
                else{
                    
                    int[] move = board.wander(head);
                    if(move!=null){
                        makeMove(head,move[0],move[1]);
                    }
                    else{
                        straight();
                    }
                }
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
    
    //Move in directions
    public void up(){
        System.out.println(0);
    }
    
    public void down(){
        System.out.println(1);
    }
    
    public void left(){
        System.out.println(2);
    }
    
    public void right(){
        System.out.println(3);
    }
    
    public void turnLeft(){
        System.out.println(4);
    }
    
    public void straight(){
        System.out.println(5);
    }
    
    public void turnRight(){
        System.out.println(6);
    }
    
    public void print(String s){
        System.out.println("log "+s);
    }
    
    
    //Make a move
    public void makeMove(Point head, int x, int y){
        int mex = head.x;
        int mey = head.y;
        if(mex == x-1){
            right();
        }
        else if(mex == x+1){
            left();
        }
        else if(mey == y-1){
            down();
        }
        else if(mey == y+1){
            up();
        }
        else{
            print("Oops: ");
            turnLeft();
        }
    }
    
    //Check if bfs is valid
    public boolean isValidBFS(int[] bfs){
        return (bfs[0]!=-1 && bfs[1]!=-1);
    }
    
    
    
}