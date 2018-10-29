import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
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
                
                SnakePoint myApple = new SnakePoint(apple2," ");
                Snake me = null;
                
                //do stuff with apples
                int mySnakeNum = Integer.parseInt(br.readLine());
                for (int i = 0; i < nSnakes; i++) {
                    String snakeLine = br.readLine();
                    if (i == mySnakeNum) {
                        //Print out snake lines
                    	me = new Snake(snakeLine);
                    }
                    //do stuff with snakes
                }
                //Move
                if(myApple.above(me.head())){
                	up();
                }
                else{
                	down();
                }
                
                //finished reading, calculate move:
                System.out.println("log calculating...");
                int move = new Random().nextInt(4);
                System.out.println(move);
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
    
    public void up(){
    	System.out.println(0);
    }
    
    public void down(){
    	System.out.println(1);
    }
    
    public void log(String s){
    	System.out.println("log " + s);
    }
}