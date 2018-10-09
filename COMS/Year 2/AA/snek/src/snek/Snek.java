/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package snek;

import java.util.ArrayList;

/**
 *
 * @author Tamlin
 */
public class Snek {
    ArrayList<Point> coords;
    
    public Snek(String snakeLine){
		coords = new ArrayList<Point>();
		String[] vals = snakeLine.split(" "); 
		if (vals[0].equals("alive")){
			for (int i=3; i<vals.length; i++){
				Point sp = new Point(vals[i],",");
				coords.add(sp);
			}
		}
                else if(vals[0].equals("invisible")){
                    for (int i=5; i<vals.length; i++){
				Point sp = new Point(vals[i],",");
				coords.add(sp);
			}
                }
	}
    
    public Point head(){
        if(!coords.isEmpty()){
            return coords.get(0);
        }
        //System.out.println("log Empty snek");
        Point p = new Point(-1,-1);
        return p;
    }
   
}
