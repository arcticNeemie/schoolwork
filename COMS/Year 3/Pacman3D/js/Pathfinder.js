/*
topright = -26,14

var map = ["XXXXXXXXXXXXXXXXXXXXXXXXXXXX",
           "XB          B 0P          PX",
           "X XXXXXXXXXX XX XXXXXXXXXX X",
           "X XXXXXXXXXX XX XXXXXXXXXX X",
           "XB B   XXB  BXXP  PXX   P PX",
           "XXX XX XX XXXXXXXX XX XX XXX",
           "XXX XX XX XXXXXXXX XX XX XXX",
           "XB BXX   B  B  P  P   XXP PX",
           "X XXXX XXXXX XX XXXXX XXXX X",
           "X XXXX XXXXX XX XXXXX XXXX X",
           "XB          BXXP          PX",
           "XXXXXXXXX XXXXXXXX XXXXXXXXX",
           "                            ",
           "                            ",
           "                            ",
           "                            ",
           "XXXXXXXXX XXXXXXXX XXXXXXXXX",
           "XR          RXXG          GX",
           "X XXXX XXXXX XX XXXXX XXXX X",
           "X XXXX XXXXX XX XXXXX XXXX X",
           "XR RXX   R  R  G  G   XXG GX",
           "XXX XX XX XXXXXXXX XX XX XXX",
           "XXX XX XX XXXXXXXX XX XX XXX",
           "XR R   XXR  RXXG  GXX   G GX",
           "X XXXXXXXXXX XX XXXXXXXXXX X",
           "X XXXXXXXXXX XX XXXXXXXXXX X",
           "XR          R 0G          GX",  // ^ -x
           "XXXXXXXXXXXXXXXXXXXXXXXXXXXX"]; //----> -z

           h: 28
           w: 28

           */
function pinkPath(){
	nodes = ["-19,8","-22,8","-22,11","-19,11","-19,13","-16,13","-16,8","-19,8","-19,-7","-22,-7","-22,-10","-19,-10","-19,-12","-16,-12","-16,-7","-19,-7"];
	return nodes;
}
		   
function orangePath(){
	nodes = ["-10,5","-9,5","-9,2","-6,2","-6,-1","-9,-1","-9,-4","-10,-4","-15,-4","-16,-4","-16,-1","-19,-1","-19,2","-16,2","-16,5","-15,5"];
	return nodes;
}
		   
function purplePath(){
	nodes = ["-25,-1","-25,-12","-22,-12","-22,-10","-19,-10","-19,-12","-16,-12","-16,-1","-19,-1","-19,-4","-22,-4","-22,-1"];
	return nodes;
}
		   
function bluePath(){
	nodes = ["-22,11","-19,11","-19,13","-16,13","-16,2","-19,2","-19,5","-22,5","-22,2","-25,2","-25,13","-22,13"];
	return nodes;
}

function redPath(){
  nodes = ["-3,13","0,13","0,2","-3,2","-3,5","-6,5","-6,2","-9,2","-9,13","-6,13","-6,11","-3,11"];
  return nodes;
}

function greenPath(){
	nodes = ["-9,-1","-6,-1","-6,-4","-3,-4","-3,-1","0,-1","0,-12","-3,-12","-3,-10","-6,-10","-6,-12","-9,-12"];
	return nodes;
}

function cyanPath(){
	nodes = ["-6,5","-3,5","-3,2","0,2","0,-1","-3,-1","-3,-4","-6,-4"];
	return nodes;
}

function findPath(colour){
    var rawNodes;
	if(colour=="red"){
		rawNodes = redPath();
	}
	else if(colour=="green"){
		rawNodes = greenPath();
	}
	else if(colour=="pink"){
		rawNodes = pinkPath();
	}
	else if(colour=="blue"){
		rawNodes = bluePath();
	}
	else if(colour=="purple"){
		rawNodes = purplePath();
	}
	else if(colour=="orange"){
		rawNodes = orangePath();
	}
	else if(colour=="cyan"){
		rawNodes = cyanPath();
	}
    xNodes = [];
    zNodes = [];
    for(var i=0;i<rawNodes.length;i++){
      var thisNode = rawNodes[i].split(",");
      xNodes.push(thisNode[0]*voxel);
      zNodes.push(thisNode[1]*voxel);
    }
	var myPathObject = {
		x: xNodes,
		z: zNodes
	}
	return myPathObject;
  }