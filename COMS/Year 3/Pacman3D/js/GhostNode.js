var map = ["XXXXXXXXXXXXXXXXXXXXXXXXXXXX",
           "X*          *  *          *X",
           "X XXXXXXXXXX XX XXXXXXXXXX X",
           "X XXXXXXXXXX XX XXXXXXXXXX X",
           "X* *  *XX*  *XX*  *XX*  * *X",
           "XXX XX XX XXXXXXXX XX XX XXX",
           "XXX XX XX XXXXXXXX XX XX XXX",
           "X* *XX*  *  *  *  *  *XX* *X",
           "X XXXX XXXXX XX XXXXX XXXX X",
           "X XXXX XXXXX XX XXXXX XXXX X",
           "X*    *  *  *XX*  *  *    *X",
           "XXXXXXXXX XXXXXXXX XXXXXXXXX",
           "                            ",
           "                            ",
           "                            ",
           "                            ",
           "XXXXXXXXX XXXXXXXX XXXXXXXXX",
           "X@    @  @  @XX@  @  @    *X",
           "X XXXX XXXXX XX XXXXX XXXX X",
           "X XXXX XXXXX XX XXXXX XXXX X",
           "X@ @XX@  @  @  @  @  @XX@ *X",
           "XXX XX XX XXXXXXXX XX XX XXX",
           "XXX XX XX XXXXXXXX XX XX XXX",
           "X@ @  @XX@  @XX@  @XX@  @ *X",
           "X XXXXXXXXXX XX XXXXXXXXXX X",
           "X XXXXXXXXXX XX XXXXXXXXXX X",
           "X@          @ 0@          *X",  // ^ -x
           "XXXXXXXXXXXXXXXXXXXXXXXXXXXX"]; //----> -z
          /*
           h: 28
           w: 28

           */

function nodeSetup(){
	//Make south nodes
	var node1 = makeNode(0,13);
	var node2 = makeNode(-3,13);
	var node3 = makeNode(-6,13);
	var node4 = makeNode(-9,13);
	var node5 = makeNode(-3,11);
	var node6 = makeNode(-6,11);
	var node7 = makeNode(-3,8);
	var node8 = makeNode(-6,8);
	var node9 = makeNode(-9,8);
	var node10 = makeNode(-3,5);
	var node11 = makeNode(-6,5);
	var node12 = makeNode(-9,5);
	var node13 = makeNode(0,2);
	var node14 = makeNode(-3,2);
	var node15 = makeNode(-6,2);
	var node16 = makeNode(-9,2);
	var node17 = makeNode(0,-1);
	var node18 = makeNode(-3,-1);
	var node19 = makeNode(-6,-1);
	var node20 = makeNode(-9,-1);
	var node21 = makeNode(-3,-4);
	var node22 = makeNode(-6,-4);
	var node23 = makeNode(-9,-4);
	var node24 = makeNode(-3,-7);
	var node25 = makeNode(-6,-7);
	var node26 = makeNode(-9,-7);
	var node27 = makeNode(-3,-10);
	var node28 = makeNode(-6,-10);
	var node29 = makeNode(0,-12);
	var node30 = makeNode(-3,-12);
	var node31 = makeNode(-6,-12);
	var node32 = makeNode(-9,-12);
	//Link nodes
	node1.up = node2;
	node1.right = node12;
	node2.down = node1;
	node2.right = node5;
	node3.right = node6;
	node3.up = node4;
	node4.down = node3;
	node4.right = node9;
	node5.left = node2;
	node5.up = node6;
	node5.right = node7;
	node6.left = node3;
	node6.down = node5;
	node7.left = node5;
	node7.up = node8;
	node8.down = node7;
	node8.up = node9;
	node8.right = node11;
	node9.down = node8;
	node9.left = node4;
	node9.right = node12;
	node10.up = node11;
	node10.right = node14;
	
}

//Make a new node
function makeNode(myX,myZ){
	var myNode = {
		x: myX,
		z: myZ,
	};
	
	return myNode;
}