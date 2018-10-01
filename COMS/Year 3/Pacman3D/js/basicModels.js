//Make Sphere given radius r and colour col
function makeSphere(r,col){
  var mySphere = new THREE.Mesh(
        new THREE.SphereGeometry(r,32,16,0),
        new THREE.MeshLambertMaterial({
            color: col
        })
    );

    return mySphere;
}

//Make Cube given width, height and depth and material
function makeCube(w,h,d,material){
    var myCube = new THREE.Mesh(
        new THREE.BoxGeometry(w,h,d),
        material
    );
    return myCube;
}

//Make the test wall
function makeTestWall(w,h,d){
    material = new THREE.MeshLambertMaterial({color:"red"});
    return makeCube(w,h,d,material);
}

//Make a line of colour col from origin to end
function makeLine(col,orig,end,w){
    var geometry = new THREE.Geometry();
    geometry.vertices.push(orig);
    geometry.vertices.push(end);
    var material = new THREE.LineBasicMaterial( { color: col } );
	material.linewidth = w;
    var myLine = new THREE.Line(geometry,material);

    return myLine;
}

//Make a cherry
function makeCherry(){
	var cherry = new THREE.Group();
	var cherry1 = makeSphere(1,"#8b0000");
	var cherry2 = makeSphere(1,"#8b0000");
	cherry2.position.set(2,2,2);
	cherry.add(cherry1);
	cherry.add(cherry2);
	var stick1 = makeLine("green",new THREE.Vector3(0,0,0),new THREE.Vector3(1,3,1),3);
	var stick2 = makeLine("green",new THREE.Vector3(2,2,2),new THREE.Vector3(0,4,0),3);
	cherry.add(stick1);
	cherry.add(stick2);
	return cherry;
	
}