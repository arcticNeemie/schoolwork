
//Constants
var circleResolution = 128;   //Resolution of spheres and circles
var pacmanRadius = 5;         //Radius of pacman
var pacmanRotation = Math.PI/6;   //Rotation of pacman's mouth

//  This function draws pacman
function makePacman(){
  //Make bottom hemisphere
  var botHem = makePacmanHemisphere();

  //Make top hemisphere
  var topHem = makePacmanHemisphere();
  topHem.rotation.z = Math.PI - pacmanRotation;                                 //Rotate to correct orientation
  topHem.position.y = pacmanRadius*Math.sin(pacmanRotation);                    //Move up
  topHem.position.x = pacmanRadius - pacmanRadius*Math.cos(pacmanRotation);     //Adjust position for rotation

  //Make left eye
  var eyeL = makeEye();
  eyeL.position.y = pacmanRadius+topHem.position.y*0.7;
  eyeL.position.z = pacmanRadius*0.5;
  eyeL.rotation.z = pacmanRotation;
  eyeL.rotation.y = 4*Math.PI/3;

  //Make right eye
  var eyeR = makeEye();
  eyeR.position.y = pacmanRadius+topHem.position.y*0.7;
  eyeR.position.z = pacmanRadius*-0.5;
  eyeR.rotation.z = pacmanRotation;
  eyeR.rotation.y = -4*Math.PI/3;

  //Create a group for top half
  var topHalf = new THREE.Group();

  //Add to top half
  topHalf.add(topHem);
  topHalf.add(eyeL);
  topHalf.add(eyeR);

  //Create a group to store components
  var pacman = new THREE.Group();

  //Add to pacman
  pacman.add(botHem);
  pacman.add(topHalf);

  return pacman;
}

//Make hemisphere
function makePacmanHemisphere(){
  //Make hemisphere
  var hSphere = new THREE.Mesh(
    new THREE.SphereGeometry(pacmanRadius,circleResolution,16,0, Math.PI*2, 0, Math.PI/2),
    new THREE.MeshPhongMaterial({
        color: "yellow",
    })
  );
  hSphere.rotation.z = Math.PI; //Rotate to correct orientation

  //Make circle
  var hCircle = new THREE.Mesh(
    new THREE.CircleGeometry(pacmanRadius,circleResolution/2),
    new THREE.MeshPhongMaterial({
        color: "red",
    })
  );
  hCircle.rotation.x = -Math.PI/2; //Rotate to correct orientation

  //Create a group to store components
  var hemisphere = new THREE.Group();

  //Add components to Group
  hemisphere.add(hSphere);
  hemisphere.add(hCircle);

  return hemisphere;
}

//Make eye
function makeEye(){
  var eye = new THREE.Mesh(
    new THREE.SphereGeometry(pacmanRadius/10,circleResolution/4,16),
    new THREE.MeshLambertMaterial({
        map: makeTexture("resources/eyeTex.png")
    })
  );

  return eye;
}
