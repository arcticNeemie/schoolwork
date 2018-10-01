var voxel = 45;
var floorW = 28*voxel;  //420
var floorL = 12*voxel;  //180
var hedgeHeight = voxel;
var floorOffset = 5;

var map = ["XXXXXXXXXXXXXXXXXXXXXXXXXXXX",
           "X                          X",
           "X XXXXXXXXXX XX XXXXXXXXXX X",
           "X XXXXXXXXXX XX XXXXXXXXXX X",
           "X      XX    XX    XX      X",
           "XXX XX XX XXXXXXXX XX XX XXX",
           "XXX XX XX XXXXXXXX XX XX XXX",
           "X   XX                XX   X",
           "X XXXX XXXXX XX XXXXX XXXX X",
           "X XXXX XXXXX XX XXXXX XXXX X",
           "X            XX            X",
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
          /*
           h: 28
           w: 28

           */
//Make South hedges and return them as a group
function makeHedges(){
  var hedges = new THREE.Group();

  //South Hedge
  hedgeS = makeHedge(voxel,floorW,1,0.5);
  //West Hedge
  hedgeW = makeHedge(floorL-2*voxel,voxel,-4.5,14);
  //East Hedge
  hedgeE = makeHedge(floorL-2*voxel,voxel,-4.5,-13);
  //WestBlock1
  hedgeW1 = makeHedge(2*voxel,10*voxel,-1.5,7.5);
  //EastBlock1
  hedgeE1 = makeHedge(2*voxel,10*voxel,-1.5,-6.5);
  //Middle1
  hedgeM1 = makeHedge(3*voxel,2*voxel,-2,0.5);
  //West2
  hedgeW2 = makeHedge(3*voxel,2*voxel,-4,6.5);
  //East2
  hedgeE2 = makeHedge(3*voxel,2*voxel,-4,-5.5);
  //Middle2
  hedgeM2 = makeHedge(2*voxel,8*voxel,-4.5,0.5);
  //West3
  hedgeW3 = makeHedge(2*voxel,2*voxel,-4.5,12.5);
  //East3
  hedgeE3 = makeHedge(2*voxel,2*voxel,-4.5,-11.5);
  //WestMiddle1
  hedgeWM1 = makeHedge(3*voxel,2*voxel,-5,9.5);
  //EastMiddle1
  hedgeEM1 = makeHedge(3*voxel,2*voxel,-5,-8.5);
  //West4
  hedgeW4 = makeHedge(2*voxel,4*voxel,-7.5,10.5);
  //East4
  hedgeE4 = makeHedge(2*voxel,4*voxel,-7.5,-9.5);
  //WestMiddle2
  hedgeWM2 = makeHedge(2*voxel,5*voxel,-7.5,5);
  //EastMiddle2
  hedgeEM2 = makeHedge(2*voxel,5*voxel,-7.5,-4);
  //Middle3
  hedgeM3 = makeHedge(3*voxel,2*voxel,-8,0.5);
  //NW
  hedgeNW = makeHedge(voxel,9*voxel,-10,10);
  //NE
  hedgeNE = makeHedge(voxel,9*voxel,-10,-9);
  //NM
  hedgeNM = makeHedge(voxel,8*voxel,-10,0.5);

  //add
  hedges.add(hedgeS);
  hedges.add(hedgeW);
  hedges.add(hedgeE);
  hedges.add(hedgeW1);
  hedges.add(hedgeE1);
  hedges.add(hedgeM1);
  hedges.add(hedgeW2);
  hedges.add(hedgeE2);
  hedges.add(hedgeM2);
  hedges.add(hedgeE3);
  hedges.add(hedgeW3);
  hedges.add(hedgeWM1);
  hedges.add(hedgeEM1);
  hedges.add(hedgeW4);
  hedges.add(hedgeE4);
  hedges.add(hedgeWM2);
  hedges.add(hedgeEM2);
  hedges.add(hedgeM3);
  hedges.add(hedgeNW);
  hedges.add(hedgeNE);
  hedges.add(hedgeNM);

  return hedges;

}

//Make North hedges and return them as a group
function makeNorthHedges(){
    var hedges = makeHedges();     //Make south hedges

    //Reset hedge positions
    hedges.position.x = 0;
    hedges.position.y = 0;

    //Rotate 180*
    hedges.rotation.y = Math.PI;
    //Translate to correct position
    hedges.position.x = -25*voxel;
    hedges.position.z = 1*voxel;

    return hedges;
}

//Make a hedge rectangle
function makeHedge(r,c,xOff,zOff){
  //Hedge texture from https://opengameart.org/node/22262
  var hedgeTex = makeTexture("resources/vegetation_hedge_34.png");    //Make texture with wrapping
  hedgeTex.wrapS = THREE.RepeatWrapping;
  hedgeTex.wrapT = THREE.RepeatWrapping;
  hedgeTex.repeat.set(c/(2*voxel),1);

  var bumpHedge = makeTexture("resources/vegetation_hedge_34_bm.png");  //Make bump map
  bumpHedge.wrapS = THREE.RepeatWrapping;
  bumpHedge.wrapT = THREE.RepeatWrapping;
  bumpHedge.repeat.set(c/(2*voxel),1);

  var material = new THREE.MeshPhongMaterial({    //Make material
      color: "white",
      map: hedgeTex,
      bumpMap: bumpHedge,
      bumpScale: 0.5
  });

  //Hedge mesh
  var myHedge = new THREE.Mesh(
      new THREE.BoxGeometry(r,hedgeHeight,c),
      material
  );

  //Move hedge to correct position
  myHedge.position.y = hedgeHeight/2 - floorOffset;
  myHedge.position.x = xOff*voxel;
  myHedge.position.z = zOff*voxel;

  return myHedge;

}

//Water
function makeWater(sun){
    var waterGeometry = new THREE.PlaneBufferGeometry( 4*voxel, 28*voxel ); //Make Plane

    //Make water
    var water = new THREE.Water(
        waterGeometry,
        {
            textureWidth: 512,
            textureHeight: 512,
            waterNormals: new THREE.TextureLoader().load( 'resources/waternormals.jpg', function ( texture ) {
                texture.wrapS = texture.wrapT = THREE.RepeatWrapping;
            }),
            alpha: 1.0,
            sunDirection: sun.position.clone().normalize(),
            sunColor: 0xffffff,
            waterColor: 0x001e0f,
            distortionScale:  3.7,
            fog: scene.fog !== undefined
        }
    );

    //Poistion water
    water.rotation.x = -Math.PI/2;
    water.position.x = -12.5*voxel;
    water.position.y = -1*floorOffset;

    return water;
}

//Makes a floor
function makeFloor(){
  //Floor texture with repeat wrapping
  var floorTexture = makeTexture("resources/grass-texture-wall-mural_1000x.jpeg");    //https://www.eazywallz.com/products/grass-texture-wall-mural
  floorTexture.wrapS = THREE.RepeatWrapping;
  floorTexture.wrapT = THREE.RepeatWrapping;
  floorTexture.repeat.set(12,28);

  var floor = makePlane(floorTexture,floorL,floorW);
  floor.position.y = -1*floorOffset;   //Move floor down
  //Translate to map origin
  floor.position.z = 0.5*voxel;
  floor.position.x = -4.5*voxel;

  return floor;
}

//Make floor
function makeNorthFloor(){
    var floor = makeFloor();

    //Move floor northwards
    floor.position.x = -20.5*voxel;

    return floor;
}

/* Do texture stuff */
function makeTexture(imageURL, material) {
    function callback() { // Function to react when image load is done.
        if (material) {
            material.map = texture;  // Add texture to material.
            material.needsUpdate = true;  // Required when material changes.
        }
        render();  // Render scene with texture that has just been loaded.
    }
    var texture = THREE.ImageUtils.loadTexture(imageURL, undefined, callback);
    return texture;
}

//Make plane
function makePlane(myTexture,x,y){
  var myPlane = new THREE.Mesh(
        new THREE.PlaneGeometry(x,y),
        new THREE.MeshLambertMaterial({
            color: "white",
            map: myTexture
        })
    );
  myPlane.rotation.x = -Math.PI/2;

  return myPlane;
}

//Make Skybox
function makeSkybox(){
    /*Cube map*/
    var textureURLs = [  // URLs of the six faces of the cube map
        "resources/skybox/px.jpg",   // Note:  The order in which
        "resources/skybox/nx.jpg",   //   the images are listed is
        "resources/skybox/py.jpg",   //   important!
        "resources/skybox/ny.jpg",
        "resources/skybox/pz.jpg",
        "resources/skybox/nz.jpg"
    ];

    texture = THREE.ImageUtils.loadTextureCube( textureURLs );

    var shader = THREE.ShaderLib[ "cube" ]; // contains the required shaders
    shader.uniforms[ "tCube" ].value = texture; // data for the shaders
    var material = new THREE.ShaderMaterial( {
        // A ShaderMaterial uses custom vertex and fragment shaders.
        fragmentShader: shader.fragmentShader,
        vertexShader: shader.vertexShader,
        uniforms: shader.uniforms,
        depthWrite: false,
        side: THREE.BackSide
    } );

    var skybox = new THREE.Mesh( new THREE.CubeGeometry( 64*voxel, 64*voxel, 64*voxel ), material );
    return skybox;
}

//Make Trampoline
function makeTrampoline(){
  var geometry = new THREE.CylinderGeometry( 10, 10, 1, 32 );
  var material = new THREE.MeshBasicMaterial( {color: "black"} );
  var cylinder = new THREE.Mesh( geometry, material );

  var trampoline = new THREE.Group();
  trampoline.add(cylinder);

  //Poles
  var poleAngle = Math.PI/2;        //Angle between poles
  var poles = new THREE.Group();

  var pole1 = makeTrampolinePole();
  pole1.position.x = 10;
  pole1.position.z = Math.sin(poleAngle);

  var pole2 = makeTrampolinePole();
  pole2.position.x = Math.sin(poleAngle);
  pole2.position.z = 10;

  var pole3 = makeTrampolinePole();
  pole3.position.x = Math.sin(poleAngle);
  pole3.position.z = -10;

  var pole4 = makeTrampolinePole();
  pole4.position.x = -10
  pole4.position.z = Math.sin(poleAngle);

  poles.add(pole1);
  poles.add(pole2);
  poles.add(pole3);
  poles.add(pole4);
  //Vertical offset
  poles.position.y = -2.5;
  trampoline.add(poles);

  trampoline.position.y = -2.5;

  return trampoline;
}

//Make the trampoline pole
function makeTrampolinePole(){
  var geometry = new THREE.CylinderGeometry( 0.1, 0.1, 8, 32 );
  var paisley = makeTexture("resources/paisley.jpg");
  var material = new THREE.MeshPhongMaterial( {
    color: "white",
    map: paisley
  } );
  var cylinder = new THREE.Mesh( geometry, material );

  return cylinder;
}

//Make all 4 trampolines and place them in their correct places
function makeTrampolines(){
  var trampolines = new THREE.Group();

  var tram1 = makeTrampoline();
  tram1.position.x = -10*voxel;
  tram1.position.z = 5*voxel;
  trampolines.add(tram1);

  var tram2 = makeTrampoline();
  tram2.position.x = -10*voxel;
  tram2.position.z = -4*voxel;
  trampolines.add(tram2);

  var tram3 = makeTrampoline();
  tram3.position.x = -15*voxel;
  tram3.position.z = -4*voxel;
  trampolines.add(tram3);

  var tram4 = makeTrampoline();
  tram4.position.x = -15*voxel;
  tram4.position.z = 5*voxel;
  trampolines.add(tram4);

  return trampolines;
}

//Make Pellet
function makePellet(x,z){
    var Pellet;
    var material = new THREE.MeshPhongMaterial(
        {
            color: makeRandomColor(),
            specular: "white"
        }
    );

    var sphereGeometry = new THREE.SphereGeometry(1,16,16,0,2*Math.PI,0,2*Math.PI);
    Pellet = new THREE.Mesh(sphereGeometry,material);
    Pellet.material = material;
    Pellet.geometry = sphereGeometry;
    Pellet.scale.set(2,1,1); //Ellipse
    Pellet.rotation.z = Math.PI/3;
    Pellet.position.x = x*voxel;
    Pellet.position.z = z*voxel;

    return Pellet;
}

//Position pellets on the south
function makeSouthPellets(){
    var pellets = new THREE.Group();
    //Bottom Row Left
    var i;
    for (i=1;i<=13;i++){
        var p = makePellet(0,i);
        pellets.add(p);
    }
    //Bottom Row Right
    for (i=-1;i>=-12;i--){
        var p = makePellet(0,i);
        pellets.add(p);
    }
    //1st columns
    for (i=-1;i>=-3;i--){
        var p1 = makePellet(i,13);
        var p2 = makePellet(i,2);
        var p3 = makePellet(i,-12);
        var p4 = makePellet(i,-1);
        pellets.add(p1);
        pellets.add(p2);
        pellets.add(p3);
        pellets.add(p4);
    }
    //1st Row
    for(i=2;i<=5;i++){
        var p = makePellet(-3,i);
        pellets.add(p);
    }
    for(i=8;i<=12;i++){
        var p = makePellet(-3,i);
        pellets.add(p);
    }
    for(i=-1;i>=-4;i--){
        var p = makePellet(-3,i);
        pellets.add(p);
    }
    for(i=-7;i>=-11;i--){
        var p = makePellet(-3,i);
        pellets.add(p);
    }
    //Middle Columns
    for (i=-4;i>=-6;i--){
        var p1 = makePellet(i,11);
        var p2 = makePellet(i,8);
        var p3 = makePellet(i,5);
        var p4 = makePellet(i,-10);
        var p5 = makePellet(i,-7);
        var p6 = makePellet(i,-4);
        pellets.add(p1);
        pellets.add(p2);
        pellets.add(p3);
        pellets.add(p4);
        pellets.add(p5);
        pellets.add(p6);
    }
    //2nd Rows
    for (i=1;i<=4;i++){
        var p = makePellet(-6,i);
        pellets.add(p);
    }
    for (i=6;i<=7;i++){
        var p = makePellet(-6,i);
        pellets.add(p);
    }
    for (i=12;i<=13;i++){
        var p = makePellet(-6,i);
        pellets.add(p);
    }
    for (i=0;i>=-4;i--){
        var p = makePellet(-6,i);
        pellets.add(p);
    }
    for (i=-5;i>=-6;i--){
        var p = makePellet(-6,i);
        pellets.add(p);
    }
    for (i=-11;i>=-12;i--){
        var p = makePellet(-6,i);
        pellets.add(p);
    }
    //Top Columns
    for (i=-7;i>=-8;i--){
        var p1 = makePellet(i,2);
        var p2 = makePellet(i,8);
        var p3 = makePellet(i,13);
        var p4 = makePellet(i,-1);
        var p5 = makePellet(i,-7);
        var p6 = makePellet(i,-12);
        pellets.add(p1);
        pellets.add(p2);
        pellets.add(p3);
        pellets.add(p4);
        pellets.add(p5);
        pellets.add(p6);
    }
    //Final Row
    for (i=2;i<=13;i++){
        var p = makePellet(-9,i);
        pellets.add(p);
    }
    for (i=-1;i>=-12;i--){
        var p = makePellet(-9,i);
        pellets.add(p);
    }
    return pellets;
}

//Make north pellets
function makeNorthPellets(){
    var nPellets = makeSouthPellets();  //Copy south pellets

    //Reset position
    nPellets.position.x = 0;
    nPellets.position.y = 0;

    //Rotate 180*
    nPellets.rotation.y = Math.PI;

    //Position correctly
    nPellets.position.x = -25*voxel;
    nPellets.position.z = 1*voxel;

    return nPellets;

}

//Random hex colour
function makeRandomColor() {
  var letters = '0123456789ABCDEF';
  var color = '#';
  for (var i = 0; i < 6; i++) {
    color += letters[Math.floor(Math.random() * 16)];
  }
  return color;
}
