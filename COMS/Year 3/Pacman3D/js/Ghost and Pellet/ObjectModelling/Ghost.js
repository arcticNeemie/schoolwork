function makeHead(material){
    var head = new THREE.Mesh(
        new THREE.SphereGeometry(1,32,32,0,2*Math.PI,Math.PI/2,3*Math.PI/2), material
    );
    head.scale.set(1,-1,1);
    head.position.y = 0.5;
    return head;
}

function makeBody(material){
    var body = new THREE.Mesh(
        new THREE.CylinderGeometry(1,1,2,32,32,0,0,2*Math.PI),material
    );
    body.position.y = -0.5;
    return body;
}

function makeMaterial(colour,specularHighlight){
    var material = new THREE.MeshPhongMaterial(
        {
            color: colour,
            transparent: true,
            specular: specularHighlight,
            opacity: 0.25
        }
    );
    return material;
}

function makeCloudMaterial(colour,specularHighlight){
    var cloudMaterial = new THREE.PointCloudMaterial(
        {
            color: colour,
            specular: specularHighlight,
            size: 0.25,
            sizeAttenuation: true
        }
    );
    return cloudMaterial;
}

//makes and returns the ghost's particle-like head
function makeHeadCloud(cloudMaterial){
    var headGeometry = new THREE.Geometry();
    for(var i = 0; i < 500; i++){
        var x = 2*Math.random()-1;
        var y = 2*Math.random();
        var z = 2*Math.random()-1;
        if(x*x + y*y + z*z < 1){
            var point = new THREE.Vector3(x,y,z);
            headGeometry.vertices.push(point);
        }
    }
    var headCloud = new THREE.PointCloud(headGeometry,cloudMaterial);
    headCloud.position.y = 0.5;
    return headCloud;

}

//makes and returns the ghost's particle-like body
function makeBodyCloud(cloudMaterial){
    var bodyGeometry = new THREE.Geometry();
    for(var i = 0; i < 500; i++){
        var x = 2*Math.random()-1;
        var y = 2*Math.random()-1;
        var z = 2*Math.random()-1;
        if(x*x + z*z <= 1){
            var point = new THREE.Vector3(x,y,z);
            bodyGeometry.vertices.push(point);
        }
    }
    var bodyCloud = new THREE.PointCloud(bodyGeometry,cloudMaterial);
    bodyCloud.position.y = -0.5;
    return bodyCloud;
}

//makes and returns a ghost
function makeGhost(colour,specularHighlight){
    var material = makeMaterial(colour,specularHighlight);
    var cloudMaterial = makeCloudMaterial(colour,specularHighlight);
    var headCloud = makeHeadCloud(cloudMaterial);
    var bodyCloud = makeBodyCloud(cloudMaterial);
    var ghost = new THREE.Object3D();
    var head = makeHead(material);
    var body = makeBody(material);
    ghost.add(head);
    ghost.add(headCloud);
    ghost.add(body);
    ghost.add(bodyCloud);
    return ghost;
}
