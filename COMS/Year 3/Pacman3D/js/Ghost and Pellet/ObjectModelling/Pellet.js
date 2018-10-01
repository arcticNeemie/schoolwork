var Pellet;
var material = new THREE.MeshPhongMaterial(
    {
        color: "yellow",
        specular: "white"
    }
);

var sphereGeometry = new THREE.SphereGeometry(1,16,16,0,2*Math.PI,0,2*Math.PI);
Pellet = new THREE.Mesh(sphereGeometry,material);
Pellet.material = material;
Pellet.geometry = sphereGeometry;
Pellet.scale.set(1,0.5,0.5);

function animatePellet(){
    Pellet.rotateY(0.1);
}

