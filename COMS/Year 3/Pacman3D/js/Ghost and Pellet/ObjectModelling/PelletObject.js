//pellet object
function Pellet(){};

//add this into scene pelletName.pelletObject
Pellet.prototype.pelletObject = new THREE.Mesh(
    new THREE.SphereGeometry(1,16,16,0,2*Math.PI,0,2*Math.PI),
    new THREE.MeshPhongMaterial(
        {
            color: "yellow",
            specular: "white"
        }
    )) ;

Pellet.prototype.pelletObject.scale.set(1,0.5,0.5);

//call this function to rotate pellet by 0.1
//call by pelletName.animate()
Pellet.prototype.animate = function(){
    this.pelletObject.rotateY(0.1);
};