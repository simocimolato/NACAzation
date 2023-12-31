/*
Assumptions: 
    - incompressible fluid but compressible "dye" to visualize the flow
    - inviscid fluid

Features:
    - staggered grid (velcity field values u and v stored on grid edges)
    - 3 steps:
        1 applying incompressibility (projection)
            - setting fluid divergence from each cell at 0:
                - computing divergence d at each cell (in general not always = 0)
                - carefully adding/subtracting d/4 from each velocity at each edge so that the new divergence will be 0
                - velocities on walls/obstacles are set to be 0, the divergence is divided by the number of the remaining "free" velocities and subracted/added to them
            - computing pressure in each cell using overrelaxation
        2 move the velocity and dye density fields (semi-lagrangian advection)
        3 draw
*/

var canvas = document.getElementById("simulation");
var c = canvas.getContext("2d");	
canvas.width = window.innerWidth;
canvas.height = window.innerHeight * 1.15;

canvas.focus();

var simHeight = 1.01;	
var cScale = canvas.height / simHeight;
var simWidth = canvas.width / cScale;

var U_FIELD = 0;
var V_FIELD = 1;
var DYE_FIELD = 2;

var scene = {
    paused: false,
    showObstacle: true,
    showPressure: false,
    showFlow: true,
    showCurl: false,
    fluid: null,
    res: 120, // default is 120
    obstacleRadius: 0.1,
    setObstacleX: 0.5,
    setObstacleY: 0.4,
    overRelaxation: 1.9, // should be between 1 and 2
    dt: 1.0 / 60.0,
    numIters: 50,
    domainHeight: 1,
    //cx: [],
    //interval: 25,
    xProportion: 0.03,
    spacing: 10,
    density: 1,
    foilObstacle: false,
};

class Fluid {
    constructor(density, numX, numY, h) {
        this.density = density;
        this.numX = numX + 4; 
        this.numY = numY + 2;
        this.numCells = this.numX * this.numY;
        this.h = h;											 // grid spacing
        this.u = new Float32Array(this.numCells);			 // horizontal velocity
        this.v = new Float32Array(this.numCells);			 // vertical velocity
        this.newU = new Float32Array(this.numCells);
        this.newV = new Float32Array(this.numCells);
        this.p = new Float32Array(this.numCells);			 // pressure
        this.s = new Float32Array(this.numCells);	 		 // keeps track of whether a cell is fluid (s_ij = 1) or wall/obstacle (s_ij = 0)
        this.dyeDensity = new Float32Array(this.numCells);	 // dye density
        this.newDyeDensity = new Float32Array(this.numCells);
        this.curl = new Float32Array(this.numCells); 		 // curl
        this.force = 0;								 		 // force exerted on the body by the fluid
        this.p.fill(0.0);
        this.curl.fill(0.0);
        var num = numX * numY;
        this.maxP = 0;
        this.minP = 0;
    }

    solveIncompressibility(numIters, dt) {
        var n = this.numY;
        var cp = this.density * this.h / dt;

        for (var iter = 0; iter < numIters; iter++) {
            for (var i = 1; i < this.numX-1; i++) {
                for (var j = 1; j < this.numY-1; j++) {
                    if (this.s[i*n + j] == 0.0)
                        continue;

                    var s = this.s[i*n + j];
                    var sx0 = this.s[(i-1)*n + j];
                    var sx1 = this.s[(i+1)*n + j];
                    var sy0 = this.s[i*n + j-1];
                    var sy1 = this.s[i*n + j+1];
                    var s = sx0 + sx1 + sy0 + sy1;

                    if (s == 0.0)
                        continue;

                    // divergence
                    var div = this.u[(i+1)*n + j] - this.u[i*n + j] + this.v[i*n + j+1] - this.v[i*n + j];

                    // computing pressure
                    var p = -div / s;
                    p *= scene.overRelaxation;
                    this.p[i*n + j] += cp * p; 
                    
                    this.u[i*n + j] -= sx0 * p;
                    this.u[(i+1)*n + j] += sx1 * p;
                    this.v[i*n + j] -= sy0 * p;
                    this.v[i*n + j+1] += sy1 * p;
                }
            }
        }

        /*
        // calculating the pressure
        this.force = 0;
        var avgP = mean(this.p);//0.5 * (this.maxP + this.minP);

        for (var i = 1; i < this.numX-1; i++) {
            for (var j = 1; j < this.numY-1; j++) {
                if (this.s[i*n + j] == 0.0)
                    continue;

                var s = this.s[i*n + j];
                var sx0 = this.s[(i-1)*n + j];
                var sx1 = this.s[(i+1)*n + j];
                var sy0 = this.s[i*n + j-1];
                var sy1 = this.s[i*n + j+1];
                var s = sx0 + sx1 + sy0 + sy1;

                if (s == 0.0)
                        continue;

                if (s < 4 && i > 5 && i <this.numX-15 && j>5 && j <this.numY-15){ // temporary condition
                    this.force += this.p[i*n + j] - avgP;
                }
            }
        }

        // CD & CL calculation and printing
        scene.cx.push(Math.abs(2 * this.force / (this.density * scene.inVel * scene.inVel * 2 * scene.obstacleRadius))); // this shit is wrong, should be 100x less
        if(scene.cx.length >= scene.interval){
            // average of the last scene.interval elements
            //document.getElementById("CX").innerHTML = "CX: " + mean(scene.cx).toFixed(4);
            scene.cx = [];
        }*/
    }
    
    setBoundaryCond() {
        var n = this.numY;
        for (var i = 0; i < this.numX; i++) {
            this.u[i*n] = this.u[i*n + 1];
            this.u[i*n + this.numY-1] = this.u[i*n + this.numY-2]; 
        }
        for (var j = 0; j < this.numY; j++) {
            this.v[j] = this.v[n + j];
            this.v[(this.numX-1)*n + j] = this.v[(this.numX-2)*n + j] 
        }
    }

    sampleField(x, y, field) {
        var n = this.numY;
        var h = this.h;
        var h1 = 1 / h;
        var h2 = 0.55 * h;

        x = Math.max(Math.min(x, this.numX * h), h);
        y = Math.max(Math.min(y, this.numY * h), h);

        var dx = 0.0;
        var dy = 0.0;

        var f;

        switch (field) {
            case U_FIELD: f = this.u; 
                dy = h2; 
                break;
            case V_FIELD: f = this.v; 
                dx = h2; 
                break;
            case DYE_FIELD: f = this.dyeDensity; 
                dx = h2; 
                dy = h2; 
                break
        }

        var x0 = Math.min(Math.floor((x-dx)*h1), this.numX-1);
        var tx = ((x-dx) - x0*h) * h1;
        var x1 = Math.min(x0 + 1, this.numX-1);
        
        var y0 = Math.min(Math.floor((y-dy)*h1), this.numY-1);
        var ty = ((y-dy) - y0*h) * h1;
        var y1 = Math.min(y0 + 1, this.numY-1);

        var sx = 1.0 - tx;
        var sy = 1.0 - ty;

        var val = sx*sy * f[x0*n + y0] +
            tx*sy * f[x1*n + y0] +
            tx*ty * f[x1*n + y1] +
            sx*ty * f[x0*n + y1];
        return val;
    }

    avgU(i, j) {
        var n = this.numY;
        var u = (this.u[i*n + j-1] + this.u[i*n + j] + this.u[(i+1)*n + j-1] + this.u[(i+1)*n + j]) * 0.25;
        return u;
    }

    avgV(i, j) {
        var n = this.numY;
        var v = (this.v[(i-1)*n + j] + this.v[i*n + j] + this.v[(i-1)*n + j+1] + this.v[i*n + j+1]) * 0.25;
        return v;
    }

    advectVel(dt) {
        this.newU.set(this.u);
        this.newV.set(this.v);

        var n = this.numY;
        var h = this.h;
        var h2 = 0.5 * h;

        for (var i = 1; i < this.numX; i++) {
            for (var j = 1; j < this.numY; j++) {

                // u component
                if (this.s[i*n + j] != 0.0 && this.s[(i-1)*n + j] != 0.0 && j < this.numY - 1) {
                    var x = i*h;
                    var y = j*h + h2;
                    var u = this.u[i*n + j];
                    var v = this.avgV(i, j);

                    x = x - dt*u;
                    y = y - dt*v;
                    u = this.sampleField(x,y, U_FIELD);
                    this.newU[i*n + j] = u;
                }

                // v component
                if (this.s[i*n + j] != 0.0 && this.s[i*n + j-1] != 0.0 && i < this.numX - 1) {
                    var x = i*h + h2;
                    var y = j*h;
                    var u = this.avgU(i, j);

                    var v = this.v[i*n + j];
                    x = x - dt*u;
                    y = y - dt*v;
                    v = this.sampleField(x,y, V_FIELD);
                    this.newV[i*n + j] = v;
                }
            }	 
        }
        this.u.set(this.newU);
        this.v.set(this.newV);
    }

    // This function is specifically designed for advecting a scalar field representing dye density within the fluid simulation.
    advectDyeDensity(dt) {
        this.newDyeDensity.set(this.dyeDensity);

        var n = this.numY;
        var h = this.h;
        var h2 = 0.5 * h;

        for (var i = 1; i < this.numX-1; i++) {
            for (var j = 1; j < this.numY-1; j++) {

                //if (this.s[i*n + j] != 0.0) {
                    var u = (this.u[i*n + j] + this.u[(i+1)*n + j]) * 0.5;
                    var v = (this.v[i*n + j] + this.v[i*n + j+1]) * 0.5;
                    var x = i*h + h2 - dt*u;
                    var y = j*h + h2 - dt*v;

                    this.newDyeDensity[i*n + j] = this.sampleField(x,y, DYE_FIELD);
                //}
            }	 
        }
        this.dyeDensity.set(this.newDyeDensity);

        // streamlines from the left side
        for (var i = 1; i < this.numX * scene.xProportion; i++) {
            for (var j = 1; j < this.numY-1; j += scene.spacing) {
                this.dyeDensity[i*this.numY + j] = 1;
            }
        }
    }

    simulate(dt, numIters) {
        this.p.fill(0.0);
        this.curl.fill(0.0);
        this.solveIncompressibility(numIters, dt);

        this.setBoundaryCond();
        this.advectVel(dt);
        this.advectDyeDensity(dt);
    }
}

function cX(x) {
    return x * cScale;
}

function cY(y) {
    return canvas.height - y * cScale;
}

function mean(array) {
    // Check if the array is not empty
    if (array.length === 0) {
        return NaN;
    }

    // Calculate the mean using reduce
    return array.reduce((sum, currentValue) => sum + currentValue, 0) / array.length;
}

function linspace(start, end, numPoints) {
    const step = (end - start) / (numPoints - 1);
    return Array.from({ length: numPoints }, (_, index) => start + index * step);
}

function calculateCurl(u, v, sizeX, sizeY) {
    f = scene.fluid;
    // Calculate the curl using finite differences
    for (let i = 1; i < sizeX - 1; i++) {
        for (let j = 1; j < sizeY - 1; j++) {
            // Calculate partial derivatives
            var du_dy = (u[(i - 1) * sizeY + j +1] - u[(i - 1) * sizeY + j -1]) / 2;
            var dv_dx = (v[(i - 1) * sizeY + j +1] - v[(i - 1) * sizeY + j -1]) / 2;

            // Calculate curl
            f.curl[(i - 1) * sizeY + j] = dv_dx - du_dy;
        }
    }
    return f.curl;
}

function setupScene() {
    var domainWidth = scene.domainHeight / simHeight * simWidth;
    var h = scene.domainHeight / scene.res;

    var numX = Math.floor(domainWidth / h);
    var numY = Math.floor(scene.domainHeight / h);

    f = scene.fluid = new Fluid(scene.density, numX, numY, h);
    var n = f.numY;

    // vortex shedding
    scene.inVel = 2.0;
    for (var i = 0; i < f.numX; i++) {
        for (var j = 0; j < f.numY; j++) {
            var s = 1.0;	// fluid
            if (i == 0 || j == 0 || j == f.numY-1)
                s = 0.0;	// solid
            f.s[i*n + j] = s
            f.u[i*n + j] = scene.inVel;
        }
    }
    setObstacle(scene.setObstacleX, scene.setObstacleY)
}

function setObstacle(x, y) {
    scene.obstacleX = x;
    scene.obstacleY = y;
    var r = scene.obstacleRadius;
    var f = scene.fluid;
    var n = f.numY;

    for (var i = 1; i < f.numX; i++) {
        for (var j = 1; j < f.numY-2; j++) {
            f.s[i*n + j] = 1.0;

            dx = (i + 0.5) * f.h - x;
            dy = (j + 0.5) * f.h - y;

            if (dx * dx + dy * dy < r * r) {
                f.s[i*n + j] = 0.0;
                f.u[i*n + j] = 0.0;
                f.u[(i+1)*n + j] = 0.0;
                f.v[i*n + j] = 0.0;
                f.v[i*n + j+1] = 0.0;
            }
        }
    }
    scene.showObstacle = true;
}

// draw -------------------------------------------------------

// inputs are values from 0 to 1 of r g and b and it returns floored 0 - 255 values
function updateScene(option) {
    switch (option) {
        case 'flow':
            scene.showCurl = false;
            scene.showFlow = true;
            scene.showPressure = false;
            break;
        case 'pressure':
            scene.showCurl = false;
            scene.showFlow = false;
            scene.showPressure = true;
            break;
        case 'curl':
            scene.showCurl = true;
            scene.showFlow = false;
            scene.showPressure = false;
            break;
        case 'pause':
            scene.paused = !scene.paused;
            break;
        default:
            break;
    }
    updateButtonStyles();
}

function updateButtonStyles() {
    const buttons = document.querySelectorAll('input[type="radio"]:not(#foil)');
    buttons.forEach(button => {
        const label = button.nextElementSibling;
        if (button.checked) {
            label.style.backgroundColor = '#3498db';
            label.style.color = '#fff';
        } else {
            label.style.backgroundColor = '#f0f0f0';
            label.style.color = '#000';
        }
    });
}

function updatePauseButtonStyle() {
    var button = document.getElementById('pause');
    var label = button.nextElementSibling;
    if (scene.paused) {
        label.style.backgroundColor = '#3498db';
        label.style.color = '#fff';
        label.textContent = "Paused";
    } else {
        label.style.backgroundColor = '#f0f0f0';
        label.style.color = '#000';
        label.textContent = "Pause";
    }
}

function updateSlider() {
    var rangeInput = document.getElementById("inputCamber").value ;
    var displayValue = document.getElementById("camberValue");
    displayValue.innerText = rangeInput + " %";

    var rangeInput = document.getElementById("inputPosition").value;
    var displayValue = document.getElementById("positionValue");
    displayValue.innerText = rangeInput + " %";

    var rangeInput = document.getElementById("inputThickness").value;
    var displayValue = document.getElementById("thicknessValue");
    displayValue.innerText = rangeInput + " %";
}

function updateFoil() {
    var camber = parseFloat(document.getElementById("inputCamber").value) / 100;
    var camberPos = parseFloat(document.getElementById("inputPosition").value) / 100;
    var thickness = parseFloat(document.getElementById("inputThickness").value) / 100;

    if (camber == 0)
        var camberStr = "0";
    else
        var camberStr = String(Math.floor(camber*100));
    if (camberPos == 0)
        var camberPosStr = "0";
    else
        var camberPosStr = String(Math.floor(camberPos*10));
    if (thickness*100 < 10) {
        var leadZero = "0";
        var thicknessStr = leadZero.concat(String(Math.floor(thickness*100)))
    }
    else
        var thicknessStr = String(Math.floor(thickness*100));

    var NACAstring = camberStr.concat(camberPosStr.concat(thicknessStr));

    document.getElementById("NACA").innerHTML = "NACA " + NACAstring;

    var x = linspace(0, Math.PI, 100);
    x = x.map(x => (1 - Math.cos(x)) / 2);
    
    var yc_front = [];

    // camber equations
    if (camberPos != 0) {
        yc_front = x.filter(x => x < camberPos).map(x => (camber / (camberPos ** 2)) * (2 * camberPos * x - x ** 2));
    }

    var yc_back = [];
    for (var i = 0; i < x.length; i++) {
        if (x[i] >= camberPos) {
            yc_back.push((camber / (1 - camberPos) ** 2) * (1 - 2 * camberPos + 2 * camberPos * x[i] - x[i] ** 2));
        }
    }

    var yc = yc_front.concat(yc_back);

    // Gradient equations
    var dycdx_front = [];
    if (camberPos !== 0) {
        dycdx_front = x.filter(x => x < camberPos).map(x => (2 * camber / camberPos ** 2) * (camberPos - x));
    }

    dycdx_back = [];
    for (var i = 0; i < x.length; i++) {
        if (x[i] >= camberPos) {
            dycdx_back.push((2 * camber / (1 - camberPos) ** 2) * (camberPos - x[i]));
        }
    }

    var dycdx = dycdx_front.concat(dycdx_back);

    // Thickness distribution
    var a0 = 0.2969
    var a1 = -0.126
    var a2 = -0.3516
    var a3 = 0.2843
    var a4 = -0.1036 // closed trailing edge, use -0.1015 for open trailing edge

    var yt = [];
    var theta = [];
    var xu = [];
    var yu = [];
    var xl = [];
    var yl = [];
    var xfoil = [];
    var yfoil = [];

    for(var i = 0; i < dycdx.length; i++){
        yt[i] = (thickness / 0.2) * (a0 * x[i] ** 0.5 + a1 * x[i] + a2 * x[i] ** 2 + a3 * x[i] ** 3 + a4 * x[i] ** 4);

        // Calculate envelope positions perpendicular to camber line
        theta[i] = Math.atan(dycdx[i]);

        // Upper line
        xu[i] = x[i] - yt[i] * Math.sin(theta[i]);
        yu[i] = yc[i] + yt[i] * Math.cos(theta[i]);

        // Lower line
        xl[i] = x[i] + yt[i] * Math.sin(theta[i]);
        yl[i] = yc[i] - yt[i] * Math.cos(theta[i]);
    }

    // Combine
    // reverse the upper and lower to ease later transformation;
    // this makes coordinates start at nose and go clockwise around the foil
    xfoil = xu.concat(xl.reverse());
    yfoil = yu.concat(yl.reverse());

    // Tranform & scale xfoil and yfoil up to canvas size
    var xoffset = 0.2;
    var yoffset = 0.4;
    var scaleFactor = 0.4;
    
    var xtran = xfoil;
    var ytran = yfoil;

    // first row is x points, second row is y points
    for(var i = 0; i < xfoil.length; i++){
        xtran[i] = Math.floor(cX(xfoil[i] * scaleFactor + xoffset));
        if (i > 99)
            ytran[i] = Math.floor(cY(yfoil[i] * scaleFactor + yoffset));
        else
            ytran[i] = Math.ceil(cY(yfoil[i] * scaleFactor + yoffset));
    }

    setFoilObstacle(xtran, ytran);
}

function setFoilObstacle(x, y) {
    scene.xfoilTRAN = x;
    scene.yfoilTRAN = y;
    var f = scene.fluid;
    var n = f.numY;
    var h = f.h;

    // narrow down the search field based on max and min boundary points
    // now for loop to check boundary only happens in the region close to the foil 
    var xmin = Math.min(...x);
    var xmax = Math.max(...x);
    var ymin = Math.min(...y);
    var ymax = Math.max(...y);

    // need to separate back to upper & lower
    // makes determining if inside or outside easier
    // note that cY() switches upper and lower values (because -y)
    var yl = y.slice(0,99);
    var yu = y.slice(100,199).reverse();

    for (var i = 1; i < f.numX; i++) {
        for (var j = 1; j < f.numY-2; j++) {

            f.s[i*n + j] = 1.0;

            var xNow = Math.floor(cX(i * h));
            var yTemp = cY((j+1) * h);
            if (yTemp < (ymax+ymin)/2)
                var yNow = Math.ceil(cY((j+1) * h));
            else
                var yNow = Math.floor(cY((j+1) * h));

            if (xNow <= xmax && xNow >=xmin && yNow <= ymax && yNow >= ymin) {
                var diffArr = x.map(temp => Math.abs(xNow - temp));
                var minNumber = Math.min(...diffArr);
                var index = diffArr.findIndex(temp => temp === minNumber);

                if (yNow > yl[index] + 1 && yNow < yu[index] - 1) {
                    f.s[i*n + j] = 0.0;
                    f.u[i*n + j] = 0.0;
                    f.u[(i+1)*n + j] = 0.0;
                    f.v[i*n + j] = 0.0;
                    f.v[i*n + j+1] = 0.0;
                }
            }
        }
    }
    scene.foilObstacle = true;
    scene.showObstacle = true;
}

function getSciColor(val, minVal, maxVal) {
    val = Math.min(Math.max(val, minVal), maxVal- 0.0001);
    var d = maxVal - minVal;
    if (d == 0.0) {
        val = 0.5;
    } else {
        val = (val - minVal) / d;
    }
    var m = 0.25;
    var num = Math.floor(val / m);
    var s = (val - num * m) / m;
    var r, g, b;

    switch(num) {
        case 0: 
            r = 0.0; 
            g = s; 
            b = 1.0; 
            break;
        case 1: 
            r = 0.0; 
            g = 1.0; 
            b = 1.0-s; 
            break;
        case 2: 
            r = s; 
            g = 1.0; 
            b = 0.0; 
            break;
        case 3: 
            r = 1.0; 
            g = 1.0 - s; 
            b = 0.0; 
            break;
    }
    if(scene.showCurl)
        return[255*r, 0*g, 255*b, 255];
    else
        return[255*r, 255*g, 255*b, 255];
}

function draw() {
    c.clearRect(0, 0, canvas.width, canvas.height);
    f = scene.fluid;
    n = f.numY;
    var h = f.h;

    if(scene.showCurl){
        f.curl = calculateCurl(f.u, f.v, f.numX, f.numY)
        minCurl = f.curl[0];
        maxCurl = f.curl[0];
        for (var i = 0; i < f.numCells; i++) {
            minCurl = Math.min(minCurl, f.curl[i]);
            maxCurl = Math.max(maxCurl, f.curl[i]);
        }
    } else if (scene.showPressure){
        var minP = f.p[0];
        var maxP = f.p[0];
        for (var i = 0; i < f.numCells; i++) {
            minP = Math.min(minP, f.p[i]);
            maxP = Math.max(maxP, f.p[i]);
        }
        f.minP = minP;
        f.maxP = maxP;
    }

    id = c.getImageData(0,0, canvas.width, canvas.height)

    var color = [255, 255, 255, 255]

    for (var i = 0; i < f.numX; i++) {
        for (var j = 0; j < f.numY; j++) {
            if (scene.showPressure) {
                var p = f.p[i*n + j];
                color = getSciColor(p, minP, maxP);
            }
            else if (scene.showCurl) {
                var curlTemp = f.curl[i*n + j];
                color = getSciColor(curlTemp, minCurl, maxCurl);
            }
            else if (scene.showFlow) {
                var s = f.dyeDensity[i*n + j];
                color[0] = 255 * s;
                color[1] = 255 * s;
                color[2] = 255 * s;
            }

            var x = Math.floor(cX(i * h));
            var y = Math.floor(cY((j+1) * h));
            var cx = Math.floor(cScale * h) + 1;
            var cy = Math.floor(cScale * h) + 1;

            r = color[0];
            g = color[1];
            b = color[2];

            for (var yi = y; yi < y + cy; yi++) {
                var p = 4 * (yi * canvas.width + x)
                for (var xi = 0; xi < cx; xi++) {
                    id.data[p++] = r;
                    id.data[p++] = g;
                    id.data[p++] = b;
                    id.data[p++] = 255;
                }
            }
        }
    }

    c.putImageData(id, 0, 0);

    if (scene.showObstacle) {
        if (scene.showPressure)
            c.fillStyle = "#000000";
        else
            c.fillStyle = "#DDDDDD";
        if (scene.foilObstacle) {
            c.lineWidth = 1.0;
            c.strokeStyle = "#000000";
            c.beginPath();
            c.moveTo(scene.xfoilTRAN[0],scene.yfoilTRAN[0]);
            for(var i = 1; i < scene.xfoilTRAN.length; i++) {
                c.lineTo(scene.xfoilTRAN[i],scene.yfoilTRAN[i]);
                c.stroke();
            }
            c.closePath();
            c.fill();
        }
        else {
            r = scene.obstacleRadius + f.h;

            c.beginPath();	
            c.arc(cX(scene.obstacleX), cY(scene.obstacleY), cScale * r, 0.0, 2.0 * Math.PI); 
            c.closePath();
            c.fill();

            c.lineWidth = 1.0;
            c.strokeStyle = "#000000";
            c.beginPath();	
            c.arc(cX(scene.obstacleX), cY(scene.obstacleY), cScale * r, 0.0, 2.0 * Math.PI); 
            c.closePath();
            c.stroke();
            c.lineWidth = 0.5;
        }
    }
}

// main -------------------------------------------------------
function simulate() {
    if (!scene.paused)
        scene.fluid.simulate(scene.dt, scene.numIters)
        scene.frameNr++;
}

function update() {
    simulate();
    draw();
    requestAnimationFrame(update);
}

setupScene();
update();