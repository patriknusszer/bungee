//=============================================================================================
// Computer Graphics Sample Program: 3D engine-let
// Shader: Gouraud, Phong, NPR
// Material: diffuse + Phong-Blinn
// Texture: CPU-procedural
// Geometry: sphere, tractricoid, torus, mobius, klein-bottle, boy, dini
// Camera: perspective
// Light: point or directional sources
//=============================================================================================
#include "framework.h"
 
//---------------------------
template<class T> struct Dnum { // Dual numbers for automatic derivation
//---------------------------
    float f; // function value
    T d;  // derivatives
    Dnum(float f0 = 0, T d0 = T(0)) { f = f0, d = d0; }
    Dnum operator+(Dnum r) { return Dnum(f + r.f, d + r.d); }
    Dnum operator-(Dnum r) { return Dnum(f - r.f, d - r.d); }
    Dnum operator*(Dnum r) {
        return Dnum(f * r.f, f * r.d + d * r.f);
    }
    Dnum operator/(Dnum r) {
        return Dnum(f / r.f, (r.f * d - r.d * f) / r.f / r.f);
    }
};
 
// Elementary functions prepared for the chain rule as well
template<class T> Dnum<T> Exp(Dnum<T> g) { return Dnum<T>(expf(g.f), expf(g.f)*g.d); }
template<class T> Dnum<T> Sin(Dnum<T> g) { return  Dnum<T>(sinf(g.f), cosf(g.f)*g.d); }
template<class T> Dnum<T> Cos(Dnum<T>  g) { return  Dnum<T>(cosf(g.f), -sinf(g.f)*g.d); }
template<class T> Dnum<T> Tan(Dnum<T>  g) { return Sin(g) / Cos(g); }
template<class T> Dnum<T> Sinh(Dnum<T> g) { return  Dnum<T>(sinh(g.f), cosh(g.f)*g.d); }
template<class T> Dnum<T> Cosh(Dnum<T> g) { return  Dnum<T>(cosh(g.f), sinh(g.f)*g.d); }
template<class T> Dnum<T> Tanh(Dnum<T> g) { return Sinh(g) / Cosh(g); }
template<class T> Dnum<T> Log(Dnum<T> g) { return  Dnum<T>(logf(g.f), g.d / g.f); }
template<class T> Dnum<T> Pow(Dnum<T> g, float n) {
    return  Dnum<T>(powf(g.f, n), n * powf(g.f, n - 1) * g.d);
}
 
typedef Dnum<vec2> Dnum2;
 
const int tessellationLevel = 200;
 
//---------------------------
struct Camera { // 3D camera
//---------------------------
    vec3 wEye, wLookat, wVup;   // extrinsic
    float fov, asp, fp, bp;        // intrinsic
public:
    Camera() {
        asp = (float)windowWidth / windowHeight;
        fov = 75.0f * (float)M_PI / 180.0f;
        fp = 1; bp = 20;
    }
    mat4 V() { // view matrix: translates the center to the origin
        vec3 w = normalize(wEye - wLookat);
        vec3 u = normalize(cross(wVup, w));
        vec3 v = cross(w, u);
        return TranslateMatrix(wEye * (-1)) * mat4(u.x, v.x, w.x, 0,
                                                   u.y, v.y, w.y, 0,
                                                   u.z, v.z, w.z, 0,
                                                   0,   0,   0,   1);
    }
 
    mat4 P() { // projection matrix
        return mat4(1 / (tan(fov / 2)*asp), 0,                0,                      0,
                    0,                      1 / tan(fov / 2), 0,                      0,
                    0,                      0,                -(fp + bp) / (bp - fp), -1,
                    0,                      0,                -2 * fp*bp / (bp - fp),  0);
    }
};
 
//---------------------------
struct Material {
//---------------------------
    vec3 kd, ks, ka;
    float shininess;
};
 
//---------------------------
struct Light {
//---------------------------
    vec3 La, Le;
    vec4 wLightPos; // homogeneous coordinates, can be at ideal point
};
 
//---------------------------
class CheckerBoardTexture : public Texture {
//---------------------------
public:
    CheckerBoardTexture(const int width, const int height) : Texture() {
        std::vector<vec4> image(width * height);
        const vec4 yellow(1, 1, 0, 1), blue(0, 0, 1, 1);
        for (int x = 0; x < width; x++) for (int y = 0; y < height; y++) {
            image[y * width + x] = (x & 1) ^ (y & 1) ? yellow : blue;
        }
        create(width, height, image, GL_NEAREST);
    }
};
 
//---------------------------
struct RenderState {
//---------------------------
    mat4               MVP, M, Minv, V, P;
    Material *         material;
    std::vector<Light> lights;
    Texture *          texture;
    vec3               wEye;
};
 
//---------------------------
class Shader : public GPUProgram {
//---------------------------
public:
    virtual void Bind(RenderState state) = 0;
 
    void setUniformMaterial(const Material& material, const std::string& name) {
        setUniform(material.kd, name + ".kd");
        setUniform(material.ks, name + ".ks");
        setUniform(material.ka, name + ".ka");
        setUniform(material.shininess, name + ".shininess");
    }
 
    void setUniformLight(const Light& light, const std::string& name) {
        setUniform(light.La, name + ".La");
        setUniform(light.Le, name + ".Le");
        setUniform(light.wLightPos, name + ".wLightPos");
    }
};
 
double E(double A, int one, int two) {
    return one + two == 0 ? 0 : A / sqrt(powf(one, 2) + powf( two, 2));
}
 
double* coeffs;
 
double A;
 
void getTerrainInfo(float x, float y, int n, double &height, vec3& norm) {
    x = x * M_PI - M_PI;
    y = y * M_PI - M_PI;
    double total = 0;
    
    for (float one = 0; one <= n; one++) {
        for (float two = 0; two <= n; two++) {
            double e= E(A, one, two);
            double c = cosf(one * x + two * y + coeffs[(int)one * n + (int)two]);
            total += e * c;
        }
    }
 
    height = total;
    
    double dx = 0;
    
    for (float one = 0; one <= n; one++) {
        for (float two = 0; two <= n; two++) {
            double e= E(A, one, two);
            double c = -1 * sinf(one * x + two * y + coeffs[(int)one * n + (int)two]) * one;
            dx += e * c;
        }
    }
    
    double dy = 0;
    
    for (float one = 0; one <= n; one++) {
        for (float two = 0; two <= n; two++) {
            double e= E(A, one, two);
            double c = -1 * sinf(one * x + two * y + coeffs[(int)one * n + (int)two]) * two;
            dy += e * c;
        }
    }
    
    norm = vec3(-dx, 1, -dy);
}
 
 
//---------------------------
class PhongShader : public Shader {
//---------------------------
    const char * vertexSource = R"(
        #version 330
        precision highp float;
 
        struct Light {
            vec3 La, Le;
            vec4 wLightPos;
        };
 
        uniform mat4  MVP, M, Minv; // MVP, Model, Model-inverse
        uniform Light[8] lights;    // light sources
        uniform int   nLights;
        uniform vec3  wEye;         // pos of eye
 
        layout(location = 0) in vec3  vtxPos;            // pos in modeling space
        layout(location = 1) in vec3  vtxNorm;           // normal in modeling space
        layout(location = 2) in float  h;
 
        out vec3 wNormal;            // normal in world space
        out vec3 wView;             // view in world space
        out vec3 wLight[8];            // light dir in world space
        out float wH;
 
        void main() {
            gl_Position = vec4(vtxPos, 1) * MVP; // to NDC
            // vectors for radiance computation
            vec4 wPos = vec4(vtxPos, 1) * M;
            for(int i = 0; i < nLights; i++) {
                wLight[i] = lights[i].wLightPos.xyz * wPos.w - wPos.xyz * lights[i].wLightPos.w;
            }
            wView  = wEye * wPos.w - wPos.xyz;
            wNormal = (Minv * vec4(vtxNorm, 0)).xyz;
            wH = h;
            //texcoord = vtxUV;
        }
    )";
 
    // fragment shader in GLSL
    const char * fragmentSource = R"(
        #version 330
        precision highp float;
 
        struct Light {
            vec3 La, Le;
            vec4 wLightPos;
        };
 
        struct Material {
            vec3 kd, ks, ka;
            float shininess;
        };
 
        uniform Material material;
        uniform Light[8] lights;    // light sources
        uniform int   nLights;
 
        in  vec3 wNormal;       // interpolated world sp normal
        in  vec3 wView;         // interpolated world sp view
        in  vec3 wLight[8];     // interpolated world sp illum dir
        in float wH;
        
        out vec4 fragmentColor; // output goes to frame buffer
 
        void main() {
            vec3 c = wH * 0.7 * (material.kd - vec3(1,0,0)) + material.kd;
            vec3 N = normalize(wNormal);
            vec3 V = normalize(wView);
            if (dot(N, V) < 0) N = -N;    // prepare for one-sided surfaces like Mobius or Klein
 
            vec3 radiance = vec3(0, 0, 0);
            for(int i = 0; i < nLights; i++) {
                vec3 L = normalize(wLight[i]);
                vec3 H = normalize(L + V);
                  float cost = abs(dot(N,L));
                  float cosd = abs(dot(N, H));
                //float cost = max(dot(N,L), 0), cosd = max(dot(N,H), 0);
                // kd and ka are modulated by the texture
                radiance += (c * cost + material.ks * 1/4 * pow(cosd, material.shininess)) * lights[i].Le;
            }
            fragmentColor = vec4(radiance, 1);
        }
    )";
public:
    PhongShader() { create(vertexSource, fragmentSource, "fragmentColor"); }
 
    void Bind(RenderState state) {
        Use();         // make this program run
        setUniform(state.MVP, "MVP");
        setUniform(state.M, "M");
        setUniform(state.Minv, "Minv");
        setUniform(state.wEye, "wEye");
        setUniformMaterial(*state.material, "material");
 
        setUniform((int)state.lights.size(), "nLights");
        for (unsigned int i = 0; i < state.lights.size(); i++) {
            setUniformLight(state.lights[i], std::string("lights[") + std::to_string(i) + std::string("]"));
        }
    }
};
 
//---------------------------
class Geometry {
//---------------------------
protected:
    unsigned int vao, vbo;        // vertex array object
public:
    Geometry() {
        glGenVertexArrays(1, &vao);
        glBindVertexArray(vao);
        glGenBuffers(1, &vbo); // Generate 1 vertex buffer object
        glBindBuffer(GL_ARRAY_BUFFER, vbo);
    }
    virtual void Draw() = 0;
    ~Geometry() {
        glDeleteBuffers(1, &vbo);
        glDeleteVertexArrays(1, &vao);
    }
};
 
//---------------------------
class ParamSurface : public Geometry {
//---------------------------
    struct VertexData {
        vec3 position, normal;
        float h;
    };
 
    unsigned int nVtxPerStrip, nStrips;
public:
    ParamSurface() { nVtxPerStrip = nStrips = 0; }
 
    virtual void eval(Dnum2& U, Dnum2& V, Dnum2& X, Dnum2& Y, Dnum2& Z) = 0;
 
    VertexData GenVertexData(float u, float v) {
        VertexData vtxData;
        //Dnum2 X, Y, Z;
        //Dnum2 U(u, vec2(1, 0)), V(v, vec2(0, 1));
        vec3 norm;
        double h;
        getTerrainInfo(u, v, 35, h, norm);
        //eval(U, V, X, Y, Z);
        //double h;
        
        //getTerrainInfo(U.f, V.f, 60, h);
        //U = U * 15;
        //V = V * 15;
        //X = U - 7.5;
        //Z = V - 7.5;
        //Y = h;
        
        vtxData.position = vec3(u * 15 - 7.5, h, v * 15 - 7.5);
        vtxData.h = h;
        //vtxData.position = vec3(X.f, Y.f, Z.f);
        //vec3 drdU(X.d.x, Y.d.x, Z.d.x), drdV(X.d.y, Y.d.y, Z.d.y);
        vtxData.normal = norm/*cross(drdU, drdV)*/;
        return vtxData;
    }
 
    void create(int N = tessellationLevel, int M = tessellationLevel) {
        nVtxPerStrip = (M + 1) * 2;
        nStrips = N;
        std::vector<VertexData> vtxData;    // vertices on the CPU
        for (int i = 0; i < N; i++) {
            for (int j = 0; j <= M; j++) {
                vtxData.push_back(GenVertexData((float)j / M, (float)i / N));
                vtxData.push_back(GenVertexData((float)j / M, (float)(i + 1) / N));
            }
        }
        
        float min = vtxData[0].h, max = vtxData[0].h;
        
        for (int i = 1; i < vtxData.size(); i++)
        {
            if (vtxData[i].h > max)
                max = vtxData[i].h;
            else if (vtxData[i].h < min)
                min = vtxData[i].h;
        }
        
        for (int i = 0; i < vtxData.size(); i++)
            vtxData[i].h = (vtxData[i].h - min) / (max - min);
        
        glBufferData(GL_ARRAY_BUFFER, nVtxPerStrip * nStrips * sizeof(VertexData), vtxData.data(), GL_STATIC_DRAW);
        // Enable the vertex attribute arrays
        glEnableVertexAttribArray(0);  // attribute array 0 = POSITION
        glEnableVertexAttribArray(1);  // attribute array 1 = NORMAL
        glEnableVertexAttribArray(2);  // attribute array 2 = TEXCOORD0
        // attribute array, components/attribute, component type, normalize?, stride, offset
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(VertexData), (void*)offsetof(VertexData, position));
        glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(VertexData), (void*)offsetof(VertexData, normal));
        glVertexAttribPointer(2, 1, GL_FLOAT, GL_FALSE, sizeof(VertexData), (void*)offsetof(VertexData, h));
    }
 
    void Draw() {
        glBindVertexArray(vao);
        for (unsigned int i = 0; i < nStrips; i++) glDrawArrays(GL_TRIANGLE_STRIP, i *  nVtxPerStrip, nVtxPerStrip);
    }
};
 
 
//---------------------------
class Terrain : public ParamSurface {
//---------------------------
    Dnum2 a = 1.0f, b = 0.15f;
public:
    Terrain() { create(); }
 
    void eval(Dnum2& U, Dnum2& V, Dnum2& X, Dnum2& Y, Dnum2& Z) {
        //double h;
        
        //getTerrainInfo(U.f, V.f, 60, h);
        //U = U * 15;
        //V = V * 15;
        //X = U - 7.5;
        //Z = V - 7.5;
        //Y = h;
    }
};
 
//---------------------------
struct Object {
//---------------------------
    Shader *   shader;
    Material * material;
    Texture *  texture;
    Geometry * geometry;
    vec3 scale, translation, rotationAxis;
    float rotationAngle;
public:
    Object(Shader * _shader, Material * _material, Texture * _texture, Geometry * _geometry) :
        scale(vec3(1, 1, 1)), translation(vec3(0, 0, 0)), rotationAxis(0, 0, 1), rotationAngle(0) {
        shader = _shader;
        texture = _texture;
        material = _material;
        geometry = _geometry;
    }
 
    virtual void SetModelingTransform(mat4& M, mat4& Minv) {
        M = ScaleMatrix(scale) * RotationMatrix(rotationAngle, rotationAxis) * TranslateMatrix(translation);
        Minv = TranslateMatrix(-translation) * RotationMatrix(-rotationAngle, rotationAxis) * ScaleMatrix(vec3(1 / scale.x, 1 / scale.y, 1 / scale.z));
    }
 
    void Draw(RenderState state) {
        mat4 M, Minv;
        SetModelingTransform(M, Minv);
        state.M = M;
        state.Minv = Minv;
        state.MVP = state.M * state.V * state.P;
        state.material = material;
        state.texture = texture;
        shader->Bind(state);
        geometry->Draw();
    }
 
    virtual void Animate(float tstart, float tend) { rotationAngle = 0.8f * tend;
        
    }
};
 
//---------------------------
class Scene {
//---------------------------
    std::vector<Object *> objects;
    Camera camera; // 3D camera
    std::vector<Light> lights;
public:
    void Build() {
        // Shaders
        Shader * phongShader = new PhongShader();
        Material * material1 = new Material;
        material1->kd = vec3(0.5f, 0.25f, 0.1f);
        material1->ks = vec3(0.2f, 0.2f, 0.2f);
        // material1->ka = vec3(0.2f, 0.2f, 0.2f);
        material1->shininess = 1;
        Geometry * terrain = new Terrain();
        Object * terrainobject = new Object(phongShader, material1, new CheckerBoardTexture(20, 20), terrain);
        terrainobject->translation = vec3(0, -3, 0);
        terrainobject->scale = vec3(0.3f, 0.3f, 0.3f);
        terrainobject->rotationAxis = vec3(0, 1, 0);
        objects.push_back(terrainobject);
 
        /*int nObjects = objects.size();
        for (int i = 0; i < 1; i++) {
            Object * object = new Object(*objects[i]);
            object->translation.y -= 3;
            object->shader = gouraudShader;
            objects.push_back(object);
            object = new Object(*objects[i]);
            object->translation.y -= 6;
            object->shader = nprShader;
            objects.push_back(object);
        }*/
 
        // Camera
        camera.wEye = vec3(0, -1, 4);
        camera.wLookat = vec3(0, -2.3, 0);
        camera.wVup = vec3(0, 1, 0);
 
 
        // Lights
        lights.resize(3);
        lights[0].wLightPos = vec4(5, 5, 4, 0);    // ideal point -> directional light source
        lights[0].La = vec3(0.1f, 0.1f, 1);
        lights[0].Le = vec3(1.2, 1 ,0.7);
 
        lights[1].wLightPos = vec4(5, 10, 20, 0);    // ideal point -> directional light source
        lights[1].La = vec3(0.2f, 0.2f, 0.2f);
        lights[1].Le = vec3(0.8, 0.8, 1.1);
 
        lights[2].wLightPos = vec4(-5, 5, 5, 0);    // ideal point -> directional light source
        lights[2].La = vec3(0.1f, 0.1f, 0.1f);
        lights[2].Le = vec3(0.8, 0.8, 0.9);
    }
 
    void Render() {
        RenderState state;
        state.wEye = camera.wEye;
        state.V = camera.V();
        state.P = camera.P();
        state.lights = lights;
        for (Object * obj : objects) obj->Draw(state);
    }
 
    void Animate(float tstart, float tend) {
        for (Object * obj : objects) obj->Animate(tstart, tend);
    }
};
 
Scene scene;
 
// Initialization, create an OpenGL context
void onInitialization() {
    srand(time(0));
    A = 0.5;
    
    coeffs = new double[1000* 1000];
    for (int i = 0; i < 1000* 1000; i++)
        coeffs[i] = (double)rand() / RAND_MAX * 500;
 
    glViewport(0, 0, windowWidth, windowHeight);
    glEnable(GL_DEPTH_TEST);
    glDisable(GL_CULL_FACE);
    scene.Build();
}
 
// Window has become invalid: Redraw
void onDisplay() {
    glClearColor(0.2f, 0.2f, 0.2f, 1.0f);                            // background color
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT); // clear the screen
    scene.Render();
    glutSwapBuffers();                                    // exchange the two buffers
}
 
// Key of ASCII code pressed
void onKeyboard(unsigned char key, int pX, int pY) { }
 
// Key of ASCII code released
void onKeyboardUp(unsigned char key, int pX, int pY) { }
 
// Mouse click event
void onMouse(int button, int state, int pX, int pY) { }
 
// Move mouse with key pressed
void onMouseMotion(int pX, int pY) {
}
 
// Idle event indicating that some time elapsed: do animation here
void onIdle() {
    static float tend = 0;
    const float dt = 0.1f; // dt is infinitesimal
    float tstart = tend;
    tend = glutGet(GLUT_ELAPSED_TIME) / 1000.0f;
 
    for (float t = tstart; t < tend; t += dt) {
        float Dt = fmin(dt, tend - t);
        scene.Animate(t, t + Dt);
    }
    glutPostRedisplay();
}
