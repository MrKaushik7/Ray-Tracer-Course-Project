#include <iostream>
#include <vector>
#include <cmath>
#include <cuda.h>
#include <cuda_runtime.h>
#include <fstream>

#define MISS -1
#define threads_per_block 16
#define bounces 5
#define eps 0.001f
#define min_throughput 0.001f

void cudaErrCheck(cudaError_t err) {
    if(err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }
}

template <typename fn>
__host__ void kernelTimer(
    fn kern,
    std::string fileName,
    int W,
    int H,
    int measureRounds = 10
) {

    if(measureRounds <= 0) {
        std::cout << "number of rounds cannot be zero or negative: " << measureRounds << '\n';
        return;
    }

    std::string fTxt = fileName + ".txt";
    std::ofstream output(fTxt);

    if(output.is_open()) {
        cudaEvent_t startEvent, stopEvent;
        cudaErrCheck(cudaEventCreate(&startEvent));
        cudaErrCheck(cudaEventCreate(&stopEvent));

        cudaErrCheck(cudaEventRecord(startEvent));

        for(int i = 0; i < measureRounds; i++) {
            kern();
        }
        cudaErrCheck(cudaGetLastError());

        cudaErrCheck(cudaEventRecord(stopEvent));
        cudaErrCheck(cudaEventSynchronize(stopEvent));

        float ms = 0.0f;
        cudaErrCheck(cudaEventElapsedTime(&ms, startEvent, stopEvent));

        output << "Average kernel time(ms): " << ms / (float)measureRounds << ", " << W << "x" << H << " resolution\n";

        cudaErrCheck(cudaEventDestroy(startEvent));
        cudaErrCheck(cudaEventDestroy(stopEvent));

        output.close();
    }
    else {
        std::cerr << "unable to open file!\n";
    }
}

struct Vec3 {
    float x;
    float y;
    float z;

    __host__ __device__
    Vec3(float a = 0, float b = 0, float c = 0) 
        : x(a), y(b), z(c) {}

    __host__ __device__    
    Vec3 operator+(Vec3 other) const {
        return Vec3(x + other.x, y + other.y, z + other.z);
    }
    __host__ __device__
    Vec3 operator-(Vec3 other) const {
        return Vec3(x - other.x, y - other.y, z - other.z);
    }
    __host__ __device__
    Vec3 operator*(float s) const {
        return Vec3(x * s, y * s, z * s);
    }


};

__host__ __device__ Vec3 operator*(float s, Vec3 v) {
        return v * s;
    }

__host__ __device__ float dot(Vec3 v1, Vec3 v2) {
    return (v1.x * v2.x + v1.y * v2.y + v1.z * v2.z);
}

__host__ __device__ Vec3 normalize(Vec3 v) {
    return v * (1.0f/sqrtf(dot(v, v)));
}

struct Sphere {
    Vec3 center;
    float radius;
    Vec3 colour;
    float reflectivity;
    bool isFloor;
};

__device__ float intersectSphere(const Sphere& sphere, Vec3 camera, Vec3 dirn) {
    Vec3 shifted_center = camera - sphere.center;
    //assume at^2 + bt + c
    float a = dot(dirn, dirn); //mx2 + my2 + mz2
    float b = 2 * dot(shifted_center, dirn); //2(kxmx + kymy + kzmz)
    float c = dot(shifted_center, shifted_center) - sphere.radius * sphere.radius; //kx2 + ky2 + kz2

    float discriminant = b * b - 4 * a * c;

    if(discriminant < 0) return MISS;
    float rootDiscriminant = sqrtf(discriminant);
    float nearest_t = (-b - rootDiscriminant)/ (2 * a);
    float farthest_t = (-b + rootDiscriminant)/ (2 * a); //if we don't assume the camera is outside a sphere, or reflections originate outside a sphere only not within overlaps.
    
    //self intersection
    if(nearest_t > eps) return nearest_t;
    else if(farthest_t > eps) return farthest_t;
    return MISS;
}

__device__ int rayCollisionPoint(const Sphere* scene, Vec3 camera, Vec3 dirn, float& tClosest, int num_spheres) {
    int best = -1;
    float closest = 1e30f;
    for(int i = 0; i < num_spheres; i++) {
        float t = intersectSphere(scene[i], camera, dirn);
        // One thread one ray one pixel so don't need to worry about atomics here
        if(t > 0 && t < closest) {
            closest = t;
            best = i;
        }
    }
    tClosest = closest;
    return best;
}

__device__ bool inShadow(const Sphere* scene, Vec3 camera, Vec3 light, int num_spheres) {
    Vec3 toLight = light - camera;
    float maxDistance = sqrtf(dot(toLight, toLight));
    toLight = normalize(toLight);
    for(int i = 0; i < num_spheres; i++) {
        float t = intersectSphere(scene[i], camera, toLight);
        if(t > 0 && t < maxDistance) {
            return true;
        }
    }
    return false;
}

__device__ Vec3 getFloorBase(Vec3 p_int) {
    int cx = (int)floorf(p_int.x), cz = (int)floorf(p_int.z);
    bool white = ((cx + cz) & 1) == 0;
    return (white ? Vec3(0.9f, 0.9f, 0.9f) : Vec3(0.2f, 0.2f, 0.2f));
}

__device__ Vec3 getSkyGradient(float y) {
// Miss: draw a simple sky gradient (white near horizon -> blue up top).
    float blueRatio = 0.5f * (y + 1.0f);
    return Vec3(1, 1, 1) * (1 - blueRatio) + Vec3(0.4f, 0.6f, 1.0f) * blueRatio;
}

__device__ float computeLighting(const Sphere* scene, Vec3 p_int, Vec3 s_normal, Vec3 light, int num_spheres) {
    Vec3 toLight = normalize(light - p_int);
    float diffuse_light = fmaxf(0.0f, dot(s_normal, toLight));
    float ambient_light = 0.15f;
    if(diffuse_light == 0) return ambient_light;

    // send out a ray towards the light, if it intersects a sphere before the light, it is in shadow
    bool blocked = inShadow(scene, p_int + (s_normal * eps), light, num_spheres);
    float shade =  blocked ? 0.3f : 1.0f;

    return ambient_light + (diffuse_light * shade);
}

__device__ Vec3 accumulatedColour(const Sphere* scene, Vec3 light, Vec3 camera, Vec3 dirn, int num_spheres, int firstCollision, float tClosest) {
    // For the number of bounces, need to keep accumulating colour based on object we collide with times the parent object's reflectivity.
    // The base colour of that pixel will change accordingly.
    Vec3 p_int = camera + tClosest * dirn; // point of intersection with sphere
    Vec3 s_normal = normalize(p_int - scene[firstCollision].center);

    Vec3 base = scene[firstCollision].colour * (1 - scene[firstCollision].reflectivity);
    if(scene[firstCollision].isFloor) {
        base = getFloorBase(p_int) * (1 - scene[firstCollision].reflectivity);
    }
    float brightness = 0.0f;
    if(scene[firstCollision].reflectivity < 1.0f) {
        brightness = computeLighting(scene, p_int, s_normal, light, num_spheres);
    }
    base = base * brightness;

    Vec3 reflColour(0.0f, 0.0f, 0.0f);
    float residue = scene[firstCollision].reflectivity;

    Vec3 p = p_int;
    Vec3 n = s_normal;
    Vec3 refl_dir = dirn;

    for(int d = 0; d < bounces && residue >= min_throughput; d++) {
        float tC;
        refl_dir = refl_dir - 2 * (dot(refl_dir, n)) * n;
        int sphereIndex = rayCollisionPoint(scene, p + n * eps, refl_dir, tC, num_spheres);
        if(sphereIndex == MISS) {
            //sky only since floor considered a sphere!
            reflColour = reflColour + residue * getSkyGradient(refl_dir.y);
            break;
        }
        Sphere sph = scene[sphereIndex];
        p = p + (n * eps) + tC * refl_dir;
        n = normalize(p - sph.center);

        Vec3 sphereColour = sph.isFloor ? getFloorBase(p) : sph.colour;
        float brightness = 0.0f;
        if(sph.reflectivity < 1.0f) {
            brightness = computeLighting(scene, p, n, light, num_spheres);
        }
        
        reflColour = reflColour + residue * brightness * (1 - sph.reflectivity) * sphereColour;
        residue = residue * sph.reflectivity;
    }
    
    return base + reflColour;
}

__device__ Vec3 rayColour(const Sphere* scene, Vec3 camera, Vec3 dirn, int num_spheres) {
    // x = right, y = up, z = toward the camera.
    Vec3 light(4, 6, 2);
    float tClosest;
    int firstCollision = rayCollisionPoint(scene, camera, dirn, tClosest, num_spheres);

    if(firstCollision < 0) {
        return getSkyGradient(dirn.y);
    }
    return accumulatedColour(scene, light, camera, dirn, num_spheres, firstCollision, tClosest);

}

__global__ void ray_trace(const Sphere* scene, unsigned char* img, Vec3 camera, const float aspect_ratio, const int H, const int W, int num_spheres) {
    int py = blockIdx.y * blockDim.y + threadIdx.y;
    int px = blockIdx.x * blockDim.x + threadIdx.x;

    if(py >= H || px >= W) return;

    float u = (2.0f * (px + 0.5f) / W - 1.0f) * aspect_ratio;
    float v = (1.0f - 2.0f * (py + 0.5f) / H);
    Vec3 ray_dir = normalize(Vec3(u, v, -1.5f));

    Vec3 colour = rayColour(scene, camera, ray_dir, num_spheres);

    int rgb_idx = (py * W + px) * 3;
    img[rgb_idx] = (unsigned char)(fminf(colour.x, 1.0f) * 255);
    img[rgb_idx + 1] = (unsigned char)(fminf(colour.y, 1.0f) * 255);
    img[rgb_idx + 2] = (unsigned char)(fminf(colour.z, 1.0f) * 255);
}

int main() {


    int num_spheres = sizeof(h_scene) / sizeof(h_scene[0]);

    Sphere* scene;
    cudaErrCheck(cudaMalloc((void**)&scene, sizeof(h_scene)));

    cudaErrCheck(cudaMemcpy(scene, h_scene, sizeof(h_scene), cudaMemcpyHostToDevice));

    const int W = 800;
    const int H = 600;

    unsigned char* image = (unsigned char* )malloc(H * W * 3 * sizeof(unsigned char));
    unsigned char* d_image;
    cudaErrCheck(cudaMalloc((void**)&d_image, H * W * 3 * sizeof(unsigned char)));

    Vec3 camera(0.0f, 1.2f, 4.0f);
    float aspect_ratio = (float)W/H;

    int blocks_x = (W + threads_per_block - 1) / threads_per_block;
    int blocks_y = (H + threads_per_block - 1) / threads_per_block;

    dim3 blocks(blocks_x, blocks_y, 1);
    dim3 threads(threads_per_block, threads_per_block);

    //warm up gpu, pay initialization costs
    for(int i = 0; i < 5; i++) {
        ray_trace<<<blocks, threads>>>(scene, d_image, camera, aspect_ratio, H, W, num_spheres);
    }

    kernelTimer(
        [&](){ray_trace<<<blocks, threads>>>(scene, d_image, camera, aspect_ratio, H, W, num_spheres);},
        "basicRayTracingBenchmark",
        W,
        H,
        20
    );        


    cudaErrCheck(cudaMemcpy(image, d_image, H * W * 3 * sizeof(unsigned char), cudaMemcpyDeviceToHost));

    // Write a PPM image file (a very simple image format).
    FILE* f = fopen("image.ppm", "wb");
    fprintf(f, "P6\n%d %d\n255\n", W, H);
    fwrite(image, 1, H * W * 3 * sizeof(unsigned char), f);
    fclose(f);
    printf("Wrote image.ppm (%dx%d)\n", W, H);

    free(image);
    cudaFree(d_image);
    cudaFree(scene);

}