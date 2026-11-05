// Isolated variant; see VARIANTS.md.
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <cuda.h>
#include <cuda_runtime.h>
#include <fstream>

#define MISS -1
#define bounces 5
#define eps 0.001f
#define min_throughput 0.001f
#define leafSize 4
#define STACK_SIZE 16

__constant__ int c_floorIndex;

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
    int nsph,
    int measureRounds = 10
) {

    if(measureRounds <= 0) {
        std::cout << "number of rounds cannot be zero or negative: " << measureRounds << '\n';
        return;
    }

    std::string fTxt = fileName + ".txt";
    std::ofstream output(fTxt, std::ios::app);

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

        output << "Average kernel time(ms): " << ms / (float)measureRounds << ", " << W << "x" << H << " resolution, number of spheres: " << nsph << "\n";

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

__host__ Vec3 minVec(const Vec3& v1, const Vec3& v2) {
    float x = fmin(v1.x, v2.x);
    float y = fmin(v1.y, v2.y);
    float z = fmin(v1.z, v2.z);

    return Vec3(x, y, z);
}

__host__ Vec3 maxVec(const Vec3& v1, const Vec3& v2) {
    float x = fmax(v1.x, v2.x);
    float y = fmax(v1.y, v2.y);
    float z = fmax(v1.z, v2.z);

    return Vec3(x, y, z);
}

struct Sphere {
    Vec3 center;
    float radius;
    Vec3 colour;
    float reflectivity;
    bool isFloor;
};

struct AABB { //axis aligned, easy to deal with 
    Vec3 minPoint;
    Vec3 maxPoint;

    AABB() {}

    AABB(Vec3 minP, Vec3 maxP):
        minPoint(minP), maxPoint(maxP) {}

};

__host__ AABB sphereAABB(const Sphere& sph) {
    float r = sph.radius;
    Vec3 radiusCube(r, r, r);
    return AABB(sph.center - radiusCube, sph.center + radiusCube);
}

__host__ AABB merge(const AABB& box1, const AABB& box2) {
    Vec3 minP = minVec(box1.minPoint, box2.minPoint);
    Vec3 maxP = maxVec(box1.maxPoint, box2.maxPoint);

    return AABB(minP, maxP);
}

struct BVHNode {
    AABB box;

    //children
    int left;
    int right;

    //leaf
    int start;
    int count;
};

__host__ int buildTree(const Sphere* scene, std::vector<BVHNode>& tree, std::vector<int>& indices, int start, int end) {
    int idx = tree.size();
    tree.emplace_back();

    if(end - start <= leafSize) {
        AABB box = sphereAABB(scene[indices[start]]);
        for(int i = start+1; i < end; i++) {
            box = merge(box, sphereAABB(scene[indices[i]]));
        }
        tree[idx].box = box;
        tree[idx].left = -1;
        tree[idx].right = -1;
        tree[idx].start = start;
        tree[idx].count = end - start;

        return idx;
    }

    Vec3 first = scene[indices[start]].center;

    float minX = first.x, maxX = first.x;
    float minY = first.y, maxY = first.y;
    float minZ = first.z, maxZ = first.z;

    int axis = 0;

    for(int i = start+1; i < end; i++) {
        const Vec3 c = scene[indices[i]].center;
        minX = fmin(minX, c.x);
        maxX = fmax(maxX, c.x);
        minY = fmin(minY, c.y);
        maxY = fmax(maxY, c.y);
        minZ = fmin(minZ, c.z);
        maxZ = fmax(maxZ, c.z);
    }

    float maxExtent = std::max({maxX - minX, maxY - minY, maxZ - minZ});

    if(maxExtent == maxX - minX) {
        axis = 0;
    }
    else if(maxExtent == maxY - minY) {
        axis = 1;
    }
    else {
        axis = 2;
    }

    int mid = (start + end) / 2;

    std::nth_element(
        indices.begin() + start,
        indices.begin() + mid,
        indices.begin() + end,
        [&](int a, int b) {
            if(axis == 0) return scene[a].center.x < scene[b].center.x;
            else if(axis == 1) return scene[a].center.y < scene[b].center.y;
            else return scene[a].center.z < scene[b].center.z;
        }
    );

    int left = buildTree(scene, tree, indices, start, mid);
    int right = buildTree(scene, tree, indices, mid, end);

    tree[idx].box = merge(tree[left].box, tree[right].box);
    tree[idx].left = left;
    tree[idx].right = right;
    tree[idx].start = start;
    tree[idx].count = 0;

    return idx;
}

__device__ bool intersectAxis(float axisCamera, float axisDirn, float minCoord, float maxCoord, float& tClosest, float& tFarthest) {
    if(axisDirn == 0.0f) { //parallel to bounding box
        return (minCoord <= axisCamera) && (axisCamera <= maxCoord); //inside box?
    }

    float t1 = (minCoord - axisCamera) / axisDirn;
    float t2 = (maxCoord - axisCamera) / axisDirn;

    if(t1 > t2) {
        float tmpT = t1;
        t1 = t2;
        t2 = tmpT;
    }

    tClosest = fmaxf(tClosest, t1);
    tFarthest = fminf(tFarthest, t2);

    return tClosest <= tFarthest;
}

__device__ bool intersectAABB(const AABB& box, Vec3 camera, Vec3 dirn, float& tClosest, float& tFarthest) {
    if(!intersectAxis(camera.x, dirn.x, box.minPoint.x, box.maxPoint.x, tClosest, tFarthest)) return false;
    if(!intersectAxis(camera.y, dirn.y, box.minPoint.y, box.maxPoint.y, tClosest, tFarthest)) return false;
    if(!intersectAxis(camera.z, dirn.z, box.minPoint.z, box.maxPoint.z, tClosest, tFarthest)) return false;
    return true;
}

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

__device__ int rayCollisionPoint(const Sphere* scene, Vec3 camera, Vec3 dirn, float& tClosest, int num_spheres, int* indices, const int root, const BVHNode* tree) {
    int best = -1;
    float closest = 1e30f;


    if(c_floorIndex >= 0) {
        float t = intersectSphere(
            scene[c_floorIndex], camera, dirn
        );

        if(t > eps && t < closest) {
            closest = t;
            best = c_floorIndex;
        }
    }

    if(root == -1) { // No objects
        tClosest = closest;
        return best;
    }

    
    // DFS through the tree.
    int stackBVH[STACK_SIZE];
    int i = 0;
    stackBVH[i++] = root;
    while(i > 0) {
        int top = stackBVH[--i];
        float tEntry = eps, tExit = closest;
        if(!intersectAABB(tree[top].box, camera, dirn, tEntry, tExit)) continue;
        if(tree[top].count > 0) { //leaf
                //check every sphere in leaf range
                int start = tree[top].start;
                int end = start + tree[top].count;

                for(int i = start; i < end; i++) {
                    float t = intersectSphere(scene[indices[i]], camera, dirn);
                    if(t > eps && t < closest) {
                        closest = t;
                        best = indices[i];
                    }
                    
                }
        }
        else { //internal node
            stackBVH[i++] = tree[top].right;
            stackBVH[i++] = tree[top].left;
        }
    }
    tClosest = closest;
    return best;
}

__device__ bool inShadow(const Sphere* scene, Vec3 camera, Vec3 light, int num_spheres, int* indices, const int root, const BVHNode* tree) {
    Vec3 toLight = light - camera;
    float maxDistance = sqrtf(dot(toLight, toLight));
    Vec3 dirn = normalize(toLight);
    

    if(c_floorIndex >= 0) {
        float t = intersectSphere(
            scene[c_floorIndex], camera, dirn
        );

        if(t > eps && t < maxDistance) {
            return true;
        }
    }

    if(root == -1) { // No objects
        return false;
    }

    int stackBVH[STACK_SIZE];
    int i = 0;
    stackBVH[i++] = root;
    while(i > 0) {
        int top = stackBVH[--i];
        float tEntry = eps, tExit = maxDistance;
        if(!intersectAABB(tree[top].box, camera, dirn, tEntry, tExit)) continue;
        if(tree[top].count > 0) { //leaf
                //check every sphere in leaf range
                int start = tree[top].start;
                int end = start + tree[top].count;

                for(int i = start; i < end; i++) {
                    float t = intersectSphere(scene[indices[i]], camera, dirn);
                    if(t > eps && t < maxDistance) {
                        return true;
                    }
                    
                }
        }
        else { //internal node
            stackBVH[i++] = tree[top].right;
            stackBVH[i++] = tree[top].left;
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

__device__ float computeLighting(const Sphere* scene, Vec3 p_int, Vec3 s_normal, Vec3 light, int num_spheres, int* indices, const int root, const BVHNode* tree) {
    Vec3 toLight = normalize(light - p_int);
    float diffuse_light = fmaxf(0.0f, dot(s_normal, toLight));
    float ambient_light = 0.15f;
    if(diffuse_light == 0) return ambient_light;

    // send out a ray towards the light, if it intersects a sphere before the light, it is in shadow
    bool blocked = inShadow(scene, p_int + (s_normal * eps), light, num_spheres, indices, root, tree);
    float shade =  blocked ? 0.3f : 1.0f;

    return ambient_light + (diffuse_light * shade);
}

__device__ Vec3 accumulatedColour(const Sphere* scene, Vec3 light, Vec3 camera, Vec3 dirn, int num_spheres, int firstCollision, float tClosest, int* indices, const int root, const BVHNode* tree) {
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
        brightness = computeLighting(scene, p_int, s_normal, light, num_spheres, indices, root, tree);
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
        int sphereIndex = rayCollisionPoint(scene, p + n * eps, refl_dir, tC, num_spheres, indices, root, tree);
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
            brightness = computeLighting(scene, p, n, light, num_spheres, indices, root, tree);
        }
        
        reflColour = reflColour + residue * brightness * (1 - sph.reflectivity) * sphereColour;
        residue = residue * sph.reflectivity;
    }
    
    return base + reflColour;
}

__device__ Vec3 rayColour(const Sphere* scene, Vec3 camera, Vec3 dirn, int num_spheres, int* indices, const int root, const BVHNode* tree) {
    // x = right, y = up, z = toward the camera.
    Vec3 light(4, 6, 2);
    float tClosest;
    int firstCollision = rayCollisionPoint(scene, camera, dirn, tClosest, num_spheres, indices, root, tree);

    if(firstCollision < 0) {
        return getSkyGradient(dirn.y);
    }
    return accumulatedColour(scene, light, camera, dirn, num_spheres, firstCollision, tClosest, indices, root, tree);

}

__global__ void ray_trace(const Sphere* scene, unsigned char* img, Vec3 camera, const float aspect_ratio, const int H, const int W, const int num_spheres, int* indices, const int root, const BVHNode* tree) {
    int py = blockIdx.y * blockDim.y + threadIdx.y;
    int px = blockIdx.x * blockDim.x + threadIdx.x;

    if(py >= H || px >= W) return;

    float u = (2.0f * (px + 0.5f) / W - 1.0f) * aspect_ratio;
    float v = (1.0f - 2.0f * (py + 0.5f) / H);
    Vec3 ray_dir = normalize(Vec3(u, v, -1.5f));

    Vec3 colour = rayColour(scene, camera, ray_dir, num_spheres, indices, root, tree);

    int rgb_idx = (py * W + px) * 3;
    img[rgb_idx] = (unsigned char)(fminf(colour.x, 1.0f) * 255);
    img[rgb_idx + 1] = (unsigned char)(fminf(colour.y, 1.0f) * 255);
    img[rgb_idx + 2] = (unsigned char)(fminf(colour.z, 1.0f) * 255);
}

int main() {
    int num_spheres = 0;
    std::cin >> num_spheres;
    Sphere h_scene[num_spheres];
    for(int i = 0; i < num_spheres; i++) {
        //center, radius, colour, reflectivity, isFloor
        Sphere sph;
        float center_x, center_y, center_z;
        float radius;
        float colour_x, colour_y, colour_z;
        float reflectivity;
        bool isFloor;

        std::cin >> center_x >> center_y >> center_z;
        std::cin >> radius;
        std::cin >> colour_x >> colour_y >> colour_z;
        std::cin >> reflectivity;
        std::cin >> isFloor;

        sph.center = Vec3(center_x, center_y, center_z);
        sph.radius = radius;
        sph.colour = Vec3(colour_x, colour_y, colour_z);
        sph.reflectivity = reflectivity;
        sph.isFloor = isFloor;

        h_scene[i] = sph;
    }

    Sphere* scene = nullptr;

    std::vector<int> indices;
    indices.reserve(num_spheres);

    int floorIndex = -1;
    for(int i = 0; i < num_spheres; i++) {
        if(h_scene[i].isFloor && floorIndex == -1) floorIndex = i;
        else indices.push_back(i);
    }
    cudaErrCheck(cudaMemcpyToSymbol(c_floorIndex, &floorIndex, sizeof(int)));

    std::vector<BVHNode> BVHTree;
    int root = -1;
    if(!indices.empty()) root = buildTree(h_scene, BVHTree, indices, 0, indices.size());

    if(num_spheres > 0) {
        cudaErrCheck(cudaMalloc((void**)&scene, num_spheres * sizeof(Sphere)));
        cudaErrCheck(cudaMemcpy(scene, h_scene, num_spheres * sizeof(Sphere), cudaMemcpyHostToDevice));
    }

    int* d_indices = nullptr;
    BVHNode* d_tree = nullptr;

    if(!indices.empty()) {
        cudaErrCheck(cudaMalloc((void**)&d_indices, sizeof(int) * indices.size()));
        cudaErrCheck(cudaMemcpy(d_indices, indices.data(), sizeof(int) * indices.size(), cudaMemcpyHostToDevice));
    }

    if(!BVHTree.empty()) {
        cudaErrCheck(cudaMalloc((void**)&d_tree, sizeof(BVHNode) * BVHTree.size()));
        cudaErrCheck(cudaMemcpy(d_tree, BVHTree.data(), sizeof(BVHNode) * BVHTree.size(), cudaMemcpyHostToDevice));    
    }

    const int W = 800;
    const int H = 600;

    unsigned char* image = (unsigned char* )malloc(H * W * 3 * sizeof(unsigned char));
    unsigned char* d_image;
    cudaErrCheck(cudaMalloc((void**)&d_image, H * W * 3 * sizeof(unsigned char)));

    Vec3 camera(0, 1, 3);
    float aspect_ratio = (float)W/H;

    dim3 threads(16, 16);

    int blocks_x = (W + threads.x - 1) / threads.x;
    int blocks_y = (H + threads.y - 1) / threads.y;

    dim3 blocks(blocks_x, blocks_y, 1);
    

    //warm up gpu, pay initialization costs
    for(int i = 0; i < 5; i++) {
        ray_trace<<<blocks, threads>>>(scene, d_image, camera, aspect_ratio, H, W, num_spheres, d_indices, root, d_tree);
    }

    kernelTimer(
        [&](){ray_trace<<<blocks, threads>>>(scene, d_image, camera, aspect_ratio, H, W, num_spheres, d_indices, root, d_tree);},
        "benchmark-bvh-separate-floor",
        W,
        H,
        num_spheres,
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
    cudaFree(d_indices);
    cudaFree(d_tree);

}