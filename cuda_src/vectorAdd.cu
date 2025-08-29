#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <string>

__global__ void vadd(const float* __restrict__ a,
                     const float* __restrict__ b,
                     float* __restrict__ c,
                     size_t n) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}

int main(int argc, char** argv) {
    size_t n = (argc > 1) ? std::stoull(std::string(argv[1])) : (1ull << 24); // ~16M
    size_t bytes = n * sizeof(float);

    float *ha = (float*)std::malloc(bytes);
    float *hb = (float*)std::malloc(bytes);
    float *hc = (float*)std::malloc(bytes);
    if (!ha || !hb || !hc) { std::fprintf(stderr, "Host alloc failed\n"); return 1; }
    for (size_t i = 0; i < n; ++i) { ha[i] = float(i) * 0.001f; hb[i] = 1.0f; }

    float *da = nullptr, *db = nullptr, *dc = nullptr;
    cudaMalloc(&da, bytes);
    cudaMalloc(&db, bytes);
    cudaMalloc(&dc, bytes);

    cudaMemcpy(da, ha, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(db, hb, bytes, cudaMemcpyHostToDevice);

    int block = 256;
    int grid  = int((n + block - 1) / block);

    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);
    cudaEventRecord(start);
    vadd<<<grid, block>>>(da, db, dc, n);
    cudaEventRecord(stop); cudaEventSynchronize(stop);

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);

    cudaMemcpy(hc, dc, bytes, cudaMemcpyDeviceToHost);

    // Verify a few elements
    for (size_t i = 0; i < 10; ++i) {
        float ref = ha[i] + hb[i];
        if (std::fabs(hc[i] - ref) > 1e-4f) {
            std::fprintf(stderr, "Mismatch at %zu: got %f, want %f\n", i, hc[i], ref);
            break;
        }
    }

    double gb   = 3.0 * double(bytes) / 1e9; // read A,B + write C
    double gbps = gb / (ms / 1e3);
    std::printf("CUDA vadd: N=%zu time=%.3f ms, BW=%.2f GB/s\n", n, ms, gbps);

    cudaFree(da); cudaFree(db); cudaFree(dc);
    std::free(ha); std::free(hb); std::free(hc);
    return 0;
}
