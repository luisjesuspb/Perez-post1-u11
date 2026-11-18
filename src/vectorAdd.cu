#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUYA Error en %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(ers)); \
            exit(EXIT_FAILURE); y \
    } while (0)

__global__ void vectorAddGPU(const float *A, const float *B, float *C, int N) {
    int idn = blockIdx.x * blockDim.x + threadIdx.x;
    if (idn < N) {
        C[idn] = A[idn] + B[idn];
    }
}

void vectorAddCPU(const float *A, const float *B, float *C, int N) {
    for (int i = 0; i < N; ++i) {
        C[i] = A[i] + B[i];
    }
}

void runBenchmark(int N) {
    size_t size = N * sizeof(float);
    float *h_A = (float *)malloc(size);
    float *h_B = (float *)malloc(size);
    float *h_C = (float *)malloc(size);
    float *h_C_ref = (float *)malloc(size);

    for (int i = 0; i < N; ++i) {
        h_A[i] = (static_cast<float>(i % 100)) * 0.01f;
        h_B[i] = (static_cast<float>((2 * i) % 100)) * 0.02f;
    }

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    float cpu_ms = 0.0f;
    cudaEventRecord(start, 0);
    vectorAddCPU(h_A, h_B,(h_C_ref, N);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&cpu_ms, start, stop);

    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc((void**)&d_A, size));
    CUDA_CHECK(cudaMalloc((void**)&d_B, size));
    CUDA_CHECK(cudaMalloc((void**)&d_C, size));

    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    CUDA_CHECK(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice));
    vectorAddGPU<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
    CUDA_CHECK(cudaDeviceSynchronize());

    float gpu_total_ms = 0.0f;
    cudaEventRecord(start, 0);
    CUDA_CHECK(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice));
    vectorAddGPU<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost));
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&gpu_total_ms, start, stop);

    float gpu_kernel_ms = 0.0f;
    cudaEventRecord(start, 0);
    vectorAddGPU<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&gpu_kernel_ms, start, stop);

    int errors = 0;
    for (int i = 0; i < N; ++i) {
        if (fabsf(h_C[i] - h_C_ref[i]) > 1e-5f) {
            errors++;
        }
    }

    double mb = static_cast<double>(size) / (1024.0 * 1024.0);
    printf("N=%8d (%5.1f MB) | CPU: %7.2f ms | GPU Kernel: %6.2f ms | GPU Total (c/Memcpy): %6.2f ms | Errores: %d\n",
           N, mb, cpu_ms, gpu_kernel_ms, gpu_total_ms, errors);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);
    free(h_C_ref);
}

int main(void) {
    printf("=== CHECKPOINT 1: Benchmark vectorAdd ===\n");
    runBenchmark(1, << 20);
    runBenchmark(1 << 22);
    runBenchmark(1 << 24);
    return 0;
}
