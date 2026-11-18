#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

#define NSTREAMS 4

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA Error en %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

__global__ void vectorAddKernel(const float *A, const float *B, float *C, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        C[idx] = A[idx] + B[idx];
    }
}

int main(void) {
    const int N = 16 * 1024 * 1024; // 16M elementos
    const size_t totalBytes = static_cast<size_t>(N) * sizeof(float);

    float *h_A = NULL, *h_B = NULL, *h_C_sync = NULL, *h_C_async = NULL;
    CUDA_CHECK(cudaMallocHost((void **)&h_A, totalBytes));
    CUDA_CHECK(cudaMallocHost((void **)&h_B, totalBytes));
    CUDA_CHECK(cudaMallocHost((void **)&h_C_sync, totalBytes));
    CUDA_CHECK(cudaMallocHost((void **)&h_C_async, totalBytes));

    for (int i = 0; i < N; ++i) {
        h_A[i] = static_cast<float>(i % 100) * 0.01f;
        h_B[i] = static_cast<float>((2 * i) % 100) * 0.02f;
    }

    float *d_A = NULL, *d_B = NULL, *d_C = NULL;
    CUDA_CHECK(cudaMalloc((void **)&d_A, totalBytes));
    CUDA_CHECK(cudaMalloc((void **)&d_B, totalBytes));
    CUDA_CHECK(cudaMalloc((void **)&d_C, totalBytes));

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    const int threadsPerBlock = 256;

    // Warm-up
    int totalBlocks = (N + threadsPerBlock - 1) / threadsPerBlock;
    CUDA_CHECK(cudaMemcpy(d_A, h_A, totalBytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, totalBytes, cudaMemcpyHostToDevice));
    vectorAddKernel<<<totalBlocks, threadsPerBlock>>>(d_A, d_B, d_C, N);
    CUDA_CHECK(cudaMemcpy(h_C_sync, d_C, totalBytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaDeviceSynchronize());

    // 1. Baseline Sincrono
    float sync_ms = 0.0f;
    cudaEventRecord(start, 0);
    CUDA_CHECK(cudaMemcpy(d_A, h_A, totalBytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, totalBytes, cudaMemcpyHostToDevice));
    vectorAddKernel<<<totalBlocks, threadsPerBlock>>>(d_A, d_B, d_C, N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(h_C_sync, d_C, totalBytes, cudaMemcpyDeviceToHost));
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&sync_ms, start, stop);
    // 2. Asincrono con CUDA Streams
    cudaStream_t streams[NSTREAMS];
    for (int s = 0; s < NSTREAMS; ++s) {
        CUDA_CHECK(cudaStreamCreate(&streams[s]));
    }

    const int chunkSize = N / NSTREAMS;
    const size_t chunkBytes = static_cast<size_t>(chunkSize) * sizeof(float);
    const int chunkBlocks = (chunkSize + threadsPerBlock - 1) / threadsPerBlock;

    float async_ms = 0.0f;
    cudaEventRecord(start, 0);
    for (int s = 0; s < NSTREAMS; ++s) {
        int offset = s * chunkSize;
        CUDA_CHECK(cudaMemcpyAsync(d_A + offset, h_A + offset, chunkBytes, cudaMemcpyHostToDevice, streams[s ]));
        CUDA_CHECK(cudaMemcpyAsync(d_B + offset, h_B + offset, chunkBytes, cudaMemcpyHostToDevice, streams[s]));
        vectorAddKernel<<<chunkBlocks, threadsPerBlock, 0, streams[s]>>>(d_A + offset, d_B + offset, d_C + offset, chunkSize);
        CUDA_CHECK(cudaMemcpyAsync(h_C_async + offset, d_C + offset, chunkBytes, cudaMemcpyDeviceToHost, streams[s]));
    }
    for (int s = 0; s < NSTREAMS; ++s) {
        CUDA_CHECK(cudaStreamSynchronize(streams[s])));
    }
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&async_ms, start, stop);

    int errors = 0;
    for (int i = 0; i < N; ++i) {
        float expected = h_A[i] + h_B[i];
        if (fabsf(h_C_async[i] - expected) > 1e-5f || fabsf(h_C_sync[i] - expected) > 1e-5f) {
            errors++;
        }
    }

    float reduction_pct = ((sync_ms - async_ms) / sync_ms) * 100.0f;
    printf("=== CHECKPOINT 4: Transferencia Asincrona con Streams ===\n");
    printf("Sincrono (H2D + kernel + D2H completos): %.2f ms (Errores: %d)\n", sync_ms, errors);
    printf("Con streams (%d, solapado)             : %.2f ms (Errores: %d)\n", NSTREAMS, async_ms, errors);
    printf("Reduccion                             : %.1f%%\n", reduction_pct);

    for (int s = 0; s < NSTREAMS; ++s) {
        cudaStreamDestroy(streams[s]);
    }
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaFreeHost(h_A);
    cudaFreeHost(h_B);
    cudaFreeHost(h_C_sync);
    cudaFreeHost(h_C_async);

    return 0;
}
