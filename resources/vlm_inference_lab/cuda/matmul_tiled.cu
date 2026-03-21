#include <iostream>
#include <cuda_runtime.h>

#define CHECK_CUDA(call)                                                 \
    do {                                                                 \
        cudaError_t err = call;                                          \
        if (err != cudaSuccess) {                                        \
            std::cerr << "CUDA Error: " << cudaGetErrorString(err)       \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            exit(EXIT_FAILURE);                                          \
        }                                                                \
    } while (0)

#define TILE_WIDTH 16

/**
 * Tiled Matrix Multiplication Kernel
 * 
 * Computes C = A * B.
 * Using shared memory tiling to reduce global memory bandwidth requirements.
 * 
 * Implementation Notes:
 * 1. Data Reuse: Each element in A and B tiles is loaded once into shared memory 
 *    and reused by all threads in the block (TILE_WIDTH times).
 * 2. Shared Memory: ds_A and ds_B store tiles of the input matrices.
 * 3. Synchronization: __syncthreads() ensures all threads in a block have 
 *    finished loading the tile before computation starts, and finished 
 *    computation before the next tile is loaded.
 */
__global__ void matMulTiled(const float* A, const float* B, float* C, int width) {
    __shared__ float ds_A[TILE_WIDTH][TILE_WIDTH];
    __shared__ float ds_B[TILE_WIDTH][TILE_WIDTH];

    int bx = blockIdx.x;  int by = blockIdx.y;
    int tx = threadIdx.x; int ty = threadIdx.y;

    int row = by * TILE_WIDTH + ty;
    int col = bx * TILE_WIDTH + tx;

    float pValue = 0;

    // Loop over the tiles of A and B required to compute the C element
    for (int p = 0; p < (width + TILE_WIDTH - 1) / TILE_WIDTH; ++p) {
        
        // Collaborative loading of A and B tiles into shared memory
        if (row < width && p * TILE_WIDTH + tx < width)
            ds_A[ty][tx] = A[row * width + p * TILE_WIDTH + tx];
        else
            ds_A[ty][tx] = 0.0;

        if (p * TILE_WIDTH + ty < width && col < width)
            ds_B[ty][tx] = B[(p * TILE_WIDTH + ty) * width + col];
        else
            ds_B[ty][tx] = 0.0;

        __syncthreads();

        for (int i = 0; i < TILE_WIDTH; ++i)
            pValue += ds_A[ty][i] * ds_B[i][tx];

        __syncthreads();
    }

    if (row < width && col < width)
        C[row * width + col] = pValue;
}

int main() {
    int width = 512;
    size_t size = width * width * sizeof(float);

    float *h_A = (float*)malloc(size);
    float *h_B = (float*)malloc(size);
    float *h_C = (float*)malloc(size);

    for (int i = 0; i < width * width; i++) {
        h_A[i] = 1.0f;
        h_B[i] = 2.0f;
    }

    float *d_A, *d_B, *d_C;
    CHECK_CUDA(cudaMalloc(&d_A, size));
    CHECK_CUDA(cudaMalloc(&d_B, size));
    CHECK_CUDA(cudaMalloc(&d_C, size));

    CHECK_CUDA(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice));

    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);
    dim3 dimGrid((width + TILE_WIDTH - 1) / TILE_WIDTH, (width + TILE_WIDTH - 1) / TILE_WIDTH);

    matMulTiled<<<dimGrid, dimBlock>>>(d_A, d_B, d_C, width);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost));

    // Verification: Each element should be 1.0 * 2.0 * width = 2 * 512 = 1024
    if (h_C[0] == 1024.0f) {
        std::cout << "Tiled MatMul successful!" << std::endl;
    } else {
        std::cerr << "Verification failed: h_C[0] = " << h_C[0] << std::endl;
    }

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}
