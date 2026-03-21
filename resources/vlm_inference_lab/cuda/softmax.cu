#include <iostream>
#include <cuda_runtime.h>
#include <math.h>

#define CHECK_CUDA(call)                                                 \
    do {                                                                 \
        cudaError_t err = call;                                          \
        if (err != cudaSuccess) {                                        \
            std::cerr << "CUDA Error: " << cudaGetErrorString(err)       \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
            exit(EXIT_FAILURE);                                          \
        }                                                                \
    } while (0)

/**
 * Simplified Softmax Kernel (1D)
 * 
 * Performs: exp(x_i - max(X)) / sum(exp(x_j - max(X)))
 * Using shared memory for block-level reduction of max and sum.
 * 
 * Implementation Notes:
 * 1. Numerical stability: Subtracting max prevents overflow in expf().
 * 2. Tree-based reduction in shared memory for fast parallel sum/max.
 * 3. Thread mapping: Each thread handles multiple elements (grid-stride loop) 
 *    if N > blockDim.x.
 */
__global__ void softmaxSimple(const float* input, float* output, int N) {
    extern __shared__ float s_data[];

    int tid = threadIdx.x;

    // 1. Find Max for numerical stability
    float local_max = -1e20f;
    for (int i = tid; i < N; i += blockDim.x) {
        local_max = fmaxf(local_max, input[i]);
    }
    s_data[tid] = local_max;
    __syncthreads();

    // Reduce within block for max
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) s_data[tid] = fmaxf(s_data[tid], s_data[tid + s]);
        __syncthreads();
    }
    float max_val = s_data[0];
    __syncthreads();

    // 2. Compute Sum of exponentials
    float local_sum = 0.0f;
    for (int i = tid; i < N; i += blockDim.x) {
        local_sum += expf(input[i] - max_val);
    }
    s_data[tid] = local_sum;
    __syncthreads();

    // Reduce within block for sum
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) s_data[tid] += s_data[tid + s];
        __syncthreads();
    }
    float total_sum = s_data[0];
    __syncthreads();

    // 3. Normalize
    for (int i = tid; i < N; i += blockDim.x) {
        output[i] = expf(input[i] - max_val) / total_sum;
    }
}

int main() {
    int N = 1024;
    size_t size = N * sizeof(float);

    float *h_input = (float*)malloc(size);
    float *h_output = (float*)malloc(size);

    for (int i = 0; i < N; i++) h_input[i] = (float)i / 100.0f;

    float *d_input, *d_output;
    CHECK_CUDA(cudaMalloc(&d_input, size));
    CHECK_CUDA(cudaMalloc(&d_output, size));

    CHECK_CUDA(cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice));

    softmaxSimple<<<1, 256, 256 * sizeof(float)>>>(d_input, d_output, N);
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost));

    float sum = 0.0f;
    for (int i = 0; i < N; i++) sum += h_output[i];
    std::cout << "Softmax total sum: " << sum << " (Expected: ~1.0)" << std::endl;

    cudaFree(d_input);
    cudaFree(d_output);
    free(h_input);
    free(h_output);

    return 0;
}
