#include <iostream>
#include <vector>
#include <numeric>
#include <cuda_runtime.h>
#include <chrono>

/**
 * MACRO for error checking. 
 * Essential for any production-grade or credible GPU code to catch asynchronous errors.
 */
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
 * 1. Naive Reduction Kernel
 * 
 * Each thread performs a global memory atomic add.
 * Bottleneck: Massive contention on a single memory location.
 * Memory access: Atomic contention (Serialization).
 */
__global__ void reduceNaive(const float* input, float* output, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        atomicAdd(output, input[tid]);
    }
}

/**
 * 2. Optimized Reduction Kernel (Shared Memory + Tree Reduction)
 * 
 * Improvements:
 * - Coalesced load from global memory to shared memory.
 * - In-place reduction in shared memory (tree structure).
 * - Minimal global memory writes (one atomic per block instead of one per thread).
 * 
 * Tradeoff: Uses shared memory (limited resource) to gain high throughput.
 */
__global__ void reduceOptimized(const float* input, float* output, int n) {
    extern __shared__ float sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Load data into shared memory
    sdata[tid] = (i < n) ? input[i] : 0.0f;
    __syncthreads();

    // Tree-based reduction in shared memory
    // Reduces O(N) to O(log N) operations within the block
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // Single atomic write per block
    if (tid == 0) {
        atomicAdd(output, sdata[0]);
    }
}

/**
 * Benchmark Helper
 * Uses CUDA Events for accurate timing, bypassing host-side measurement noise.
 */
float runBenchmark(void (*kernel)(const float*, float*, int), 
                   const float* d_input, float* d_output, int n, 
                   int threadsPerBlock, int blocksPerGrid, bool isOptimized) {
    
    float zero = 0.0f;
    CHECK_CUDA(cudaMemcpy(d_output, &zero, sizeof(float), cudaMemcpyHostToDevice));

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    size_t sharedMemSize = isOptimized ? (threadsPerBlock * sizeof(float)) : 0;

    CHECK_CUDA(cudaEventRecord(start));
    if (isOptimized) {
        kernel<<<blocksPerGrid, threadsPerBlock, sharedMemSize>>>(d_input, d_output, n);
    } else {
        kernel<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output, n);
    }
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float milliseconds = 0;
    CHECK_CUDA(cudaEventElapsedTime(&milliseconds, start, stop));

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    return milliseconds;
}

int main(int argc, char** argv) {
    int n = 1 << 24; // ~16 Million elements for clearer delta
    if (argc > 1) n = atoi(argv[1]);

    size_t size = n * sizeof(float);
    std::vector<float> h_input(n, 1.0f);
    float expected_sum = (float)n;

    float *d_input, *d_output;
    CHECK_CUDA(cudaMalloc(&d_input, size));
    CHECK_CUDA(cudaMalloc(&d_output, sizeof(float)));

    CHECK_CUDA(cudaMemcpy(d_input, h_input.data(), size, cudaMemcpyHostToDevice));

    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;

    std::cout << "Benchmarking Reduction | N = " << n << " | Threads/Block = " << threadsPerBlock << std::endl;

    // Warm-up run
    runBenchmark(reduceOptimized, d_input, d_output, n, threadsPerBlock, blocksPerGrid, true);

    // Naive
    float timeNaive = runBenchmark(reduceNaive, d_input, d_output, n, threadsPerBlock, blocksPerGrid, false);
    float resultNaive = 0;
    CHECK_CUDA(cudaMemcpy(&resultNaive, d_output, sizeof(float), cudaMemcpyDeviceToHost));

    // Optimized
    float timeOptimized = runBenchmark(reduceOptimized, d_input, d_output, n, threadsPerBlock, blocksPerGrid, true);
    float resultOptimized = 0;
    CHECK_CUDA(cudaMemcpy(&resultOptimized, d_output, sizeof(float), cudaMemcpyDeviceToHost));

    std::cout << "\n--- Performance Results ---" << std::endl;
    printf("Naive:     %8.4f ms | Sum: %f %s\n", timeNaive, resultNaive, (std::abs(resultNaive - expected_sum) < 1.0f ? "(OK)" : "(FAIL)"));
    printf("Optimized: %8.4f ms | Sum: %f %s\n", timeOptimized, resultOptimized, (std::abs(resultOptimized - expected_sum) < 1.0f ? "(OK)" : "(FAIL)"));
    
    float speedup = timeNaive / timeOptimized;
    printf("Speedup:   %8.2fx\n", speedup);

    // Print machine-readable results for structured parsing
    std::cout << "\nMETRICS_START" << std::endl;
    std::cout << "baseline_ms=" << timeNaive << std::endl;
    std::cout << "optimized_ms=" << timeOptimized << std::endl;
    std::cout << "speedup=" << speedup << std::endl;
    std::cout << "correct=" << (std::abs(resultOptimized - expected_sum) < 1.0f ? "true" : "false") << std::endl;
    std::cout << "METRICS_END" << std::endl;

    // Discussion points for educational purposes:
    // - Memory bandwidth utilization
    // - Occupancy vs latency hiding
    // - Warp divergence impact in tree reduction
    
    // Structured JSON for automation (compatibility)
    std::cout << "\nRESULTS_JSON: {\"n\": " << n << ", \"naive_ms\": " << timeNaive 
              << ", \"optimized_ms\": " << timeOptimized << ", \"speedup\": " << speedup << "}" << std::endl;

    CHECK_CUDA(cudaFree(d_input));
    CHECK_CUDA(cudaFree(d_output));

    return 0;
}
