#include <iostream>
#include <cuda_runtime.h>

#define BLOCK_SIZE 256

__global__ void parallelAddition(int *input, int *result, int N) {
    __shared__ int sdata[BLOCK_SIZE];

    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x = threadIdx.x;

    sdata[tid] = (i < N) ? input[i] : 0;
    __syncthreads(); 

    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            sdata[tid] += sdata[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        result[blockIdx.x] = sdata[0];
    }
}

int main() {
    int N = 1024; 
    int *h_input = new int[N];
    int *h_result = new int[N / BLOCK_SIZE]; 

    for (int i = 0; i < N; i++) {
        h_input[i] = 1;
    }

    int *d_input, int *d_result;
    cudaMalloc(&d_input, N * sizeof(int));
    cudaMalloc(&d_result, (N / BLOCK_SIZE ) * sizeof(int));

    cudaMemcpy(d_input, h_input, N * sizeof(int), cudaMemcpyHostToDevice);

    int gridSize = N / BLOCK_SIZE;
    parallelSum<<gridSize, BLOCK_SIZE>>>(d_input, d_result, N);

    cudaMemcpy(h_result, d_result, (N / BLOCK_SIZE ) * sizeof(int), cudaMemcpyDeviceToHost); 
    int finalsum = 0;
    for (int i = 0; i < gridSize; i++) {
        finalsum += h_result[i];
    }
    std::cout << "Final Sum: " << finalsum << std::endl;
}
