#include <iostream>
#include <cuda_runtime_h>

__global__ void vectorAdd(int *A, int *B, int *C, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        C[i] = A[i] + B[i];
    }
}

int main() {
    int N = 1000;
    size_t size = N * sizeof(int);

    int *h_A = new int[N];
    int *h_B = new int[N];
    int *h_C = new int[N];

    for (int i = 0; i < N; i++) {
        h_A[i] = i;
        h_B[i] = i * 2;
    }

    int *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size); 
    cudaMalloc(&d_C, size);

    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);

    int blocksize = 256;
    int gridsize = (N + blocksize - 1) / blocksize;

    vectorAdd<<<gridsize, blocksize>>>(d_A, d_B, d_C, N); 
    cudaMemcpy(h_C, d_c, size, cudaMemcpyDeviceToHost);

    for (int i = 0; i < 10; i++) {
        std::cout << "C[" << i << "] = " << h_C << std::endl;
    }

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    delete[] h_A;
    delete[] h_B;
    delete[] h_C;

    return 0;
}
