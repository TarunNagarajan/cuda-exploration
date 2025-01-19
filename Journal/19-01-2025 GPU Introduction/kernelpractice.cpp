#include <stdio.h>
#include <cuda_runtime.h> // Header for CUDA runtime functions

// GPU kernel function to process an array by multiplying each element by 2
__global__ void ProcessArray(int *d_array, int size) {
    // Calculate the global thread index
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    // Ensure the thread index does not exceed the size of the array
    if (idx < size) {
        d_array[idx] *= 2; // Multiply the element at index `idx` by 2
        printf("Thread %d processed element: %d\n", idx, d_array[idx]); // Debug info
    }
}

int main(void) {
    // Print a message from the CPU
    printf("Hello World from CPU!\n");

    const int arraySize = 10; // Define the size of the array
    int h_array[arraySize];   // Declare the host array (on CPU)

    // Initialize the host array with values from 1 to 10
    for (int i = 0; i < arraySize; ++i) {
        h_array[i] = i + 1;
    }

    int *d_array;            // Pointer to the array on the GPU (device)
    size_t size = arraySize * sizeof(int); // Calculate the size of the array in bytes

    // Allocate memory on the GPU for the array
    cudaMalloc((void **)&d_array, size);

    // Copy the contents of the host array (CPU) to the device array (GPU)
    cudaMemcpy(d_array, h_array, size, cudaMemcpyHostToDevice);

    // Configure the kernel launch parameters
    int threadsPerBlock = 5; // Number of threads in each block
    int blocksPerGrid = (arraySize + threadsPerBlock - 1) / threadsPerBlock;
    // The formula ensures enough blocks to cover all elements

    // Create CUDA events for measuring GPU execution time
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Record the start time
    cudaEventRecord(start);

    // Launch the kernel with the specified grid and block dimensions
    ProcessArray<<<blocksPerGrid, threadsPerBlock>>>(d_array, arraySize);

    // Record the stop time
    cudaEventRecord(stop);

    // Copy the processed data from the device array (GPU) back to the host array (CPU)
    cudaMemcpy(h_array, d_array, size, cudaMemcpyDeviceToHost);

    // Wait for the GPU to finish all operations before measuring elapsed time
    cudaEventSynchronize(stop);

    // Calculate the elapsed time between `start` and `stop`
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Time taken by kernel: %f ms\n", milliseconds);

    // Print the processed array to verify the results
    printf("Processed array: ");
    for (int i = 0; i < arraySize; ++i) {
        printf("%d ", h_array[i]);
    }
    printf("\n");

    // Free the memory allocated on the GPU
    cudaFree(d_array);

    // Reset the GPU device to release resources
    cudaDeviceReset();

    return 0; // Exit the program
}
