#include <stdio.h> // Includes the standard I/O library for the printf function

// Kernel function to be executed on the GPU
__global__ void HelloFromGPU(void) {
  // Each thread prints this message when the kernel is executed
  printf("Hello World from GPU! \n");
}

int main(void) {
  // Print a message from the CPU (host)
  printf("Hello World from CPU! \n");

  // Launch the GPU kernel with 1 block and 10 threads
  // <<<1, 10>>> specifies the kernel configuration:
  // - 1: Number of blocks
  // - 10: Number of threads per block
  HelloFromGPU <<<1, 10>>>();

  // Reset the GPU to ensure all resources are released
  cudaDeviceReset();

  // Indicate successful program termination
  return 0;
}
