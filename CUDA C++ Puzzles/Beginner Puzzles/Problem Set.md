| Day | Puzzle Title                       | Description                                                                                                                                 |
|-----|------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------|
| 1   | Hello, Thread!                     | Launch a kernel where each thread prints "Hello from thread X". Learn to use `threadIdx.x`, `blockIdx.x`, and `blockDim.x`.                 |
| 2   | Global ID Calculation              | Implement a kernel that writes each thread's global ID into an array.                                                                      |
| 3   | Vector Add (GPU)                   | Add two float arrays on the GPU. Launch one thread per element. Verify result on host.                                                     |
| 4   | Grid of Threads                    | Modify the vector add to use multiple blocks. Print `(blockIdx.x, threadIdx.x)` in the kernel.                                              |
| 5   | Scalar Multiply                    | Multiply each element of a float array by a scalar on the GPU. Test for arrays of size 1024+.                                              |
| 6   | Square Each Element                | Square each element in an input array (element-wise `x[i] *= x[i]`) using the GPU.                                                         |
| 7   | Host vs Device Timing              | Use `clock()` or timers in your playground to measure the time taken by a GPU kernel vs a CPU loop.                                        |
| 8   | Allocate on Device                 | Allocate a device array using `cudaMalloc`, fill it in a kernel, and copy it back with `cudaMemcpy`.                                      |
| 9   | First Shared Memory Use            | Load a 1D tile (e.g., 8 elements) into `__shared__` memory, modify it, then write it back to global memory.                                |
| 10  | Thread Mapping Debug               | Create a 2D grid and 2D block launch. Print each thread’s global coordinates (row, col).                                                   |
