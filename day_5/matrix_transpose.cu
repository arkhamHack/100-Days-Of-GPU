#include <cuda_runtime.h>
#include <iostream>

// Define the size of the matrix
#define WIDTH 1024
#define HEIGHT 1024
#define TILE_SIZE 32

#define checkCudaError(err) { gpuAssert((err), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line) {
    if (code != cudaSuccess) {
        std::cerr << "CUDA Error: " << cudaGetErrorString(code)
                  << " in " << file << " at line " << line << std::endl;
        exit(code);
    }
}

__global__ void transpose(float* input, float* output, int width, int height){
__shared__ float tile[TILE_SIZE][TILE_SIZE + 1];
    int x = blockIdx.x * TILE_SIZE + threadIdx.x;
    int y = blockIdx.y * TILE_SIZE + threadIdx.y;

    if(x<width && y<height){
        tile[threadIdx.y][threadIdx.x]=input[y*width+x];
    }
    __syncthreads();

    x = blockIdx.y*TILE_SIZE+threadIdx.x;
    y = blockIdx.x * TILE_SIZE + threadIdx.y;

    if (x < height && y < width) {
        output[y * height + x] = tile[threadIdx.x][threadIdx.y];
    }

}

int main(){
    size_t matrix_size = WIDTH * HEIGHT * sizeof(float);
    float* h_input = (float*)malloc(matrix_size);
    float* h_output = (float*)malloc(matrix_size);
    
    for(int i=0;i<WIDTH*HEIGHT;i++){
        h_input[i]=static_cast<float>(rand()) / RAND_MAX;
    }
    
    float *d_input,*d_output;
    cudaMalloc(&d_input,matrix_size);
    cudaMalloc(&d_output,matrix_size);
    cudaMemcpy(d_input,h_input,matrix_size,cudaMemcpyHostToDevice);

    dim3 threads(TILE_SIZE, TILE_SIZE);
    dim3 blocks(WIDTH / TILE_SIZE, HEIGHT / TILE_SIZE);
    transpose<<<blocks, threads>>>(d_input, d_output, WIDTH, HEIGHT);

    checkCudaError(cudaGetLastError());             // Check for kernel launch errors
    checkCudaError(cudaDeviceSynchronize());        // Wait for kernel to finish

    checkCudaError(cudaMemcpy(h_output, d_output, matrix_size, cudaMemcpyDeviceToHost));


    bool success = true;
    for (int i = 0; i < WIDTH; i++) {
        for (int j = 0; j < HEIGHT; j++) {
            if (h_output[i * HEIGHT + j] != h_input[j * WIDTH + i]) {
                success = false;
                break;
            }
        }
    }

    std::cout << (success ? "Matrix transposition succeeded!" : "Matrix transposition failed!") << std::endl;


    cudaFree(d_input);
    cudaFree(d_output);

    free(h_input);
    free(h_output);

    return 0;
}