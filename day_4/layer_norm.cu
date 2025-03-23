#include<iostream>
#include<cuda_runtime.h>
#include <cmath>

#define WARP_SIZE 32
#define EPSILON 1e-5f

__global__ void layerNorm(const float* __restrict__ A, float* __restrict__ B,int rows, int cols){
    extern __shared__ float shared[]; //using shared memory for row ops
    int row = blockIdx.x*blockDim.x+threadIdx.x;
    if(row>=rows)
    return;
    for(int r = row;r<rows;r+=gridDim.x*blockDim.x){
        float *row_data =shared;
        for(int col=threadIdx.y;col<cols;col+=blockDim.y){
            row_data[col]=A[r*cols+col];

        }

        __syncthreads();

        float sum = 0.0f;
        for(int col = threadIdx.y;col<cols;col+=blockDim.y){
            sum+=row_data[col];
        }

        #pragma unroll
        for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        sum +=__shfl_down_sync(0xFFFFFFFF,sum,offset);
        }

        float mean = sum/cols;
        float variance_sum =0.0f;
        for(int col = threadIdx.y;col<cols;col+=blockDim.y){
        float diff=row_data[col]-mean;
        variance_sum=diff*diff;
        }

        #pragma unroll
        for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        variance_sum +=__shfl_down_sync(0xFFFFFFFF,variance_sum,offset);
        }

        float variance = variance_sum / cols;
        float stddev = rsqrtf(variance + EPSILON);

        //normalizing row
        for(int col = threadIdx.y;col<cols;col+=blockDim.y){
            B[r*cols+col]=(row_data[col]-mean)*stddev;
        }
    }
}


int main(){
    const int rows = 1024, cols = 512;
    float *A, *B;
    A = (float*)malloc(rows * cols * sizeof(float));
    B = (float*)malloc(rows * cols * sizeof(float));
    for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
            A[i * cols + j] = static_cast<float>(rand()) / RAND_MAX;
        }
    }
       
    float *d_a, *d_b;
    cudaMalloc(&d_a, rows * cols * sizeof(float));
    cudaMalloc(&d_b, rows * cols * sizeof(float));

    cudaMemcpy(d_a, A, rows * cols * sizeof(float), cudaMemcpyHostToDevice);
    
    dim3 blockDim(32, 16);
    dim3 gridDim((rows + blockDim.x - 1) / blockDim.x);
    size_t shared_memory_size = cols * sizeof(float);

    layerNorm<<<gridDim, blockDim,shared_memory_size>>>(d_a,d_b,rows,cols);
    
    cudaDeviceSynchronize();

    cudaMemcpy(B, d_b, rows * cols * sizeof(float), cudaMemcpyDeviceToHost);

    std::cout << "Sample Result:" << std::endl;
    for (int i = 0; i < 5; ++i) {
        for (int j = 0; j < 5; ++j) {
            std::cout << B[i * cols + j] << " ";
        }
        std::cout << std::endl;
    }

    cudaFree(d_a);
    cudaFree(d_b);
    free(A);
    free(B);


return 0;
}