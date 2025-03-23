#include<iostream>
#include <cuda_runtime.h>

#define N 10

__global__ void matrixMul(float *A,float *B,float*C){
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if(row <N && col<N){
        float sum =0.0f;
        for(int i=0;i<N;i++){
        sum+=A[row*N+i]*B[i*N+col];
        }
    C[row*N+col]=sum;
    }
}


int main(){
    float *A, *B, *C;
    const int size = N * N * sizeof(float);

    // initialize the input matrices
    A = (float *)malloc( N * N* sizeof(float));
    B = (float *)malloc(N * N* sizeof(float));
    C = (float *)malloc(N * N* sizeof(float));

    // Initialize matrices
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            A[i * N + j] = 1.0f;
            B[i * N + j] = 2.0f;
            C[i * N + j] = 0.0f;
        }
    }

    float *d_A, *d_B, *d_C;
    cudaMalloc((void **)&d_A, size);
    cudaMalloc((void **)&d_B, size);
    cudaMalloc((void **)&d_C, size);
    cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);

    dim3 blockDim(16, 16);
    dim3 gridDim((N+blockDim.x-1)/blockDim.x,(N+blockDim.y-1)/blockDim.y);
    matrixMul<<<gridDim,blockDim>>>(d_A,d_B,d_C);
    cudaDeviceSynchronize();

    cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost);

    printf("A:\n");
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {

            printf("%.2f ", A[i * N + j]);
        }
        printf("\n"); 
    }

    printf("B:\n");
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {

            printf("%.2f ", B[i * N + j]);
        }
        printf("\n");
    }

    std::cout << "Result Matrix C:" << std::endl;
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            std::cout << C[i * N + j] << " ";
        }
        std::cout << std::endl;
    }

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
     return 0;
}