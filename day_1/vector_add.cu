#include <iostream>
__global__ void vectorAdd(float *A,float *B,float *C,int n){
int i = blockDim.x*blockIdx.x+threadIdx.x;
if (i<n) C[i] = A[i]+B[i];
}
int main(){
const int n =10;
float A[n],B[n],C[n];
float *d_a,*d_b,*d_c;
cudaMalloc(&d_a,n*sizeof(float));
cudaMalloc(&d_b,n*sizeof(float));
cudaMalloc(&d_c,n*sizeof(float));
cudaMemcpy(d_a,A,n*sizeof(float),cudaMemcpyHostToDevice);
cudaMemcpy(d_b,B,n*sizeof(float),cudaMemcpyHostToDevice);
int blockSize = 256;
int gridSize = ceil(n/blockSize);
vectorAdd<<<gridSize,blockSize>>>(d_a,d_b,d_c,n);
cudaMemcpy(d_c,C,n*sizeof(float),cudaMemcpyDeviceToHost);
cudaFree(d_a);
cudaFree(d_b);
cudaFree(d_c);

}