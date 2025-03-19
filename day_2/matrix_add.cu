#include<iostream>

__global__ void matrixAdd(const float *A,const float *B, float *C,const int n)
{
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    int j=blockIdx.y*blockDim.y+threadIdx.y;
    if( (i>=n) && (j>=n)){
    return ;
    }
        C[i*n+j] = A[i*n+j]+B[i*n+j];
}


int main() {
    const int n = 10;
    float *A, *B, *C;

    // initialize the input matrices
    A = (float *)malloc( n*n* sizeof(float));
    B = (float *)malloc(n*n* sizeof(float));
    C = (float *)malloc(n*n * sizeof(float));

    for(int i =0;i<n;i++){
        for (int j=0;j<n;j++){
            A[i*n+j]=1.0f;
            B[i*n+j]=2.0f;
            C[i*n+j]=0.0f;
        }
    }
    float *d_a,*d_b,*d_c;
    cudaMalloc((void **)&d_a,n*n*sizeof(float));
    cudaMalloc((void **)&d_b,n*n*sizeof(float));
    cudaMalloc((void **)&d_c,n*n*sizeof(float));

    cudaMemcpy(d_a,A,n*n*sizeof(float),cudaMemcpyHostToDevice);
    cudaMemcpy(d_b,B,n*n*sizeof(float),cudaMemcpyHostToDevice);

    dim3 dimBlock(32,16,1);
    dim3 dimGrid(ceil(n/32.0f),ceil(n/16.0f),1);
    matrixAdd<<<dimGrid,dimBlock>>>(d_a,d_b,d_c,n);
    cudaDeviceSynchronize();

    cudaMemcpy(C,d_c,n*n*sizeof(float),cudaMemcpyDeviceToHost);

    //print OG A and B Matrix
         printf("A:\n");
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {

            printf("%.2f ", A[i * n + j]);
        }
        printf("\n"); 
    }
     printf("B:\n");
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {

            printf("%.2f ", B[i * n + j]);
        }
        printf("\n");
    }

    //printing resulting C matrix
    printf("C:\n");
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {

            printf("%.2f ",C[i * n + j]);
        }
        printf("\n"); 
    }
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
}