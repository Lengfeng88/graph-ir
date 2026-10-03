#include <stdio.h>
#include <cuda_runtime.h>

// 改小 block size，每个 block thread 数从 1024 → 256
// register/block = 37 * 256 = 9472，SM 能放 65536/9472 = 6 个 block
#define BLOCK_SIZE 16

__global__ void matmul_tiled_v3(
    const float* A, const float* B, float* C,
    int M, int N, int K)
{
    int row = blockIdx.y * BLOCK_SIZE + threadIdx.y;
    int col = blockIdx.x * BLOCK_SIZE + threadIdx.x;

    __shared__ float smem_A[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float smem_B[BLOCK_SIZE][BLOCK_SIZE];

    float sum = 0.0f;

    int num_tiles = (K + BLOCK_SIZE - 1) / BLOCK_SIZE;
    for (int t = 0; t < num_tiles; t++) {
        int a_col = t * BLOCK_SIZE + threadIdx.x;
        int b_row = t * BLOCK_SIZE + threadIdx.y;

        smem_A[threadIdx.y][threadIdx.x] = (row < M && a_col < K)
            ? A[row * K + a_col] : 0.0f;
        smem_B[threadIdx.y][threadIdx.x] = (b_row < K && col < N)
            ? B[b_row * N + col] : 0.0f;

        __syncthreads();

        for (int k = 0; k < BLOCK_SIZE; k++)
            sum += smem_A[threadIdx.y][k] * smem_B[k][threadIdx.x];

        __syncthreads();
    }

    if (row < M && col < N)
        C[row * N + col] = sum;
}

int main() {
    const int M = 1024, N = 1024, K = 1024;

    float *hA = new float[M*K], *hB = new float[K*N];
    srand(42);
    for (int i = 0; i < M*K; i++) hA[i] = (float)rand()/RAND_MAX;
    for (int i = 0; i < K*N; i++) hB[i] = (float)rand()/RAND_MAX;

    float *dA, *dB, *dC;
    cudaMalloc(&dA, M*K*sizeof(float));
    cudaMalloc(&dB, K*N*sizeof(float));
    cudaMalloc(&dC, M*N*sizeof(float));
    cudaMemcpy(dA, hA, M*K*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB, K*N*sizeof(float), cudaMemcpyHostToDevice);

    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((N+BLOCK_SIZE-1)/BLOCK_SIZE, (M+BLOCK_SIZE-1)/BLOCK_SIZE);

    matmul_tiled_v3<<<grid, block>>>(dA, dB, dC, M, N, K);
    cudaDeviceSynchronize();

    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);
    cudaEventRecord(start);
    for (int i = 0; i < 10; i++)
        matmul_tiled_v3<<<grid, block>>>(dA, dB, dC, M, N, K);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms; cudaEventElapsedTime(&ms, start, stop); ms /= 10;
    double tflops = 2.0*M*N*K / (ms*1e-3) / 1e12;

    printf("BLOCK_SIZE=16: %.2f ms, %.3f TFLOPS, %.1f%% peak\n",
           ms, tflops, tflops/48.7*100);
    printf("BLOCK_SIZE=32: 1.66 ms, 1.291 TFLOPS,  2.7%% peak\n");

    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    delete[] hA; delete[] hB;
    return 0;
}
