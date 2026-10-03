#include <stdio.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 32

__global__ void matmul_tiled_v2(
    const float* A, const float* B, float* C,
    int M, int N, int K)
{
    int row = blockIdx.y * BLOCK_SIZE + threadIdx.y;
    int col = blockIdx.x * BLOCK_SIZE + threadIdx.x;

    // +1 padding 消除 bank conflict
    __shared__ float smem_A[BLOCK_SIZE][BLOCK_SIZE + 1];
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

void matmul_cpu(const float* A, const float* B, float* C, int M, int N, int K) {
    for (int i = 0; i < M; i++)
        for (int j = 0; j < N; j++) {
            float s = 0;
            for (int k = 0; k < K; k++)
                s += A[i*K+k] * B[k*N+j];
            C[i*N+j] = s;
        }
}

bool verify(const float* ref, const float* out, int n, float eps=1e-3f) {
    for (int i = 0; i < n; i++) {
        float diff = fabsf(ref[i] - out[i]);
        if (diff > eps * fabsf(ref[i]) + eps) {
            printf("MISMATCH at %d: ref=%.6f got=%.6f\n", i, ref[i], out[i]);
            return false;
        }
    }
    return true;
}

int main() {
    const int M = 1024, N = 1024, K = 1024;
    const int Mv = 64, Nv = 64, Kv = 64;

    float *sA = new float[Mv*Kv], *sB = new float[Kv*Nv];
    float *sC_ref = new float[Mv*Nv], *sC_gpu = new float[Mv*Nv];
    srand(42);
    for (int i = 0; i < Mv*Kv; i++) sA[i] = (float)rand()/RAND_MAX;
    for (int i = 0; i < Kv*Nv; i++) sB[i] = (float)rand()/RAND_MAX;
    matmul_cpu(sA, sB, sC_ref, Mv, Nv, Kv);

    float *dA, *dB, *dC;
    cudaMalloc(&dA, Mv*Kv*sizeof(float));
    cudaMalloc(&dB, Kv*Nv*sizeof(float));
    cudaMalloc(&dC, Mv*Nv*sizeof(float));
    cudaMemcpy(dA, sA, Mv*Kv*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dB, sB, Kv*Nv*sizeof(float), cudaMemcpyHostToDevice);
    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gv((Nv+BLOCK_SIZE-1)/BLOCK_SIZE, (Mv+BLOCK_SIZE-1)/BLOCK_SIZE);
    matmul_tiled_v2<<<gv, block>>>(dA, dB, dC, Mv, Nv, Kv);
    cudaDeviceSynchronize();
    cudaMemcpy(sC_gpu, dC, Mv*Nv*sizeof(float), cudaMemcpyDeviceToHost);
    printf("Correctness: %s\n", verify(sC_ref, sC_gpu, Mv*Nv) ? "PASSED" : "FAILED");
    cudaFree(dA); cudaFree(dB); cudaFree(dC);

    float *hA = new float[M*K], *hB = new float[K*N];
    srand(42);
    for (int i = 0; i < M*K; i++) hA[i] = (float)rand()/RAND_MAX;
    for (int i = 0; i < K*N; i++) hB[i] = (float)rand()/RAND_MAX;

    cudaMalloc(&dA, M*K*sizeof(float));
    cudaMalloc(&dB, K*N*sizeof(float));
    cudaMalloc(&dC, M*N*sizeof(float));
    cudaMemcpy(dA, hA, M*K*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB, K*N*sizeof(float), cudaMemcpyHostToDevice);

    dim3 grid((N+BLOCK_SIZE-1)/BLOCK_SIZE, (M+BLOCK_SIZE-1)/BLOCK_SIZE);

    matmul_tiled_v2<<<grid, block>>>(dA, dB, dC, M, N, K);
    cudaDeviceSynchronize();

    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);
    cudaEventRecord(start);
    for (int i = 0; i < 10; i++)
        matmul_tiled_v2<<<grid, block>>>(dA, dB, dC, M, N, K);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms; cudaEventElapsedTime(&ms, start, stop); ms /= 10;
    double tflops = 2.0*M*N*K / (ms*1e-3) / 1e12;

    printf("Tiled v2 MatMul 1024x1024: %.2f ms, %.3f TFLOPS\n", ms, tflops);
    printf("Tiled v1 MatMul 1024x1024: 1.13 ms, 1.901 TFLOPS\n");
    printf("Naive    MatMul 1024x1024: 1.43 ms, 1.497 TFLOPS\n");
    printf("Speedup over naive: %.1fx\n", 1.43f / ms);
    printf("Peak utilization: %.1f%%\n", tflops/48.7*100);

    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    delete[] hA; delete[] hB; delete[] sA; delete[] sB;
    delete[] sC_ref; delete[] sC_gpu;
    return 0;
}
