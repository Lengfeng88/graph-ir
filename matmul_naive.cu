#include <stdio.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 32

__global__ void matmul_naive(
    const float* A, const float* B, float* C,
    int M, int N, int K)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M || col >= N) return;
    float sum = 0.0f;
    for (int k = 0; k < K; k++)
        sum += A[row * K + k] * B[k * N + col];
    C[row * N + col] = sum;
}

void matmul_cpu(const float* A, const float* B, float* C,
                int M, int N, int K)
{
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
    // 验证用小矩阵，独立分配，stride 正确
    const int Mv = 64, Nv = 64, Kv = 64;

    float *hA = new float[M*K];
    float *hB = new float[K*N];
    float *hC = new float[M*N];

    // 独立的小矩阵用于 CPU reference
    float *sA = new float[Mv*Kv];
    float *sB = new float[Kv*Nv];
    float *sC_ref = new float[Mv*Nv];
    float *sC_gpu = new float[Mv*Nv];

    srand(42);
    for (int i = 0; i < M*K; i++) hA[i] = (float)rand() / RAND_MAX;
    for (int i = 0; i < K*N; i++) hB[i] = (float)rand() / RAND_MAX;

    // 小矩阵填相同随机数
    srand(42);
    for (int i = 0; i < Mv*Kv; i++) sA[i] = (float)rand() / RAND_MAX;
    for (int i = 0; i < Kv*Nv; i++) sB[i] = (float)rand() / RAND_MAX;

    matmul_cpu(sA, sB, sC_ref, Mv, Nv, Kv);

    float *dA, *dB, *dC;
    cudaMalloc(&dA, Mv*Kv * sizeof(float));
    cudaMalloc(&dB, Kv*Nv * sizeof(float));
    cudaMalloc(&dC, Mv*Nv * sizeof(float));
    cudaMemcpy(dA, sA, Mv*Kv*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dB, sB, Kv*Nv*sizeof(float), cudaMemcpyHostToDevice);

    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid_v((Nv+BLOCK_SIZE-1)/BLOCK_SIZE, (Mv+BLOCK_SIZE-1)/BLOCK_SIZE);
    matmul_naive<<<grid_v, block>>>(dA, dB, dC, Mv, Nv, Kv);
    cudaDeviceSynchronize();
    cudaMemcpy(sC_gpu, dC, Mv*Nv*sizeof(float), cudaMemcpyDeviceToHost);

    if (verify(sC_ref, sC_gpu, Mv*Nv))
        printf("Correctness: PASSED\n");
    else
        printf("Correctness: FAILED\n");

    cudaFree(dA); cudaFree(dB); cudaFree(dC);

    // 性能测试用大矩阵
    cudaMalloc(&dA, M*K * sizeof(float));
    cudaMalloc(&dB, K*N * sizeof(float));
    cudaMalloc(&dC, M*N * sizeof(float));
    cudaMemcpy(dA, hA, M*K*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB, K*N*sizeof(float), cudaMemcpyHostToDevice);

    dim3 grid((N+BLOCK_SIZE-1)/BLOCK_SIZE, (M+BLOCK_SIZE-1)/BLOCK_SIZE);
    printf("Grid: (%d, %d)  Block: (%d, %d)\n", grid.x, grid.y, block.x, block.y);

    // warmup
    matmul_naive<<<grid, block>>>(dA, dB, dC, M, N, K);
    cudaDeviceSynchronize();

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    for (int i = 0; i < 10; i++)
        matmul_naive<<<grid, block>>>(dA, dB, dC, M, N, K);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    ms /= 10;

    double tflops = 2.0 * M * N * K / (ms * 1e-3) / 1e12;
    printf("Naive MatMul 1024x1024: %.2f ms, %.3f TFLOPS\n", ms, tflops);
    printf("Peak utilization: %.1f%%  (RTX 4080 FP32 peak ~48.7 TFLOPS)\n",
           tflops / 48.7 * 100);

    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    delete[] hA; delete[] hB; delete[] hC;
    delete[] sA; delete[] sB; delete[] sC_ref; delete[] sC_gpu;
    return 0;
}
