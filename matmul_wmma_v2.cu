#include <stdio.h>
#include <cuda_runtime.h>
#include <mma.h>
using namespace nvcuda::wmma;

#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

// 简化：每个 block 只有 1 个 warp，负责一个 16×16 的输出 tile
// 彻底排除 warp 索引 bug
__global__ void matmul_wmma_simple(
    const half* __restrict__ A,
    const half* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K)
{
    int c_row = blockIdx.y * WMMA_M;
    int c_col = blockIdx.x * WMMA_N;

    if (c_row >= M || c_col >= N) return;

    fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc;
    fill_fragment(acc, 0.0f);

    for (int k = 0; k < K; k += WMMA_K) {
        fragment<matrix_a, WMMA_M, WMMA_N, WMMA_K, half, row_major> a_frag;
        fragment<matrix_b, WMMA_M, WMMA_N, WMMA_K, half, row_major> b_frag;

        if (k + WMMA_K <= K) {
            load_matrix_sync(a_frag, A + c_row * K + k, K);
            load_matrix_sync(b_frag, B + k * N + c_col, N);
            mma_sync(acc, a_frag, b_frag, acc);
        }
    }

    store_matrix_sync(C + c_row * N + c_col, acc, N, mem_row_major);
}

void to_half(const float* src, half* dst, int n) {
    for (int i = 0; i < n; i++) dst[i] = __float2half(src[i]);
}

void matmul_cpu(const float* A, const float* B, float* C, int M, int N, int K) {
    for (int i = 0; i < M; i++)
        for (int j = 0; j < N; j++) {
            float s = 0;
            for (int k = 0; k < K; k++) s += A[i*K+k] * B[k*N+j];
            C[i*N+j] = s;
        }
}

bool verify(const float* ref, const float* out, int n) {
    int bad = 0;
    for (int i = 0; i < n; i++) {
        float diff = fabsf(ref[i] - out[i]);
        if (diff > 0.1f * fabsf(ref[i]) + 0.1f)
            if (bad++ < 3)
                printf("MISMATCH at %d: ref=%.4f got=%.4f\n", i, ref[i], out[i]);
    }
    if (bad) { printf("Total: %d / %d\n", bad, n); return false; }
    return true;
}

int main() {
    const int M = 1024, N = 1024, K = 1024;
    const int Mv = 64, Nv = 64, Kv = 64;

    float *sA_f = new float[Mv*Kv], *sB_f = new float[Kv*Nv];
    float *sC_ref = new float[Mv*Nv], *sC_gpu = new float[Mv*Nv];
    srand(42);
    for (int i = 0; i < Mv*Kv; i++) sA_f[i] = (float)rand()/RAND_MAX * 0.1f;
    for (int i = 0; i < Kv*Nv; i++) sB_f[i] = (float)rand()/RAND_MAX * 0.1f;
    matmul_cpu(sA_f, sB_f, sC_ref, Mv, Nv, Kv);

    half *sA_h = new half[Mv*Kv], *sB_h = new half[Kv*Nv];
    to_half(sA_f, sA_h, Mv*Kv); to_half(sB_f, sB_h, Kv*Nv);

    half *dA, *dB; float *dC;
    cudaMalloc(&dA, Mv*Kv*sizeof(half));
    cudaMalloc(&dB, Kv*Nv*sizeof(half));
    cudaMalloc(&dC, Mv*Nv*sizeof(float));
    cudaMemcpy(dA, sA_h, Mv*Kv*sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(dB, sB_h, Kv*Nv*sizeof(half), cudaMemcpyHostToDevice);
    cudaMemset(dC, 0, Mv*Nv*sizeof(float));

    // 1 warp per block，grid 覆盖整个输出
    dim3 block(32, 1);
    dim3 gv((Nv+WMMA_N-1)/WMMA_N, (Mv+WMMA_M-1)/WMMA_M);
    matmul_wmma_simple<<<gv, block>>>(dA, dB, dC, Mv, Nv, Kv);
    cudaDeviceSynchronize();
    cudaMemcpy(sC_gpu, dC, Mv*Nv*sizeof(float), cudaMemcpyDeviceToHost);
    printf("Correctness: %s\n", verify(sC_ref, sC_gpu, Mv*Nv) ? "PASSED" : "FAILED");
    cudaFree(dA); cudaFree(dB); cudaFree(dC);

    // 性能（大矩阵）
    float *hA_f = new float[M*K], *hB_f = new float[K*N];
    srand(42);
    for (int i = 0; i < M*K; i++) hA_f[i] = (float)rand()/RAND_MAX * 0.1f;
    for (int i = 0; i < K*N; i++) hB_f[i] = (float)rand()/RAND_MAX * 0.1f;
    half *hA_h = new half[M*K], *hB_h = new half[K*N];
    to_half(hA_f, hA_h, M*K); to_half(hB_f, hB_h, K*N);

    cudaMalloc(&dA, M*K*sizeof(half));
    cudaMalloc(&dB, K*N*sizeof(half));
    cudaMalloc(&dC, M*N*sizeof(float));
    cudaMemcpy(dA, hA_h, M*K*sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB_h, K*N*sizeof(half), cudaMemcpyHostToDevice);

    dim3 grid((N+WMMA_N-1)/WMMA_N, (M+WMMA_M-1)/WMMA_M);
    matmul_wmma_simple<<<grid, block>>>(dA, dB, dC, M, N, K);
    cudaDeviceSynchronize();

    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);
    cudaEventRecord(start);
    for (int i = 0; i < 10; i++)
        matmul_wmma_simple<<<grid, block>>>(dA, dB, dC, M, N, K);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms; cudaEventElapsedTime(&ms, start, stop); ms /= 10;
    double tflops = 2.0*M*N*K / (ms*1e-3) / 1e12;
    printf("WMMA simple 1024x1024: %.2f ms, %.3f TFLOPS, %.1f%% TC peak\n",
           ms, tflops, tflops/330.0*100);

    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    delete[] hA_f; delete[] hB_f; delete[] hA_h; delete[] hB_h;
    delete[] sA_f; delete[] sB_f; delete[] sA_h; delete[] sB_h;
    delete[] sC_ref; delete[] sC_gpu;
    return 0;
}
