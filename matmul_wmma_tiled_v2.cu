#include <stdio.h>
#include <cuda_runtime.h>
#include <mma.h>
using namespace nvcuda::wmma;

#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

#define BM 64
#define BN 64
#define BK 16
// warp 数 = (BM/WMMA_M) × (BN/WMMA_N) = 4×4 = 16
// block = 16 warps × 32 = 512 threads
#define WM 4
#define WN 4

__global__ void matmul_wmma_tiled_v2(
    const half* __restrict__ A,
    const half* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K)
{
    int warp_id  = threadIdx.x / 32;
    int warp_row = warp_id / WN;        // 0..3
    int warp_col = warp_id % WN;        // 0..3

    int block_row = blockIdx.y * BM;
    int block_col = blockIdx.x * BN;

    int c_row = block_row + warp_row * WMMA_M;
    int c_col = block_col + warp_col * WMMA_N;

    __shared__ half smem_A[BM][BK];    // 64×16 × 2 = 2 KB
    __shared__ half smem_B[BK][BN];    // 16×64 × 2 = 2 KB

    fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc;
    fill_fragment(acc, 0.0f);

    int tid = threadIdx.x;
    int num_threads = blockDim.x;   // 512

    int num_tiles = (K + BK - 1) / BK;

    for (int t = 0; t < num_tiles; t++) {
        int k_base = t * BK;

        // smem_A: BM×BK = 64×16 = 1024 half，512 threads 各搬 2
        for (int i = tid; i < BM * BK; i += num_threads) {
            int r = i / BK, c = i % BK;
            int gr = block_row + r, gc = k_base + c;
            smem_A[r][c] = (gr < M && gc < K)
                ? A[gr * K + gc] : __float2half(0.0f);
        }

        // smem_B: BK×BN = 16×64 = 1024 half，512 threads 各搬 2
        for (int i = tid; i < BK * BN; i += num_threads) {
            int r = i / BN, c = i % BN;
            int gr = k_base + r, gc = block_col + c;
            smem_B[r][c] = (gr < K && gc < N)
                ? B[gr * N + gc] : __float2half(0.0f);
        }

        __syncthreads();

        // 每个 warp 做 1 次 mma（BK == WMMA_K == 16）
        fragment<matrix_a, WMMA_M, WMMA_N, WMMA_K, half, row_major> a_frag;
        fragment<matrix_b, WMMA_M, WMMA_N, WMMA_K, half, row_major> b_frag;

        load_matrix_sync(a_frag, &smem_A[warp_row * WMMA_M][0], BK);
        load_matrix_sync(b_frag, &smem_B[0][warp_col * WMMA_N], BN);
        mma_sync(acc, a_frag, b_frag, acc);

        __syncthreads();
    }

    if (c_row < M && c_col < N)
        store_matrix_sync(&C[c_row * N + c_col], acc, N, mem_row_major);
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
        if (diff > 0.1f * fabsf(ref[i]) + 0.05f)
            if (bad++ < 3)
                printf("MISMATCH at %d: ref=%.4f got=%.4f\n", i, ref[i], out[i]);
    }
    if (bad) { printf("Total: %d/%d\n", bad, n); return false; }
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

    dim3 block(512, 1);
    dim3 gv((Nv+BN-1)/BN, (Mv+BM-1)/BM);
    matmul_wmma_tiled_v2<<<gv, block>>>(dA, dB, dC, Mv, Nv, Kv);
    cudaDeviceSynchronize();
    cudaMemcpy(sC_gpu, dC, Mv*Nv*sizeof(float), cudaMemcpyDeviceToHost);
    printf("Correctness: %s\n", verify(sC_ref, sC_gpu, Mv*Nv) ? "PASSED" : "FAILED");
    cudaFree(dA); cudaFree(dB); cudaFree(dC);

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

    dim3 grid((N+BN-1)/BN, (M+BM-1)/BM);
    matmul_wmma_tiled_v2<<<grid, block>>>(dA, dB, dC, M, N, K);
    cudaDeviceSynchronize();

    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);
    cudaEventRecord(start);
    for (int i = 0; i < 10; i++)
        matmul_wmma_tiled_v2<<<grid, block>>>(dA, dB, dC, M, N, K);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms; cudaEventElapsedTime(&ms, start, stop); ms /= 10;
    double tflops = 2.0*M*N*K / (ms*1e-3) / 1e12;

    printf("WMMA tiled v2 1024x1024: %.2f ms, %.3f TFLOPS, %.1f%% TC peak\n",
           ms, tflops, tflops/330.0*100);
    printf("WMMA simple   1024x1024: 0.20 ms, 10.799 TFLOPS,  3.3%% TC peak\n");
    printf("Coarse v2     1024x1024: 0.47 ms,  4.585 TFLOPS,  9.4%% FP32 peak\n");

    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    delete[] hA_f; delete[] hB_f; delete[] hA_h; delete[] hB_h;
    delete[] sA_f; delete[] sB_f; delete[] sA_h; delete[] sB_h;
    delete[] sC_ref; delete[] sC_gpu;
    return 0;
}
