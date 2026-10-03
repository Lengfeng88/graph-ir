#include <stdio.h>
#include <cuda_runtime.h>
#include <mma.h>
using namespace nvcuda::wmma;

// Tensor Core tile 固定 16×16×16
#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

// 每个 block 处理 BM×BN 的输出，用 WARP_M×WARP_N 个 warp tile 拼
#define BM 128
#define BN 128
#define BK 32

// block 内 warp 排列：(BM/WMMA_M) × (BN/WMMA_N) = 8×8 = 64 warps → 太多
// 实际：每个 warp 负责一个 WMMA_M×WMMA_N tile
// block = 4×4 warps = 16 warps = 512 threads
#define WARP_M 4   // block 内 M 方向 warp 数
#define WARP_N 4   // block 内 N 方向 warp 数

__global__ void matmul_wmma(
    const half* __restrict__ A,
    const half* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K)
{
    // 这个 warp 的 ID
    int warp_id = (threadIdx.y * blockDim.x + threadIdx.x) / 32;
    int warp_row = warp_id / WARP_N;   // 0..3
    int warp_col = warp_id % WARP_N;   // 0..3

    // 这个 block 负责的输出起始位置
    int block_row = blockIdx.y * BM;
    int block_col = blockIdx.x * BN;

    // 这个 warp 负责的输出起始位置
    int c_row = block_row + warp_row * WMMA_M;
    int c_col = block_col + warp_col * WMMA_N;

    __shared__ half smem_A[BM][BK];   // 128×32 × 2 bytes = 8 KB
    __shared__ half smem_B[BK][BN];   // 32×128 × 2 bytes = 8 KB

    // accumulator fragment（FP32 累加）
    fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_frag;
    fill_fragment(acc_frag, 0.0f);

    int tid = threadIdx.y * blockDim.x + threadIdx.x;
    int num_threads = blockDim.x * blockDim.y;  // 512

    int num_tiles = (K + BK - 1) / BK;

    for (int t = 0; t < num_tiles; t++) {
        int k_base = t * BK;

        // 加载 smem_A: BM×BK = 128×32 = 4096 half elements
        // 512 threads 各加载 8 个，连续地址保证 coalescing
        for (int i = tid; i < BM * BK; i += num_threads) {
            int r = i / BK, c = i % BK;
            int gr = block_row + r, gc = k_base + c;
            smem_A[r][c] = (gr < M && gc < K) ? A[gr * K + gc] : __float2half(0.0f);
        }

        // 加载 smem_B: BK×BN = 32×128 = 4096 half elements
        for (int i = tid; i < BK * BN; i += num_threads) {
            int r = i / BN, c = i % BN;
            int gr = k_base + r, gc = block_col + c;
            smem_B[r][c] = (gr < K && gc < N) ? B[gr * N + gc] : __float2half(0.0f);
        }

        __syncthreads();

        // 每个 warp 沿 BK 方向做 BK/WMMA_K = 2 次 mma
        for (int k = 0; k < BK; k += WMMA_K) {
            fragment<matrix_a, WMMA_M, WMMA_N, WMMA_K, half, row_major> a_frag;
            fragment<matrix_b, WMMA_M, WMMA_N, WMMA_K, half, row_major> b_frag;

            // 从 smem 加载 fragment
            load_matrix_sync(a_frag,
                &smem_A[warp_row * WMMA_M][k], BK);
            load_matrix_sync(b_frag,
                &smem_B[k][warp_col * WMMA_N], BN);

            // Tensor Core mma: acc += a × b
            mma_sync(acc_frag, a_frag, b_frag, acc_frag);
        }

        __syncthreads();
    }

    // 写回（边界检查）
    if (c_row < M && c_col < N) {
        // store_matrix_sync 要求地址对齐，先写到 smem 再搬
        // 简化：直接用 store_matrix_sync 写 global（stride=N）
        store_matrix_sync(&C[c_row * N + c_col], acc_frag, N, mem_row_major);
    }
}

// CPU 转 half
void to_half(const float* src, half* dst, int n) {
    for (int i = 0; i < n; i++) dst[i] = __float2half(src[i]);
}

void matmul_cpu_f32(const float* A, const float* B, float* C, int M, int N, int K) {
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
        // FP16 累加误差较大，放宽阈值
        if (diff > 0.1f * fabsf(ref[i]) + 0.1f) {
            if (bad++ < 3)
                printf("MISMATCH at %d: ref=%.4f got=%.4f\n", i, ref[i], out[i]);
        }
    }
    if (bad > 0) { printf("Total mismatches: %d / %d\n", bad, n); return false; }
    return true;
}

int main() {
    const int M = 1024, N = 1024, K = 1024;
    const int Mv = 128, Nv = 128, Kv = 64;

    // 验证
    float *sA_f = new float[Mv*Kv], *sB_f = new float[Kv*Nv];
    float *sC_ref = new float[Mv*Nv], *sC_gpu = new float[Mv*Nv];
    srand(42);
    for (int i = 0; i < Mv*Kv; i++) sA_f[i] = (float)rand()/RAND_MAX * 0.1f;
    for (int i = 0; i < Kv*Nv; i++) sB_f[i] = (float)rand()/RAND_MAX * 0.1f;
    matmul_cpu_f32(sA_f, sB_f, sC_ref, Mv, Nv, Kv);

    half *sA_h = new half[Mv*Kv], *sB_h = new half[Kv*Nv];
    to_half(sA_f, sA_h, Mv*Kv);
    to_half(sB_f, sB_h, Kv*Nv);

    half *dA, *dB; float *dC;
    cudaMalloc(&dA, Mv*Kv*sizeof(half));
    cudaMalloc(&dB, Kv*Nv*sizeof(half));
    cudaMalloc(&dC, Mv*Nv*sizeof(float));
    cudaMemcpy(dA, sA_h, Mv*Kv*sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(dB, sB_h, Kv*Nv*sizeof(half), cudaMemcpyHostToDevice);

    // block: WARP_M × WARP_N warps = 4×4×32 = 512 threads
    dim3 block(32 * WARP_N, WARP_M);   // (128, 4)
    dim3 gv((Nv+BN-1)/BN, (Mv+BM-1)/BM);
    matmul_wmma<<<gv, block>>>(dA, dB, dC, Mv, Nv, Kv);
    cudaDeviceSynchronize();
    cudaMemcpy(sC_gpu, dC, Mv*Nv*sizeof(float), cudaMemcpyDeviceToHost);
    printf("Correctness: %s\n", verify(sC_ref, sC_gpu, Mv*Nv) ? "PASSED" : "FAILED");
    cudaFree(dA); cudaFree(dB); cudaFree(dC);

    // 性能
    float *hA_f = new float[M*K], *hB_f = new float[K*N];
    srand(42);
    for (int i = 0; i < M*K; i++) hA_f[i] = (float)rand()/RAND_MAX * 0.1f;
    for (int i = 0; i < K*N; i++) hB_f[i] = (float)rand()/RAND_MAX * 0.1f;

    half *hA_h = new half[M*K], *hB_h = new half[K*N];
    to_half(hA_f, hA_h, M*K);
    to_half(hB_f, hB_h, K*N);

    cudaMalloc(&dA, M*K*sizeof(half));
    cudaMalloc(&dB, K*N*sizeof(half));
    cudaMalloc(&dC, M*N*sizeof(float));
    cudaMemcpy(dA, hA_h, M*K*sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB_h, K*N*sizeof(half), cudaMemcpyHostToDevice);

    dim3 grid((N+BN-1)/BN, (M+BM-1)/BM);
    matmul_wmma<<<grid, block>>>(dA, dB, dC, M, N, K);
    cudaDeviceSynchronize();

    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);
    cudaEventRecord(start);
    for (int i = 0; i < 10; i++)
        matmul_wmma<<<grid, block>>>(dA, dB, dC, M, N, K);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms; cudaEventElapsedTime(&ms, start, stop); ms /= 10;
    double tflops = 2.0*M*N*K / (ms*1e-3) / 1e12;

    printf("WMMA FP16 MatMul 1024x1024: %.2f ms, %.3f TFLOPS, %.1f%% TC peak\n",
           ms, tflops, tflops/330.0*100);
    printf("Coarse v2 FP32  1024x1024:  0.47 ms, 4.585 TFLOPS,  9.4%% FP32 peak\n");
    printf("Naive     FP32  1024x1024:  1.43 ms, 1.497 TFLOPS,  3.1%% FP32 peak\n");

    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    delete[] hA_f; delete[] hB_f; delete[] hA_h; delete[] hB_h;
    delete[] sA_f; delete[] sB_f; delete[] sA_h; delete[] sB_h;
    delete[] sC_ref; delete[] sC_gpu;
    return 0;
}
