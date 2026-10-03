#include <stdio.h>
#include <cuda_runtime.h>

#define BM 128
#define BN 128
#define BK 8
#define TM 8
#define TN 8
// block = (BN/TN) × (BM/TM) = 16 × 16 = 256 threads = 8 warps

__global__ __launch_bounds__(256, 2)
void matmul_coarse_v2(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K)
{
    int block_row = blockIdx.y * BM;
    int block_col = blockIdx.x * BN;

    // thread 在 block 内的位置
    // threadIdx.x: 0..15 (N方向), threadIdx.y: 0..15 (M方向)
    int ty = threadIdx.y;  // 0..15
    int tx = threadIdx.x;  // 0..15
    int tid = ty * 16 + tx;  // 0..255

    __shared__ float smem_A[BM][BK];   // 128×8 = 4KB
    __shared__ float smem_B[BK][BN];   // 8×128 = 4KB  总 8KB

    float acc[TM][TN] = {};

    // smem_A 加载：BM×BK = 1024 elements，256 threads 各搬 4
    // 每个 thread 搬连续的 4 个元素，保证 coalescing
    // smem_A[row][col]: row = tid/BK * 4 + i，col = tid%BK
    // 改为：tid 决定行，连续 thread 读连续列
    // row = tid / BK = tid / 8，col = tid % BK = tid % 8
    // 256/8 = 32 个不同行，每行由 8 个 thread 填，每 thread 填 1 element
    // 但 BM=128，需要 128×8=1024 elements，256 threads 各填 4
    // 方案：每个 thread 填 4 行，同一列
    // thread tid 填 smem_A[tid/8 * 4 + 0..3][tid%8]
    // → 同一 warp 内 tid 连续，tid%8 不同 → 访问不同 bank → 无冲突

    int smem_a_col = tid % BK;           // 0..7，决定 smem_A 的列
    int smem_a_row_base = (tid / BK) * 4; // 0,4,8,...,124

    int smem_b_row = tid / BN;           // 0..1，smem_B 的行
    int smem_b_col = tid % BN;           // 0..127，smem_B 的列

    // thread 负责 C 的哪些行列
    int c_row_base = ty * TM;   // 0,8,16,...,120
    int c_col_base = tx * TN;   // 0,8,16,...,120

    int num_tiles = (K + BK - 1) / BK;

    for (int t = 0; t < num_tiles; t++) {
        int k_base = t * BK;

        // 加载 smem_A: 每个 thread 加载 4 个元素（同列，连续行）
        for (int i = 0; i < 4; i++) {
            int gr = block_row + smem_a_row_base + i;
            int gc = k_base + smem_a_col;
            smem_A[smem_a_row_base + i][smem_a_col] =
                (gr < M && gc < K) ? A[gr * K + gc] : 0.0f;
        }

        // 加载 smem_B: 每个 thread 加载 4 个元素
        // smem_B[BK][BN] = 8×128，256 threads 各搬 4
        // thread tid → row = tid/(BN/4) = tid/32，col = (tid%32)*4 + 0..3
        // 同一 warp 32 threads → col 连续 → coalesced
        {
            int b_row = tid / (BN / 4);        // 0..7
            int b_col_base2 = (tid % (BN / 4)) * 4;  // 0,4,8,...,124
            for (int i = 0; i < 4; i++) {
                int gr = k_base + b_row;
                int gc = block_col + b_col_base2 + i;
                smem_B[b_row][b_col_base2 + i] =
                    (gr < K && gc < N) ? B[gr * N + gc] : 0.0f;
            }
        }

        __syncthreads();

        // 计算：每个 thread 做 TM×TN×BK 次 fma
        for (int k = 0; k < BK; k++) {
            float a_reg[TM], b_reg[TN];
            #pragma unroll
            for (int m = 0; m < TM; m++)
                a_reg[m] = smem_A[c_row_base + m][k];
            #pragma unroll
            for (int n = 0; n < TN; n++)
                b_reg[n] = smem_B[k][c_col_base + n];
            #pragma unroll
            for (int m = 0; m < TM; m++)
                #pragma unroll
                for (int n = 0; n < TN; n++)
                    acc[m][n] += a_reg[m] * b_reg[n];
        }

        __syncthreads();
    }

    // 写回
    #pragma unroll
    for (int m = 0; m < TM; m++)
        #pragma unroll
        for (int n = 0; n < TN; n++) {
            int r = block_row + c_row_base + m;
            int c = block_col + c_col_base + n;
            if (r < M && c < N)
                C[r * N + c] = acc[m][n];
        }
}

void matmul_cpu(const float* A, const float* B, float* C, int M, int N, int K) {
    for (int i = 0; i < M; i++)
        for (int j = 0; j < N; j++) {
            float s = 0;
            for (int k = 0; k < K; k++) s += A[i*K+k] * B[k*N+j];
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
    const int Mv = 128, Nv = 128, Kv = 64;

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
    dim3 bv(BN/TN, BM/TM);
    dim3 gv((Nv+BN-1)/BN, (Mv+BM-1)/BM);
    matmul_coarse_v2<<<gv, bv>>>(dA, dB, dC, Mv, Nv, Kv);
    cudaDeviceSynchronize();
    cudaMemcpy(sC_gpu, dC, Mv*Nv*sizeof(float), cudaMemcpyDeviceToHost);
    printf("Correctness: %s\n", verify(sC_ref, sC_gpu, Mv*Nv) ? "PASSED" : "FAILED");
    cudaFree(dA); cudaFree(dB); cudaFree(dC);

    float *hA = new float[M*K], *hB = new float[K*N];
    srand(42);
    for (int i = 0; i < M*K; i++) hA[i] = (float)rand()/RAND_MAX;
    for (int i = 0; i < K*N; i++) hB[i] = (float)rand()/RAND_MAX;
    cudaMalloc(&dA, M*K*sizeof(float)); cudaMalloc(&dB, K*N*sizeof(float));
    cudaMalloc(&dC, M*N*sizeof(float));
    cudaMemcpy(dA, hA, M*K*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB, K*N*sizeof(float), cudaMemcpyHostToDevice);

    dim3 block(BN/TN, BM/TM);
    dim3 grid((N+BN-1)/BN, (M+BM-1)/BM);

    matmul_coarse_v2<<<grid, block>>>(dA, dB, dC, M, N, K);
    cudaDeviceSynchronize();

    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);
    cudaEventRecord(start);
    for (int i = 0; i < 10; i++)
        matmul_coarse_v2<<<grid, block>>>(dA, dB, dC, M, N, K);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms; cudaEventElapsedTime(&ms, start, stop); ms /= 10;
    double tflops = 2.0*M*N*K / (ms*1e-3) / 1e12;

    printf("Coarse v2 MatMul 1024x1024: %.2f ms, %.3f TFLOPS, %.1f%% peak\n",
           ms, tflops, tflops/48.7*100);
    printf("Coarse v1 MatMul 1024x1024: 0.83 ms, 2.589 TFLOPS,  5.3%% peak\n");
    printf("Naive     MatMul 1024x1024: 1.43 ms, 1.497 TFLOPS,  3.1%% peak\n");

    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    delete[] hA; delete[] hB; delete[] sA; delete[] sB;
    delete[] sC_ref; delete[] sC_gpu;
    return 0;
}
