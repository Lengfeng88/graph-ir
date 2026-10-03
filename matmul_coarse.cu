#include <stdio.h>
#include <cuda_runtime.h>

// 每个 thread 计算 TM×TN 个输出元素
// block 负责 BM×BN 的输出 tile
// K 方向每次加载 BK 列
#define BM 64
#define BN 64
#define BK 16
#define TM 8   // 每个 thread 在 M 方向算 8 个元素
#define TN 8   // 每个 thread 在 N 方向算 8 个元素

// block size = (BM/TM) × (BN/TN) = 8 × 8 = 64 threads
__global__ void matmul_coarse(
    const float* A, const float* B, float* C,
    int M, int N, int K)
{
    // 这个 block 负责 C 的哪个 tile
    int block_row = blockIdx.y * BM;
    int block_col = blockIdx.x * BN;

    // 这个 thread 在 block 内的位置
    int thread_row = threadIdx.y * TM;  // 在 BM 内的起始行
    int thread_col = threadIdx.x * TN;  // 在 BN 内的起始列

    __shared__ float smem_A[BM][BK];
    __shared__ float smem_B[BK][BN];

    // 每个 thread 的 accumulator，TM×TN 个
    float acc[TM][TN] = {0.0f};

    // 加载 smem 时，64 个 thread 分工
    // smem_A: BM×BK = 64×16 = 1024 elements，64 threads 各搬 16 个
    // smem_B: BK×BN = 16×64 = 1024 elements，64 threads 各搬 16 个
    int tid = threadIdx.y * blockDim.x + threadIdx.x;  // 0..63

    int num_tiles = (K + BK - 1) / BK;

    for (int t = 0; t < num_tiles; t++) {
        int k_base = t * BK;

        // 加载 smem_A: BM×BK，每个 thread 加载 BM*BK/64 = 16 个元素
        for (int i = 0; i < BM * BK / 64; i++) {
            int idx = tid * (BM * BK / 64) + i;
            int r = idx / BK;
            int c = idx % BK;
            int global_row = block_row + r;
            int global_col = k_base + c;
            smem_A[r][c] = (global_row < M && global_col < K)
                ? A[global_row * K + global_col] : 0.0f;
        }

        // 加载 smem_B: BK×BN，每个 thread 加载 16 个元素
        for (int i = 0; i < BK * BN / 64; i++) {
            int idx = tid * (BK * BN / 64) + i;
            int r = idx / BN;
            int c = idx % BN;
            int global_row = k_base + r;
            int global_col = block_col + c;
            smem_B[r][c] = (global_row < K && global_col < N)
                ? B[global_row * N + global_col] : 0.0f;
        }

        __syncthreads();

        // 计算：每个 thread 做 TM×TN×BK 次 fma
        for (int k = 0; k < BK; k++) {
            // 先把 smem_A 的一列和 smem_B 的一行读进 register
            float a_reg[TM], b_reg[TN];
            for (int m = 0; m < TM; m++)
                a_reg[m] = smem_A[thread_row + m][k];
            for (int n = 0; n < TN; n++)
                b_reg[n] = smem_B[k][thread_col + n];
            // outer product
            for (int m = 0; m < TM; m++)
                for (int n = 0; n < TN; n++)
                    acc[m][n] += a_reg[m] * b_reg[n];
        }

        __syncthreads();
    }

    // 写回
    for (int m = 0; m < TM; m++)
        for (int n = 0; n < TN; n++) {
            int r = block_row + thread_row + m;
            int c = block_col + thread_col + n;
            if (r < M && c < N)
                C[r * N + c] = acc[m][n];
        }
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
    dim3 block_v(BN/TN, BM/TM);
    dim3 grid_v((Nv+BN-1)/BN, (Mv+BM-1)/BM);
    matmul_coarse<<<grid_v, block_v>>>(dA, dB, dC, Mv, Nv, Kv);
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

    dim3 block(BN/TN, BM/TM);
    dim3 grid((N+BN-1)/BN, (M+BM-1)/BM);

    matmul_coarse<<<grid, block>>>(dA, dB, dC, M, N, K);
    cudaDeviceSynchronize();

    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);
    cudaEventRecord(start);
    for (int i = 0; i < 10; i++)
        matmul_coarse<<<grid, block>>>(dA, dB, dC, M, N, K);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms; cudaEventElapsedTime(&ms, start, stop); ms /= 10;
    double tflops = 2.0*M*N*K / (ms*1e-3) / 1e12;

    printf("Coarse  MatMul 1024x1024: %.2f ms, %.3f TFLOPS, %.1f%% peak\n",
           ms, tflops, tflops/48.7*100);
    printf("Tiled16 MatMul 1024x1024: 1.10 ms, 1.958 TFLOPS,  4.0%% peak\n");
    printf("Naive   MatMul 1024x1024: 1.43 ms, 1.497 TFLOPS,  3.1%% peak\n");
    printf("Speedup over naive: %.1fx\n", 1.43f/ms);

    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    delete[] hA; delete[] hB; delete[] sA; delete[] sB;
    delete[] sC_ref; delete[] sC_gpu;
    return 0;
}
