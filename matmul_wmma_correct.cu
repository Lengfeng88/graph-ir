#include <stdio.h>
#include <cuda_runtime.h>
#include <mma.h>
using namespace nvcuda::wmma;

// 参数设计原则：
// - 每个 warp 负责 WARP_M×WARP_N 个 WMMA tile
// - block 内 BLOCK_WARP_M×BLOCK_WARP_N 个 warp
// - block 负责 BM×BN 输出，BM=BLOCK_WARP_M*WARP_M*16，BN=BLOCK_WARP_N*WARP_N*16

#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

#define WARP_M 2          // 每个 warp 在 M 方向负责 2 个 16×16 tile
#define WARP_N 2          // 每个 warp 在 N 方向负责 2 个 16×16 tile
#define BLOCK_WARP_M 4    // block 内 M 方向 4 个 warp
#define BLOCK_WARP_N 4    // block 内 N 方向 4 个 warp

// block 负责输出大小
#define BM (BLOCK_WARP_M * WARP_M * WMMA_M)  // 4*2*16 = 128
#define BN (BLOCK_WARP_N * WARP_N * WMMA_N)  // 4*2*16 = 128
#define BK 32

// block = 4*4 = 16 warps = 512 threads
#define BLOCK_SIZE (BLOCK_WARP_M * BLOCK_WARP_N * 32)

__global__ __launch_bounds__(BLOCK_SIZE)
void matmul_wmma_correct(
    const half* __restrict__ A,
    const half* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K)
{
    // warp 的位置
    int warp_id   = threadIdx.x / 32;
    int warp_row  = warp_id / BLOCK_WARP_N;   // 0..3
    int warp_col  = warp_id % BLOCK_WARP_N;   // 0..3

    // block 负责的输出起始
    int block_row = blockIdx.y * BM;
    int block_col = blockIdx.x * BN;

    // smem
    __shared__ half smem_A[BM][BK];   // 128×32 × 2 = 8 KB
    __shared__ half smem_B[BK][BN];   // 32×128 × 2 = 8 KB  总 16 KB

    // 每个 warp 有 WARP_M×WARP_N 个 accumulator
    fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, float>
        acc[WARP_M][WARP_N];
    for (int m = 0; m < WARP_M; m++)
        for (int n = 0; n < WARP_N; n++)
            fill_fragment(acc[m][n], 0.0f);

    int tid = threadIdx.x;
    int num_threads = BLOCK_SIZE;

    for (int t = 0; t < (K + BK - 1) / BK; t++) {
        int k_base = t * BK;

        // 加载 smem_A: BM×BK = 128×32 = 4096 half
        // 512 threads 各加载 8 个，stride = 512
        for (int i = tid; i < BM * BK; i += num_threads) {
            int r = i / BK, c = i % BK;
            int gr = block_row + r, gc = k_base + c;
            smem_A[r][c] = (gr < M && gc < K)
                ? A[gr * K + gc] : __float2half(0.0f);
        }
        // 加载 smem_B: BK×BN = 32×128 = 4096 half
        for (int i = tid; i < BK * BN; i += num_threads) {
            int r = i / BN, c = i % BN;
            int gr = k_base + r, gc = block_col + c;
            smem_B[r][c] = (gr < K && gc < N)
                ? B[gr * N + gc] : __float2half(0.0f);
        }

        __syncthreads();

        // 每个 warp 计算 WARP_M×WARP_N 个 tile，沿 BK 做 BK/WMMA_K 次 mma
        #pragma unroll
        for (int k = 0; k < BK; k += WMMA_K) {
            fragment<matrix_a, WMMA_M, WMMA_N, WMMA_K, half, row_major>
                a_frag[WARP_M];
            fragment<matrix_b, WMMA_M, WMMA_N, WMMA_K, half, row_major>
                b_frag[WARP_N];

            // 加载这个 warp 负责的所有 A tiles（M 方向）
            #pragma unroll
            for (int m = 0; m < WARP_M; m++) {
                int smem_row = (warp_row * WARP_M + m) * WMMA_M;
                load_matrix_sync(a_frag[m], &smem_A[smem_row][k], BK);
            }
            // 加载这个 warp 负责的所有 B tiles（N 方向）
            #pragma unroll
            for (int n = 0; n < WARP_N; n++) {
                int smem_col = (warp_col * WARP_N + n) * WMMA_N;
                load_matrix_sync(b_frag[n], &smem_B[k][smem_col], BN);
            }
            // outer product: WARP_M × WARP_N 个 mma
            #pragma unroll
            for (int m = 0; m < WARP_M; m++)
                #pragma unroll
                for (int n = 0; n < WARP_N; n++)
                    mma_sync(acc[m][n], a_frag[m], b_frag[n], acc[m][n]);
        }

        __syncthreads();
    }

    // 写回
    #pragma unroll
    for (int m = 0; m < WARP_M; m++) {
        #pragma unroll
        for (int n = 0; n < WARP_N; n++) {
            int c_row = block_row + (warp_row * WARP_M + m) * WMMA_M;
            int c_col = block_col + (warp_col * WARP_N + n) * WMMA_N;
            if (c_row < M && c_col < N)
                store_matrix_sync(&C[c_row * N + c_col],
                                  acc[m][n], N, mem_row_major);
        }
    }
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
                printf("  MISMATCH at %d: ref=%.4f got=%.4f\n",
                       i, ref[i], out[i]);
    }
    if (bad) printf("  Total bad: %d/%d\n", bad, n);
    return bad == 0;
}

void bench(int M, int N, int K, bool check) {
    float *hA_f = new float[M*K], *hB_f = new float[K*N];
    float *hC_ref = nullptr, *hC_gpu = nullptr;
    srand(42);
    for (int i = 0; i < M*K; i++) hA_f[i] = (float)rand()/RAND_MAX * 0.1f;
    for (int i = 0; i < K*N; i++) hB_f[i] = (float)rand()/RAND_MAX * 0.1f;

    half *hA_h = new half[M*K], *hB_h = new half[K*N];
    to_half(hA_f, hA_h, M*K); to_half(hB_f, hB_h, K*N);

    if (check) {
        hC_ref = new float[M*N]; hC_gpu = new float[M*N];
        matmul_cpu(hA_f, hB_f, hC_ref, M, N, K);
    }

    half *dA, *dB; float *dC;
    cudaMalloc(&dA, M*K*sizeof(half));
    cudaMalloc(&dB, K*N*sizeof(half));
    cudaMalloc(&dC, M*N*sizeof(float));
    cudaMemcpy(dA, hA_h, M*K*sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB_h, K*N*sizeof(half), cudaMemcpyHostToDevice);
    cudaMemset(dC, 0, M*N*sizeof(float));

    dim3 block(BLOCK_SIZE, 1);
    dim3 grid((N+BN-1)/BN, (M+BM-1)/BM);

    matmul_wmma_correct<<<grid, block>>>(dA, dB, dC, M, N, K);
    cudaDeviceSynchronize();

    if (check) {
        cudaMemcpy(hC_gpu, dC, M*N*sizeof(float), cudaMemcpyDeviceToHost);
        printf("  Correctness: %s\n",
               verify(hC_ref, hC_gpu, M*N) ? "PASSED" : "FAILED");
    }

    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);
    cudaEventRecord(start);
    for (int i = 0; i < 10; i++)
        matmul_wmma_correct<<<grid, block>>>(dA, dB, dC, M, N, K);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms; cudaEventElapsedTime(&ms, start, stop); ms /= 10;
    double tflops = 2.0*M*N*K / (ms*1e-3) / 1e12;
    printf("  %5d×%5d: %6.2f ms  %6.1f TFLOPS  %5.1f%% TC peak\n",
           M, N, ms, tflops, tflops/330.0*100);

    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    delete[] hA_f; delete[] hB_f; delete[] hA_h; delete[] hB_h;
    if (check) { delete[] hC_ref; delete[] hC_gpu; }
}

int main() {
    printf("BM=%d BN=%d BK=%d  WARP_M=%d WARP_N=%d  "
           "BLOCK_WARP=%dx%d  threads=%d\n\n",
           BM, BN, BK, WARP_M, WARP_N,
           BLOCK_WARP_M, BLOCK_WARP_N, BLOCK_SIZE);

    bench(128,  128,  128,  true);
    bench(1024, 1024, 1024, false);
    bench(4096, 4096, 4096, false);

    printf("\nSimple WMMA reference:\n");
    printf("  1024×1024:  0.21 ms   10.2 TFLOPS   3.1%% TC peak\n");
    printf("  4096×4096: 12.30 ms   11.2 TFLOPS   3.4%% TC peak\n");
    return 0;
}
