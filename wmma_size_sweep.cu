#include <stdio.h>
#include <cuda_runtime.h>
#include <mma.h>
using namespace nvcuda::wmma;

#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16

__global__ void wmma_simple(
    const half* A, const half* B, float* C, int M, int N, int K)
{
    int c_row = blockIdx.y * WMMA_M;
    int c_col = blockIdx.x * WMMA_N;
    if (c_row >= M || c_col >= N) return;

    fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc;
    fill_fragment(acc, 0.0f);

    for (int k = 0; k + WMMA_K <= K; k += WMMA_K) {
        fragment<matrix_a, WMMA_M, WMMA_N, WMMA_K, half, row_major> a;
        fragment<matrix_b, WMMA_M, WMMA_N, WMMA_K, half, row_major> b;
        load_matrix_sync(a, A + c_row * K + k, K);
        load_matrix_sync(b, B + k * N + c_col, N);
        mma_sync(acc, a, b, acc);
    }
    store_matrix_sync(C + c_row * N + c_col, acc, N, mem_row_major);
}

void bench(int M, int N, int K) {
    half *dA, *dB; float *dC;
    cudaMalloc(&dA, M*K*sizeof(half));
    cudaMalloc(&dB, K*N*sizeof(half));
    cudaMalloc(&dC, M*N*sizeof(float));
    cudaMemset(dA, 0, M*K*sizeof(half));
    cudaMemset(dB, 0, K*N*sizeof(half));

    dim3 block(32, 1);
    dim3 grid((N+WMMA_N-1)/WMMA_N, (M+WMMA_M-1)/WMMA_M);

    // warmup
    wmma_simple<<<grid, block>>>(dA, dB, dC, M, N, K);
    cudaDeviceSynchronize();

    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);
    cudaEventRecord(start);
    for (int i = 0; i < 10; i++)
        wmma_simple<<<grid, block>>>(dA, dB, dC, M, N, K);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms; cudaEventElapsedTime(&ms, start, stop); ms /= 10;
    double tflops = 2.0*M*N*K / (ms*1e-3) / 1e12;
    double data_gb = (2.0*M*K + 2.0*K*N + 4.0*M*N) / 1e9;

    printf("  %5d×%5d: %6.2f ms  %6.1f TFLOPS  %4.1f%% TC  data=%.1fGB\n",
           M, N, ms, tflops, tflops/330.0*100, data_gb);

    cudaFree(dA); cudaFree(dB); cudaFree(dC);
}

int main() {
    printf("WMMA simple (1 warp/block, global memory direct)\n");
    printf("RTX 4080: TC peak=330 TFLOPS, L2=64MB, HBM BW=717 GB/s\n\n");
    bench(1024,  1024,  1024);
    bench(2048,  2048,  2048);
    bench(4096,  4096,  4096);
    bench(8192,  8192,  8192);
    return 0;
}
