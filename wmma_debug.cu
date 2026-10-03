#include <stdio.h>
#include <cuda_runtime.h>
#include <mma.h>
using namespace nvcuda::wmma;

// 最小案例：M=N=K=16，1 个 block，1 个 warp
// 直接从 global memory 读，不用 smem
__global__ void wmma_16x16(
    const half* A, const half* B, float* C, int N)
{
    fragment<matrix_a, 16, 16, 16, half, row_major> a;
    fragment<matrix_b, 16, 16, 16, half, row_major> b;
    fragment<accumulator, 16, 16, 16, float> acc;
    fill_fragment(acc, 0.0f);
    load_matrix_sync(a, A, 16);   // leading dim = K = 16
    load_matrix_sync(b, B, 16);   // leading dim = N = 16
    mma_sync(acc, a, b, acc);
    store_matrix_sync(C, acc, 16, mem_row_major);
}

int main() {
    const int S = 16;
    float hA[S*S], hB[S*S], hC[S*S], hC_ref[S*S];
    half hA_h[S*S], hB_h[S*S];

    srand(42);
    for (int i = 0; i < S*S; i++) {
        hA[i] = (float)rand()/RAND_MAX * 0.1f;
        hB[i] = (float)rand()/RAND_MAX * 0.1f;
        hA_h[i] = __float2half(hA[i]);
        hB_h[i] = __float2half(hB[i]);
    }

    // CPU ref
    for (int i = 0; i < S; i++)
        for (int j = 0; j < S; j++) {
            float s = 0;
            for (int k = 0; k < S; k++) s += hA[i*S+k] * hB[k*S+j];
            hC_ref[i*S+j] = s;
        }

    half *dA, *dB; float *dC;
    cudaMalloc(&dA, S*S*sizeof(half));
    cudaMalloc(&dB, S*S*sizeof(half));
    cudaMalloc(&dC, S*S*sizeof(float));
    cudaMemcpy(dA, hA_h, S*S*sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB_h, S*S*sizeof(half), cudaMemcpyHostToDevice);
    cudaMemset(dC, 0, S*S*sizeof(float));

    wmma_16x16<<<1, 32>>>(dA, dB, dC, S);
    cudaDeviceSynchronize();
    cudaMemcpy(hC, dC, S*S*sizeof(float), cudaMemcpyDeviceToHost);

    int bad = 0;
    float max_err = 0;
    for (int i = 0; i < S*S; i++) {
        float diff = fabsf(hC_ref[i] - hC[i]);
        float rel = diff / (fabsf(hC_ref[i]) + 1e-6f);
        if (rel > max_err) max_err = rel;
        if (diff > 0.05f * fabsf(hC_ref[i]) + 0.01f) bad++;
    }
    printf("16x16 WMMA: %s (max_rel_err=%.4f, bad=%d/256)\n",
           bad==0?"PASSED":"FAILED", max_err, bad);

    // 现在测 smem 路径：32×32×16，2 warps
    // warp 0 → C[0:16][0:16]，warp 1 → C[0:16][16:32]
    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    return 0;
}
