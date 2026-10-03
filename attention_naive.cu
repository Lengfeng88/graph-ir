#include <stdio.h>
#include <math.h>
#include <cuda_runtime.h>

// Naive attention: 完整实现 softmax(QK^T) @ V
// Q,K,V: [N, D]，FP32
// 显式写出 N×N 的 attention matrix S

__global__ void attention_naive(
    const float* Q,   // [N, D]
    const float* K,   // [N, D]
    const float* V,   // [N, D]
    float* O,         // [N, D]
    int N, int D)
{
    // 每个 thread 负责一个 query row
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= N) return;

    // Step 1: S[row, :] = Q[row] @ K^T，结果存在 register（N 个 float）
    // 注意：这里为了简单，用动态 array 是不行的（N 可变）
    // 实际实现要么用 smem，要么限制 N 大小
    // 这个版本先放到 global memory 里的 scratch space
    // 用 extern __shared__ 动态分配

    extern __shared__ float smem[];
    float* S_row = smem + threadIdx.x * N;  // 每个 thread 一行 S

    // Q[row] @ K^T
    float scale = 1.0f / sqrtf((float)D);
    for (int j = 0; j < N; j++) {
        float dot = 0.0f;
        for (int d = 0; d < D; d++)
            dot += Q[row * D + d] * K[j * D + d];
        S_row[j] = dot * scale;
    }

    // Step 2: softmax(S_row)
    float m = S_row[0];
    for (int j = 1; j < N; j++) m = fmaxf(m, S_row[j]);

    float sum = 0.0f;
    for (int j = 0; j < N; j++) {
        S_row[j] = expf(S_row[j] - m);
        sum += S_row[j];
    }
    for (int j = 0; j < N; j++) S_row[j] /= sum;

    // Step 3: O[row] = S_row @ V
    for (int d = 0; d < D; d++) {
        float acc = 0.0f;
        for (int j = 0; j < N; j++)
            acc += S_row[j] * V[j * D + d];
        O[row * D + d] = acc;
    }
}

// CPU reference
void attention_cpu(
    const float* Q, const float* K, const float* V,
    float* O, int N, int D)
{
    float scale = 1.0f / sqrtf((float)D);
    float* S = new float[N * N];

    // QK^T
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++) {
            float dot = 0;
            for (int d = 0; d < D; d++)
                dot += Q[i*D+d] * K[j*D+d];
            S[i*N+j] = dot * scale;
        }

    // softmax
    for (int i = 0; i < N; i++) {
        float m = S[i*N];
        for (int j = 1; j < N; j++) m = fmaxf(m, S[i*N+j]);
        float s = 0;
        for (int j = 0; j < N; j++) { S[i*N+j] = expf(S[i*N+j]-m); s += S[i*N+j]; }
        for (int j = 0; j < N; j++) S[i*N+j] /= s;
    }

    // PV
    for (int i = 0; i < N; i++)
        for (int d = 0; d < D; d++) {
            float acc = 0;
            for (int j = 0; j < N; j++) acc += S[i*N+j] * V[j*D+d];
            O[i*D+d] = acc;
        }
    delete[] S;
}

bool verify(const float* ref, const float* out, int n, float eps=1e-3f) {
    float max_err = 0;
    int bad = 0;
    for (int i = 0; i < n; i++) {
        float diff = fabsf(ref[i] - out[i]);
        float rel  = diff / (fabsf(ref[i]) + 1e-6f);
        if (rel > max_err) max_err = rel;
        if (rel > eps) bad++;
    }
    printf("  max_rel_err=%.6f  bad=%d/%d\n", max_err, bad, n);
    return bad == 0;
}

int main() {
    // 小矩阵验证
    const int N = 128, D = 64;
    const int threads = 32;   // threads per block，每 thread 一行
    // smem per block = threads * N * sizeof(float)
    const int smem_bytes = threads * N * sizeof(float);

    printf("N=%d D=%d  smem/block=%.1fKB\n\n", N, D, smem_bytes/1024.0f);

    float *hQ = new float[N*D], *hK = new float[N*D];
    float *hV = new float[N*D], *hO = new float[N*D];
    float *hO_ref = new float[N*D];

    srand(42);
    for (int i = 0; i < N*D; i++) {
        hQ[i] = ((float)rand()/RAND_MAX - 0.5f) * 0.1f;
        hK[i] = ((float)rand()/RAND_MAX - 0.5f) * 0.1f;
        hV[i] = ((float)rand()/RAND_MAX - 0.5f) * 0.1f;
    }

    attention_cpu(hQ, hK, hV, hO_ref, N, D);

    float *dQ, *dK, *dV, *dO;
    cudaMalloc(&dQ, N*D*sizeof(float));
    cudaMalloc(&dK, N*D*sizeof(float));
    cudaMalloc(&dV, N*D*sizeof(float));
    cudaMalloc(&dO, N*D*sizeof(float));
    cudaMemcpy(dQ, hQ, N*D*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dK, hK, N*D*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dV, hV, N*D*sizeof(float), cudaMemcpyHostToDevice);

    dim3 block(threads);
    dim3 grid((N + threads - 1) / threads);

    attention_naive<<<grid, block, smem_bytes>>>(dQ, dK, dV, dO, N, D);
    cudaDeviceSynchronize();

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
        return 1;
    }

    cudaMemcpy(hO, dO, N*D*sizeof(float), cudaMemcpyDeviceToHost);

    printf("Correctness (N=%d D=%d):\n", N, D);
    bool ok = verify(hO_ref, hO, N*D);
    printf("  Result: %s\n\n", ok ? "PASSED" : "FAILED");

    // 性能测试
    const int N_perf = 1024, D_perf = 64;
    const int smem_perf = threads * N_perf * sizeof(float);
    printf("Perf test N=%d D=%d  smem/block=%.1fKB\n",
           N_perf, D_perf, smem_perf/1024.0f);

    float *dQ2, *dK2, *dV2, *dO2;
    cudaMalloc(&dQ2, N_perf*D_perf*sizeof(float));
    cudaMalloc(&dK2, N_perf*D_perf*sizeof(float));
    cudaMalloc(&dV2, N_perf*D_perf*sizeof(float));
    cudaMalloc(&dO2, N_perf*D_perf*sizeof(float));

    dim3 block2(threads);
    dim3 grid2((N_perf + threads - 1) / threads);

    // warmup
    attention_naive<<<grid2, block2, smem_perf>>>(dQ2, dK2, dV2, dO2, N_perf, D_perf);
    cudaDeviceSynchronize();

    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);
    cudaEventRecord(start);
    for (int i = 0; i < 10; i++)
        attention_naive<<<grid2, block2, smem_perf>>>(
            dQ2, dK2, dV2, dO2, N_perf, D_perf);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms; cudaEventElapsedTime(&ms, start, stop); ms /= 10;

    // HBM traffic estimate: read Q+K+V, write O, write S (N×N)
    // S 是隐式的（在 smem），但逻辑上是 O(N²)
    double hbm_gb = (3.0*N_perf*D_perf*4 + N_perf*D_perf*4) / 1e9;
    printf("  Latency: %.3f ms\n", ms);
    printf("  (baseline for FlashAttention comparison)\n");

    cudaFree(dQ); cudaFree(dK); cudaFree(dV); cudaFree(dO);
    cudaFree(dQ2); cudaFree(dK2); cudaFree(dV2); cudaFree(dO2);
    delete[] hQ; delete[] hK; delete[] hV; delete[] hO; delete[] hO_ref;
    return 0;
}
