#include <stdio.h>
#include <math.h>
#include <cuda_runtime.h>

#define Br 32    // query block size
#define Bc 32    // key/value block size

// FlashAttention FA2：online softmax，不写 N×N attention matrix
// Q,K,V,O: [N, D]，FP32
// 一个 block 处理 Q 的一个 row block [Br 行]
__global__ void flash_attention(
    const float* Q,   // [N, D]
    const float* K,   // [N, D]
    const float* V,   // [N, D]
    float* O,         // [N, D]
    int N, int D)
{
    // 这个 block 负责 Q 的哪些行
    int q_block = blockIdx.x;
    int q_start = q_block * Br;
    if (q_start >= N) return;

    int tid = threadIdx.x;  // 0..Br-1，每个 thread 负责一行

    // smem 布局：
    // smem_Q [Br][D]，smem_K [Bc][D]，smem_V [Bc][D]
    // smem_S [Br][Bc]
    extern __shared__ float smem[];
    float* smem_Q = smem;                          // Br*D
    float* smem_K = smem_Q + Br * D;              // Bc*D
    float* smem_V = smem_K + Bc * D;              // Bc*D
    float* smem_S = smem_V + Bc * D;              // Br*Bc

    // 每个 thread 的 online softmax 状态
    float m = -1e9f;   // running max
    float l = 0.0f;    // running sum of exp

    // 每个 thread 的 output accumulator，存在 register
    // D 最大 64，register 够放
    float acc[64] = {};   // O accumulator，最多 D=64

    // 加载 Q block 到 smem
    // thread tid 加载 Q[q_start + tid][:]
    {
        int q_row = q_start + tid;
        if (q_row < N) {
            for (int d = 0; d < D; d++)
                smem_Q[tid * D + d] = Q[q_row * D + d];
        } else {
            for (int d = 0; d < D; d++)
                smem_Q[tid * D + d] = 0.0f;
        }
    }
    __syncthreads();

    float scale = 1.0f / sqrtf((float)D);

    // 遍历所有 KV block
    int num_kv_blocks = (N + Bc - 1) / Bc;
    for (int kv = 0; kv < num_kv_blocks; kv++) {
        int kv_start = kv * Bc;

        // 加载 K block [Bc, D] 到 smem_K
        // Br 个 thread 分工：每个 thread 加载 Bc*D/Br 个元素
        for (int i = tid; i < Bc * D; i += Br) {
            int r = i / D, c = i % D;
            int kr = kv_start + r;
            smem_K[r * D + c] = (kr < N) ? K[kr * D + c] : 0.0f;
        }

        // 加载 V block [Bc, D] 到 smem_V
        for (int i = tid; i < Bc * D; i += Br) {
            int r = i / D, c = i % D;
            int vr = kv_start + r;
            smem_V[r * D + c] = (vr < N) ? V[vr * D + c] : 0.0f;
        }
        __syncthreads();

        // 计算 S[tid, :] = Q[tid] @ K^T * scale，结果存 smem_S[tid, :]
        for (int j = 0; j < Bc; j++) {
            float dot = 0.0f;
            for (int d = 0; d < D; d++)
                dot += smem_Q[tid * D + d] * smem_K[j * D + d];
            smem_S[tid * Bc + j] = dot * scale;
        }
        __syncthreads();

        // Online softmax update
        // m_new = max(m, rowmax(S[tid,:]))
        float m_new = m;
        for (int j = 0; j < Bc; j++)
            m_new = fmaxf(m_new, smem_S[tid * Bc + j]);

        // alpha = exp(m - m_new)：rescale 旧的 l 和 O
        float alpha = expf(m - m_new);

        // P[j] = exp(S[tid,j] - m_new)
        float P[Bc];
        float l_local = 0.0f;
        for (int j = 0; j < Bc; j++) {
            P[j] = expf(smem_S[tid * Bc + j] - m_new);
            l_local += P[j];
        }

        // 更新 l 和 O
        l = alpha * l + l_local;

        // O = alpha * O + P @ V
        for (int d = 0; d < D; d++) {
            acc[d] *= alpha;
            float pv = 0.0f;
            for (int j = 0; j < Bc; j++)
                pv += P[j] * smem_V[j * D + d];
            acc[d] += pv;
        }

        m = m_new;
        __syncthreads();
    }

    // 归一化并写回
    int q_row = q_start + tid;
    if (q_row < N) {
        for (int d = 0; d < D; d++)
            O[q_row * D + d] = acc[d] / l;
    }
}

// CPU reference（和之前一样）
void attention_cpu(
    const float* Q, const float* K, const float* V,
    float* O, int N, int D)
{
    float scale = 1.0f / sqrtf((float)D);
    float* S = new float[N * N];
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++) {
            float dot = 0;
            for (int d = 0; d < D; d++) dot += Q[i*D+d] * K[j*D+d];
            S[i*N+j] = dot * scale;
        }
    for (int i = 0; i < N; i++) {
        float m = S[i*N];
        for (int j = 1; j < N; j++) m = fmaxf(m, S[i*N+j]);
        float s = 0;
        for (int j = 0; j < N; j++) { S[i*N+j] = expf(S[i*N+j]-m); s += S[i*N+j]; }
        for (int j = 0; j < N; j++) S[i*N+j] /= s;
    }
    for (int i = 0; i < N; i++)
        for (int d = 0; d < D; d++) {
            float acc = 0;
            for (int j = 0; j < N; j++) acc += S[i*N+j] * V[j*D+d];
            O[i*D+d] = acc;
        }
    delete[] S;
}

bool verify(const float* ref, const float* out, int n) {
    float max_err = 0; int bad = 0;
    for (int i = 0; i < n; i++) {
        float diff = fabsf(ref[i] - out[i]);
        float rel  = diff / (fabsf(ref[i]) + 1e-6f);
        if (rel > max_err) max_err = rel;
        if (rel > 1e-3f) bad++;
    }
    printf("  max_rel_err=%.6f  bad=%d/%d\n", max_err, bad, n);
    return bad == 0;
}

void bench(int N, int D) {
    // smem: Q[Br*D] + K[Bc*D] + V[Bc*D] + S[Br*Bc]，全 float
    int smem_bytes = (Br*D + Bc*D + Bc*D + Br*Bc) * sizeof(float);
    printf("N=%d D=%d  smem/block=%.1fKB\n", N, D, smem_bytes/1024.0f);

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

    dim3 block(Br);
    dim3 grid((N + Br - 1) / Br);

    flash_attention<<<grid, block, smem_bytes>>>(dQ, dK, dV, dO, N, D);
    cudaDeviceSynchronize();

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("  CUDA error: %s\n", cudaGetErrorString(err));
        return;
    }

    cudaMemcpy(hO, dO, N*D*sizeof(float), cudaMemcpyDeviceToHost);
    printf("Correctness:\n");
    bool ok = verify(hO_ref, hO, N*D);
    printf("  %s\n", ok ? "PASSED" : "FAILED");

    // 性能
    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);
    cudaEventRecord(start);
    for (int i = 0; i < 10; i++)
        flash_attention<<<grid, block, smem_bytes>>>(dQ, dK, dV, dO, N, D);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms; cudaEventElapsedTime(&ms, start, stop); ms /= 10;

    // HBM traffic：读 Q,K,V 各 N*D，写 O N*D，不读写 N×N S
    double hbm_read_gb  = 3.0 * N * D * sizeof(float) / 1e9;
    double hbm_write_gb = 1.0 * N * D * sizeof(float) / 1e9;
    double hbm_total_gb = hbm_read_gb + hbm_write_gb;
    double hbm_bw = hbm_total_gb / (ms * 1e-3);  // GB/s

    printf("Perf: %.3f ms  HBM_traffic=%.3f GB  eff_BW=%.1f GB/s\n\n",
           ms, hbm_total_gb, hbm_bw);

    cudaFree(dQ); cudaFree(dK); cudaFree(dV); cudaFree(dO);
    delete[] hQ; delete[] hK; delete[] hV; delete[] hO; delete[] hO_ref;
}

int main() {
    printf("Br=%d Bc=%d\n\n", Br, Bc);
    bench(128,  64);
    bench(512,  64);
    bench(1024, 64);
    bench(2048, 64);
    return 0;
}
