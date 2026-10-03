#include <stdio.h>
#include <math.h>
#include <cuda_runtime.h>

#define Br 16    // query rows per block
#define Bc 16    // kv cols per block
#define D  64    // head dimension，固定
#define WARP_SIZE 32

// warp-level reduction：求 warp 内所有 lane 的 sum
__device__ float warp_reduce_sum(float val) {
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

// warp-level reduction：求 max
__device__ float warp_reduce_max(float val) {
    for (int offset = 16; offset > 0; offset >>= 1)
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    return val;
}

// FA v2：1 warp per query row，warp 内并行做 dot product
// block = Br warps × 32 threads = Br × 32 threads
// 每个 warp 负责 1 个 query row
__global__ void flash_attention_v2(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    float* __restrict__ O,
    int N)
{
    // 这个 warp 的 ID 和 lane
    int warp_id = threadIdx.x / WARP_SIZE;   // 0..Br-1
    int lane    = threadIdx.x % WARP_SIZE;   // 0..31

    // 这个 warp 负责的 query row
    int q_row = blockIdx.x * Br + warp_id;

    // smem 布局
    extern __shared__ float smem[];
    // smem_Q: [Br][D]
    // smem_K: [Bc][D]
    // smem_V: [Bc][D]
    // smem_S: [Br][Bc]
    float* smem_Q = smem;
    float* smem_K = smem_Q + Br * D;
    float* smem_V = smem_K + Bc * D;
    float* smem_S = smem_V + Bc * D;

    // 加载 Q[q_row] 到 smem_Q[warp_id]
    // lane 0..31 并行加载 D=64 个元素，每 lane 加载 2 个
    if (q_row < N) {
        for (int d = lane; d < D; d += WARP_SIZE)
            smem_Q[warp_id * D + d] = Q[q_row * D + d];
    } else {
        for (int d = lane; d < D; d += WARP_SIZE)
            smem_Q[warp_id * D + d] = 0.0f;
    }

    // online softmax 状态（每个 warp 独立，存在 lane 0 的 register）
    float m = -1e9f;
    float l = 0.0f;
    float acc[D / WARP_SIZE + 1];   // 每个 lane 负责 D/32 = 2 个输出维度
    // lane i 负责 O[q_row][i], O[q_row][i+32]
    for (int i = 0; i < D / WARP_SIZE; i++) acc[i] = 0.0f;

    float scale = 1.0f / sqrtf((float)D);

    __syncthreads();  // 等 Q 加载完

    int num_kv = (N + Bc - 1) / Bc;

    for (int kv = 0; kv < num_kv; kv++) {
        int kv_start = kv * Bc;

        // 加载 K,V block 到 smem
        // block 内所有 threads (Br*32) 协作加载 Bc*D 个元素
        int total_threads = Br * WARP_SIZE;
        int tid = threadIdx.x;

        for (int i = tid; i < Bc * D; i += total_threads) {
            int r = i / D, c = i % D;
            int kr = kv_start + r;
            smem_K[r * D + c] = (kr < N) ? K[kr * D + c] : 0.0f;
            smem_V[r * D + c] = (kr < N) ? V[kr * D + c] : 0.0f;
        }
        __syncthreads();

        // 每个 warp 计算 S[warp_id, 0:Bc]
        // S[warp_id, j] = Q[warp_id] · K[j]
        // lane 并行做 dot product：每个 lane 算部分 sum，然后 warp reduce
        for (int j = 0; j < Bc; j++) {
            float partial = 0.0f;
            for (int d = lane; d < D; d += WARP_SIZE)
                partial += smem_Q[warp_id * D + d] * smem_K[j * D + d];
            // warp reduce sum
            partial = warp_reduce_sum(partial);
            // lane 0 写结果
            if (lane == 0)
                smem_S[warp_id * Bc + j] = partial * scale;
        }
        // 不需要 __syncthreads，smem_S 只被本 warp 读写

        // Online softmax（只有 lane 0 执行，其他 lane 参与 PV 计算）
        // 先让所有 lane 计算各自负责的 P[j]，然后 reduce

        // 找 row max：lane 并行
        float m_new = -1e9f;
        for (int j = lane; j < Bc; j += WARP_SIZE)
            m_new = fmaxf(m_new, smem_S[warp_id * Bc + j]);
        m_new = warp_reduce_max(m_new);
        // broadcast m_new 给所有 lane
        m_new = __shfl_sync(0xffffffff, m_new, 0);

        // P[j] = exp(S[j] - m_new)，l_local = sum(P)
        float P[Bc];   // 每个 lane 存完整 P（Bc=16，可以放进 register）
        float l_local_partial = 0.0f;
        for (int j = 0; j < Bc; j++) {
            P[j] = expf(smem_S[warp_id * Bc + j] - m_new);
        }
        // lane 并行求 sum
        for (int j = lane; j < Bc; j += WARP_SIZE)
            l_local_partial += P[j];
        float l_local = warp_reduce_sum(l_local_partial);
        l_local = __shfl_sync(0xffffffff, l_local, 0);

        // 更新 l, O
        float alpha = expf(m - m_new);
        l = alpha * l + l_local;

        // O 更新：lane i 负责维度 i 和 i+32
        // O[d] = alpha * O[d] + sum_j(P[j] * V[j][d])
        for (int di = 0; di < D / WARP_SIZE; di++) {
            int d = lane + di * WARP_SIZE;  // 实际维度索引
            float pv = 0.0f;
            for (int j = 0; j < Bc; j++)
                pv += P[j] * smem_V[j * D + d];
            acc[di] = alpha * acc[di] + pv;
        }

        m = m_new;
        __syncthreads();  // 下一轮 KV block 加载前同步
    }

    // 写回：lane i 负责维度 i, i+32
    if (q_row < N) {
        for (int di = 0; di < D / WARP_SIZE; di++) {
            int d = lane + di * WARP_SIZE;
            O[q_row * D + d] = acc[di] / l;
        }
    }
}

void attention_cpu(const float* Q, const float* K, const float* V,
                   float* O, int N, int Dh) {
    float scale = 1.0f / sqrtf((float)Dh);
    float* S = new float[N * N];
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++) {
            float dot = 0;
            for (int d = 0; d < Dh; d++) dot += Q[i*Dh+d] * K[j*Dh+d];
            S[i*N+j] = dot * scale;
        }
    for (int i = 0; i < N; i++) {
        float mv = S[i*N];
        for (int j = 1; j < N; j++) mv = fmaxf(mv, S[i*N+j]);
        float s = 0;
        for (int j = 0; j < N; j++) { S[i*N+j] = expf(S[i*N+j]-mv); s += S[i*N+j]; }
        for (int j = 0; j < N; j++) S[i*N+j] /= s;
    }
    for (int i = 0; i < N; i++)
        for (int d = 0; d < Dh; d++) {
            float acc = 0;
            for (int j = 0; j < N; j++) acc += S[i*N+j] * V[j*Dh+d];
            O[i*Dh+d] = acc;
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
    printf("  max_rel_err=%.6f  bad=%d/%d  %s\n",
           max_err, bad, n, bad==0?"PASSED":"FAILED");
    return bad == 0;
}

void bench(int N) {
    int smem_bytes = (Br*D + Bc*D + Bc*D + Br*Bc) * sizeof(float);
    printf("N=%d D=%d Br=%d Bc=%d  smem=%.1fKB  threads/block=%d\n",
           N, D, Br, Bc, smem_bytes/1024.0f, Br*WARP_SIZE);

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

    dim3 block(Br * WARP_SIZE);
    dim3 grid((N + Br - 1) / Br);

    flash_attention_v2<<<grid, block, smem_bytes>>>(dQ, dK, dV, dO, N);
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) { printf("  error: %s\n", cudaGetErrorString(err)); return; }

    cudaMemcpy(hO, dO, N*D*sizeof(float), cudaMemcpyDeviceToHost);
    verify(hO_ref, hO, N*D);

    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);
    cudaEventRecord(start);
    for (int i = 0; i < 10; i++)
        flash_attention_v2<<<grid, block, smem_bytes>>>(dQ, dK, dV, dO, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms; cudaEventElapsedTime(&ms, start, stop); ms /= 10;

    // 和 v1 对比
    double flops = 2.0 * N * N * D + 2.0 * N * N * D;  // QK^T + PV
    printf("  %.3f ms  (FA v1 reference below)\n\n", ms);

    cudaFree(dQ); cudaFree(dK); cudaFree(dV); cudaFree(dO);
    delete[] hQ; delete[] hK; delete[] hV; delete[] hO; delete[] hO_ref;
}

int main() {
    bench(128);
    bench(512);
    bench(1024);
    bench(2048);

    printf("FA v1 reference:\n");
    printf("  N=128:  0.452 ms\n");
    printf("  N=512:  1.807 ms\n");
    printf("  N=1024: 3.558 ms\n");
    printf("  N=2048: 7.157 ms\n");
    return 0;
}
