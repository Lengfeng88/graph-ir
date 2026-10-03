#include <stdio.h>
#include <cuda_runtime.h>
int main() {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("Max smem per block:    %zu KB\n", prop.sharedMemPerBlock/1024);
    printf("Max smem per SM:       %zu KB\n", prop.sharedMemPerMultiprocessor/1024);
    printf("Max threads per block: %d\n", prop.maxThreadsPerBlock);
    printf("SM count:              %d\n", prop.multiProcessorCount);
    return 0;
}
