#include <cuda.h>
#include <cuda_runtime.h>
#include <time.h>
#include <iostream>

#define N 32*1024*1024
#define BLOCK_SIZE 256

__global__ void reduce_v3(float *g_idata,float *g_odata){
    __shared__ float sdata[BLOCK_SIZE];

    // each thread loads one element from global to shared mem
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*(blockDim.x*2) + threadIdx.x;
    sdata[tid] = g_idata[i] + g_idata[i + blockDim.x];
    __syncthreads();

    // do reduction in shared mem
    for(unsigned int s=blockDim.x/2; s>0; s >>= 1) {
        if (tid < s){
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // write result for this block to global mem
    if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

int main() {
    float *input_host = (float*)malloc(N*sizeof(float));
    float *input_device;
    cudaMalloc((void **)&input_device, N*sizeof(float));
    for (int i = 0; i < N; i++) input_host[i] = 2.0;
    cudaMemcpy(input_device, input_host, N*sizeof(float), cudaMemcpyHostToDevice);

    int32_t block_num = (N + BLOCK_SIZE - 1) / BLOCK_SIZE / 2;
    float *output_host = (float*)malloc((block_num) * sizeof(float));
    float *output_device;
    cudaMalloc((void **)&output_device, (block_num) * sizeof(float));
    
    dim3 grid(block_num, 1);
    dim3 block(BLOCK_SIZE, 1);
    reduce_v3<<<grid, block>>>(input_device, output_device);
    cudaMemcpy(output_host, output_device, block_num * sizeof(float), cudaMemcpyDeviceToHost);
    for (int i = 0; i < min(block_num, 30); ++i) {
        std::cout << output_host[i] << ' ';
    }
    std::cout << std::endl;

    return 0;
}
