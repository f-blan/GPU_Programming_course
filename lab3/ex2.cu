#include <stdio.h>
#include <stdlib.h>

#define N 32
#define PRINT 0
#define A 1
#define B 1
#define C 1

__global__ void kernel0(){
	unsigned int tid = (blockDim.y * threadIdx.y) + threadIdx.x;
	unsigned int xIdx = threadIdx.x;
	unsigned int yIdx = threadIdx.y;
	printf("%d: %d %d\n", tid, xIdx, yIdx);
}

__global__ void kernelA(float *d){
	unsigned int tid = (blockDim.y * threadIdx.y) + threadIdx.x;
	unsigned int i = threadIdx.x + blockDim.x * threadIdx.y;
	d[i]=1.0;
#if PRINT
	printf("%d: %d %1f\n",tid,i, d[i]);
#endif
}

__global__ void kernelB(float *d){
	unsigned int tid = (blockDim.y * threadIdx.y) + threadIdx.x;
	unsigned int i =threadIdx.y + blockDim.y * threadIdx.x;
	d[i]=1.0;
#if PRINT
	printf("%d: %d %1f\n",tid,i, d[i]);
#endif
}

__global__ void kernelC(float *d){
        unsigned int tid = (blockDim.y * threadIdx.y) + threadIdx.x;
	unsigned int i =1+threadIdx.x + blockDim.x * threadIdx.y;
	d[i]=1.0;

#if PRINT
	printf("%d: %d %1f\n",tid,i, d[i]);
#endif
}



int main(void){
	dim3 gridDim(1);
	dim3 blockDim(N,N,1);

	//kernel1<<<gridDim, blockDim>>>();
	float *d;
	cudaMalloc((void **)&d, N*N*sizeof(float));

#if A
	kernelA<<<gridDim, blockDim>>>(d);
#endif
#if B
	kernelB<<<gridDim,blockDim>>>(d);
#endif
#if C
	kernelC<<<gridDim,blockDim>>>(d);
#endif
	cudaFree(d);
	cudaDeviceReset();
}
