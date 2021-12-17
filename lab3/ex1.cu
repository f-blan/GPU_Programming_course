#include <stdlib.h>
#include <stdio.h>

#define N 1048576
#define MODE 0
#define PRINT 0
#define M 0

__global__ void init_global(double *d){
	unsigned int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
	
	d[tid] = (double) tid;
	for(int t=0;t<M;++t){
		d[tid] += (double) tid; 
		
	}

#if PRINT
	if(tid>0)
		printf("%d : %2f\n", tid, d[tid]);
#endif
}

__global__ void init_shared(void){
	
	extern __shared__ double s[];
	unsigned int tid = (blockIdx.x * blockDim.x) + threadIdx.x;
	unsigned int tIdx = threadIdx.x;
	
	s[tIdx] = (double) tid;
	for(int t=0; t<M; ++t){
		s[tIdx] = (double) tid;
	}
#if PRINT	
	if(tIdx>0)
		printf("%d : %d : %2f\n", tid,tIdx, s[tIdx-1]);
#endif
	
	
}



int main(void){
	int blocks = N/32;
#if MODE
	printf("executing global\n");
	double *d_g;

	//g = (double *) malloc(N*sizeof(double));
	cudaMalloc((void**)&d_g, N*sizeof(double));

	init_global<<<blocks, 32>>>(d_g);

	cudaFree(d_g);
	printf("global version executed\n");
#else
	printf("executing shared\n");

	init_shared<<<blocks, 32, 32*sizeof(double)>>>();
	printf("shared version executed\n");
#endif


	cudaDeviceReset();
}
