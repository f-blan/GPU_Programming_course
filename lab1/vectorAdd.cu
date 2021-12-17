#include <stdio.h>
#define N 2048

__global__ void vectorAdd(int *a, int *b, int *c){
	int tid = blockDim.x * blockIdx.x + threadIdx.x;

	c[tid]=a[tid]+b[tid];
	//printf("tid: %d done, res: %d\n",tid, c[tid]);

}

int main(void){
	printf("starting vectorAdd\n");
	int *a, *b, *c;
	int *dev_a, *dev_b, *dev_c;

	//allocate and init cpu mem
	a =(int*) malloc( N* sizeof(int));
	b =(int*) malloc( N* sizeof(int));
	c =(int*) malloc( N* sizeof(int));

	for(int t=0; t<N; ++t){
		a[t]=t;
		b[t]=10;
		c[t]=0;
	}

	//allocate GPU mem
	cudaMalloc((void**)&dev_a, N*sizeof(int));
	cudaMalloc((void**)&dev_b, N*sizeof(int));
	cudaMalloc((void**)&dev_c, N*sizeof(int));

	//copy the cpu arrays into gpu ones
	cudaMemcpy(dev_a, a, N*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_b, b, N*sizeof(int), cudaMemcpyHostToDevice);

	//exec
	int n_blocks = N/1024;
	if(n_blocks ==0){
		n_blocks = 1;
	}
	vectorAdd<<<n_blocks,N/n_blocks>>>(dev_a, dev_b, dev_c);

	//copy in c
	cudaMemcpy(c, dev_c, N*sizeof(int), cudaMemcpyDeviceToHost);

	//check
	printf("check\n");
	int bad=0;
	for(int t=0; t<N && bad==0; ++t){
		if(c[t]!=a[t]+b[t]){
			printf("bad\n");
			bad=1;
		}
	}
        printf("\nend\n");
	//frees
	free(a);
	free(b);
	free(c);

	cudaFree(dev_a);
	cudaFree(dev_b);
	cudaFree(dev_c);
	

	cudaDeviceReset();
	return 0;
}
