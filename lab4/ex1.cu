#include <stdlib.h>
#include <stdio.h>

#define N 	4
#define BLOCKS	1
#define THREADS	4
#define PRINT	0
#define VERSION 0

__global__ void simple_kernel0(float *x, float *y){
	int i = threadIdx.x + blockDim.x*blockIdx.x;
	
	__syncthreads();
	if((i%blockDim.x) == 0){
		x[i] = x[i] +1;	
	}
	
	if((i%blockDim.x) == 1){
		y[i] = x[i-1]/2.0;	
	}
	
	if((i%blockDim.x) == 2){
		x[i-1] = y[i-1]*y[i-1];	
	}
	
	if((i%blockDim.x) == 3){
		y[i-1] = x[i-2]*y[i-2] + x[i-3];	
	}
}

__global__ void simple_kernel1(float *x, float *y){
	int i = threadIdx.x + blockDim.x*blockIdx.x;
	//printf("%d\n", i);
	if((i%blockDim.x) == 0){
#if PRINT
		printf("fetching 0\n");
#endif

		float xi = x[i];
		__syncthreads();

#if PRINT
		printf("calc 0\n");
#endif
		x[i] = xi +1;	
	}
	
	if((i%blockDim.x) == 1){
#if PRINT
		printf("fetching 1\n");
#endif
		float xim1 = x[i-1];
		__syncthreads();
#if PRINT
		printf("calc 1\n");
#endif
		y[i] = xim1/2.0;	
	}
	
	if((i%blockDim.x) == 2){
#if PRINT
		printf("fetching 2\n");
#endif
		float yim1 = y[i-1];
		__syncthreads();
#if PRINT
		printf("calc 2\n");
#endif
		x[i-1] = yim1*yim1;	
	}
	
	if((i%blockDim.x) == 3){
#if PRINT
		printf("fetching 3\n");
#endif
		float xim2 = x[i-2];
		float yim2 = y[i-2];
		float xim3 = x[i-3];
		__syncthreads();
#if PRINT
		printf("calc 3\n");
#endif
		y[i-1] = xim2*yim2 + xim3;	
	}
}

__global__ void simple_kernel2(float *x, float *y){
	int i = threadIdx.x + blockDim.x*blockIdx.x;
	float a;
	if((i%blockDim.x) == 0){
		a = x[i] +1;	
	}
	
	if((i%blockDim.x) == 1){
		a = x[i-1]/2.0;	
	}
	
	if((i%blockDim.x) == 2){
		a = y[i-1]*y[i-1];	
	}
	
	if((i%blockDim.x) == 3){
		a = x[i-2]*y[i-2] + x[i-3];	
	}
	__syncthreads();
	if((i%blockDim.x) == 0){
		x[i] = a;	
	}
	
	if((i%blockDim.x) == 1){
		y[i] = a;	
	}
	
	if((i%blockDim.x) == 2){
		x[i-1] = a;	
	}
	
	if((i%blockDim.x) == 3){
		y[i-1] = a;	
	}
}

int main(void){
	float * x, *y;
	float *d_x, *d_y;

	x = (float *) malloc(N*sizeof(float));
	y = (float *) malloc(N*sizeof(float));
	
	for(int t =0; t<N; ++t){
		x[t] =t;
		y[t] = t;
	}

	cudaMalloc((void**)&d_x, N*sizeof(float));
	cudaMalloc((void**)&d_y, N*sizeof(float));
	
	cudaMemcpy(d_x, x, N*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_y, y, N*sizeof(float), cudaMemcpyHostToDevice);

#if VERSION
	printf("no sync\n");
	simple_kernel0<<<BLOCKS, THREADS>>>(d_x, d_y);
#else
	printf("with sync\n");
	simple_kernel2<<<BLOCKS, THREADS>>>(d_x, d_y);
#endif

	cudaMemcpy(x, d_x, N*sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(y, d_y, N*sizeof(float), cudaMemcpyDeviceToHost);
	

	for(int t =0; t<N; ++t){
		printf("x[%d] =%.2f - y[%d] = %.2f\n", t, x[t],t, y[t]);
	}


	cudaFree(d_x);
	cudaFree(d_y);

	free(x);
	free(y);

}
