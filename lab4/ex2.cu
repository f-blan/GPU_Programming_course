#include <stdio.h>
#include <stdlib.h>

#define N_DIM 64
#define M_DIM 6
#define SHARED 1

__constant__ float Mask[M_DIM];

__global__ void conv_1(float *M, float *P, float *N, int Mask_Width, int vector_Width){
	int tid= blockIdx.x * blockDim.x +threadIdx.x;

	float Pvalue = 0;

	int N_start_point = tid - (Mask_Width/2);

	for(int t=0; t<Mask_Width; ++t){
		if(N_start_point +t >=0 && N_start_point +t < vector_Width){
			Pvalue+= Mask[t] * N[N_start_point+t];
		}
	}
	P[tid] = Pvalue;
}

__global__ void conv_1_s(float *M, float *P, float *N, int Mask_Width, int vector_Width){
	int tid= blockIdx.x * blockDim.x +threadIdx.x;
	
	extern __shared__ float s[];
	
	float Pvalue = 0;

	s[threadIdx.x] = N[tid];
	int base = tid - threadIdx.x;
	int roof = tid - threadIdx.x + blockDim.x;
	
	__syncthreads();

	int N_start_point = tid - (Mask_Width/2);
	int N_start_point_shared = threadIdx.x - (Mask_Width/2);

	for(int t=0; t<Mask_Width; ++t){
		if(N_start_point +t >=0 && N_start_point +t < vector_Width){
			if(N_start_point_shared +t >=base && N_start_point_shared + t<roof){

				Pvalue+= Mask[t] * s[N_start_point_shared+t];
			}else{
				Pvalue+= Mask[t] * N[N_start_point+t];
			}		
		}
	}
	P[tid] = Pvalue;
}


int main(void){
	float * N, *M, *P;
	float *d_N, *d_M, *d_P;

	N = (float *) malloc(N_DIM*sizeof(float));
	M = (float *) malloc(M_DIM*sizeof(float));
	P = (float *) malloc(N_DIM*sizeof(float));


	
	for(int t =0; t<N_DIM; ++t){
		N[t] =1;
	}
	for(int t =0; t<M_DIM; ++t){
		M[t] =1;
	}

	cudaMemcpyToSymbol(Mask, M, sizeof(float)*M_DIM);

	cudaMalloc((void**)&d_N, N_DIM*sizeof(float));
	cudaMalloc((void**)&d_M, M_DIM*sizeof(float));
	cudaMalloc((void**)&d_P, N_DIM*sizeof(float));

	cudaMemcpy(d_N, N, N_DIM*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_M, M, M_DIM*sizeof(float), cudaMemcpyHostToDevice);
	
	//int threads = 512;
	//int blocks = N_DIM/threads +1;
	int threads = 8;
	int blocks = 8;
#if SHARED
	printf("shared\n");
	conv_1_s<<<blocks,threads, (threads)*sizeof(float)>>>(d_M, d_P, d_N, M_DIM, N_DIM);
#else
	printf("not shared\n");
	conv_1<<<blocks,threads>>>(d_M, d_P, d_N, M_DIM, N_DIM);
#endif
	cudaMemcpy(P, d_P, N_DIM*sizeof(float), cudaMemcpyDeviceToHost);
	for(int t=0; t<N_DIM; ++t){
		printf("%.2f\n", P[t]);

	}

	cudaFree(d_N);
	cudaFree(d_M);
	cudaFree(d_P);

	free(N);
	free(M);
	free(P);

}
