#include <stdio.h>
#define N 1024


__global__ void mean_0(float *input,float *mean_output, 
		int total_elements){

	 unsigned int tid = (blockIdx.x * blockDim.x) + threadIdx.x; 
	
	 if(tid ==0 || tid == total_elements-1){
		 mean_output[tid]=-1;
		 return;
	 }

	 float i0 = input[tid-1];
	 float i1 = input[tid];
	 float i2 = input[tid+1];

	 mean_output[tid] = (i0+i1+i2)/3;

	 printf("tid %d computed %f\n",tid, mean_output[tid]);


	 
} 

__global__ void mean_1(float *i, float *o, int n){
	unsigned int tid = (blockIdx.x*blockDim.x) + threadIdx.x;

	if(tid ==0 || tid == n/2){
		return;
	}
	int ind = (tid-1)*2+1;
	//printf("tid %d taking care of %d and %d \n",tid, ind, ind+1);
	o[ind]= (i[ind]+i[ind-1]+i[ind+1])/3;
	ind = ind+1;
        o[ind]=(i[ind]+i[ind+1]+i[ind-1])/3;
	
	//printf("tid %d computed %f and %f\n", tid, o[ind-1], o[ind]); 	

}

int main(void){
	 
	 printf("starting vectorAdd\n");
	 float *i,*o;
	 float *dev_i, *dev_o;

	 //allocate and init cpu mem
	 i =(float*) malloc( N* sizeof(float));
	 o =(float*) malloc(N*sizeof(float));

	 int v =0;
	 printf("allocate\n");
	 for(int t=0; t<N; ++t){
	 	i[v]=2;
		//printf("%d ", d[i]);
		v++;
	}
	//allocate GPU mem
	cudaMalloc((void**)&dev_i, N*sizeof(float));
	cudaMalloc((void**)&dev_o, N*sizeof(float));

	//copy the cpu arrays into gpu ones
	cudaMemcpy(dev_i, i, N*sizeof(float), cudaMemcpyHostToDevice);
		        
	//exec
	printf("executing\n");
        int n_blocks = N/128;
	if(n_blocks ==0){
		n_blocks=1;
	}
	int n_threads = N/n_blocks;
	n_threads = n_threads/2;
	printf("using %d blocks and %d threads per block\n", 
				 n_blocks, n_threads);
	mean_1<<<1,N/2>>>(dev_i,dev_o, N);
		 
	cudaMemcpy(o, dev_o, N*sizeof(float), cudaMemcpyDeviceToHost);
	int bad=0;
	//checks
	printf("\ncheck\n");
	for(int t=1; t<N-1 && bad==0; ++t){
		if(o[t]!=2)
		bad=1;
	}

	if(bad==1){
		for(int t=0;t<N;++t){
			printf("%f ", o[t]);
		}
	 }else{
		printf("\nall good\n");
	 }
         printf("\nend\n");
	 free(i);
	 free(o);
	 cudaFree(dev_i);
	 cudaFree(dev_o);

}
