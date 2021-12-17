#include <stdio.h>
#define N 256


__global__ void parallel_sort(int *data, int num_elem){

	 unsigned int tid = (blockIdx.x * blockDim.x) + threadIdx.x; 
	  unsigned int tid_idx; 
	   unsigned int offset = 0; 
	    unsigned int d0, d1; 
	     unsigned int  tid_idx_max = (num_elem - 1); 
	     //printf("executing tid %d\n", tid);  
	      for (int i = 0; i < num_elem; i++){ 
		        tid_idx = (tid * 2) + offset; 
			  if (tid_idx < tid_idx_max){ 
				     d0 = data[tid_idx]; 
				        d1 = data[tid_idx + 1]; 
					   if (d0 > d1){ 
						//printf("%d thread swapped: %d %d\n",tid,d0,d1);
						       data[tid_idx] = d1; 
						           data[tid_idx + 1] = d0; 
							      } 
					     } 
			    if (offset == 0){ 
				       offset = 1; 
				         }
			      else{ 
				         offset = 0; 
					   }
			       }
} 

int main(void){
	 
	 printf("starting vectorAdd\n");
	  int *d;
	   int *dev_d;

	    //allocate and init cpu mem
	    d =(int*) malloc( N* sizeof(int));

	     int i =0;
	     printf("allocate\n");
	      for(int t=N-1; t>=0
			      ; --t){
		        d[i]=t;
			  printf("%d ", d[i]);
			  i++;
			   }
	       //allocate GPU mem
	       cudaMalloc((void**)&dev_d, N*sizeof(int));

	        //copy the cpu arrays into gpu ones
	        cudaMemcpy(dev_d, d, N*sizeof(int), cudaMemcpyHostToDevice);
		        
		 //exec
		 printf("executing\n");
                 int n_blocks = N/128;
		 if(n_blocks ==0){
			 n_blocks=1;
		 }
		 int n_threads = N/n_blocks;
		 n_threads = n_threads/2;
		 printf("using %d blocks and %d threads per block", 
				 n_blocks, n_threads);
		 parallel_sort<<<1,N>>>(dev_d, N);
		 
		 cudaMemcpy(d, dev_d, N*sizeof(int), cudaMemcpyDeviceToHost);
		 int bad=0;
		 //checks
		   printf("\ncheck\n");
	         for(int t=0; t<N && bad==0; ++t){
		    if(d[t]!=t)
			    bad=1;
	         }

		 if(bad==1){
			 for(int t=0;t<N;++t){
				 printf("%d ", d[t]);
			 }
		 }else{
			 printf("\nall good\n");
		 }
                   printf("\nend\n");
		    free(d);
		     cudaFree(dev_d);

}
