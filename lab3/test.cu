#include <stdio.h>
#include <stdlib.h>

__global__ void init_shared(void){
	printf("hi\n");

}

int main(void){
	init_shared<<<2,32>>>();
	cudaDeviceReset();
}
