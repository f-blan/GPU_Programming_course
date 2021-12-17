#include <stdio.h>

__global__ void helloFromGPU(void){
	printf("hello from jetson GPU\n");

}

int main(void){
	printf("Hello from jetson CCCCCCC\n");
	helloFromGPU<<<1,10>>>();
	cudaDeviceReset();
	return 0;
}
