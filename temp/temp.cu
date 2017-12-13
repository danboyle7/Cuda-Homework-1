#include <iostream>
#include <cstdlib>
#include <iomanip>
#include <cstring>
#include <cuda_runtime.h>
#include <cstdio>

#define CUDA_WARN(XXX) \
      do { if (XXX != cudaSuccess) std::cerr << "CUDA Error: " << \
          cudaGetErrorString(XXX) << ", at line " << __LINE__ \
          << std::endl; cudaDeviceSynchronize(); } while (0)


int main() {
	double *h1 = new double[5];
	double *h2 = new double[5]; 
    double *d;

	h1[0] = 7;
	h1[1] = 4;	
	h1[2] = 1;
	h1[3] = 8;
	h1[4] = 9;
	
	std::cout << "1) Mallocing Space and copying from host to device" << std::endl;	
	CUDA_WARN(cudaMalloc(&d, 5*sizeof(double)));
	CUDA_WARN(cudaMemcpy(d, h1, 5*sizeof(double), cudaMemcpyHostToDevice));

	std::cout << "\n2) Copying back from device to host"  << std::endl;
	CUDA_WARN(cudaMemcpy(h2, d, 5*sizeof(double), cudaMemcpyDeviceToHost));

	std::cout << "\n3) Copied data (back on host): " << std::endl;
	for(int i = 0; i < 5; i++) {
		std::cout << h2[i] << std::endl;
	} 
 
	
	return 0;
}

