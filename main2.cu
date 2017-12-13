#include <iostream>
#include <cstdlib>
#include <iomanip>
#include <cstring>
#include <cuda_runtime.h>
#include <cstdio>

//C++ timers
#include <chrono>

#define BLOCK_SIZE 16

#define CUDA_WARN(XXX) \
    do { if (XXX != cudaSuccess) std::cerr << "CUDA Error: " << \
        cudaGetErrorString(XXX) << ", at line " << __LINE__ \
        << std::endl; cudaDeviceSynchronize(); } while (0)

typedef std::chrono::high_resolution_clock Clock;
typedef std::chrono::nanoseconds nanoseconds;

typedef struct {
	
	//Dimensions of A
	int M;
	int N;

	//Scalars
	double alpha;
	double beta;
	
	//First dimension of A 
	int lda;

	//Increments for X and Y
	int incX;
	int incY;

	//Array A and vectors X, Y;
	double *A, *X, *Y;
} dgemv_data;


// BASIC DGEMM METHOD
void dgemm(int N,       double alpha,
		   double *X,   double *Y,
           double beta, double *A) {

 	
  	for (int i = 0; i < N; ++i) {
    	for (int j = 0; j < N; ++j) {
            double tmp = 0;
            for (int k = 0; k < N; ++k) {
                tmp += X[i * N + k] * Y[k * N + j];
            }
            A[i * N + j] = alpha * tmp + beta * A[i * N + j];
        }
    }
}

 
// CUDA DGEMM W/O SHARED MEMORY
__global__ void dgemm_cuda (int N,       double alpha,
						    double *X,   double *Y,
							double beta, double *A) {

  	int i = blockDim.y * blockIdx.y + threadIdx.y;
  	int j = blockDim.x * blockIdx.x + threadIdx.x;
  
  	if (i < N && j < N) {
  
  		double temp = 0;
  		for (int k = 0; k < N; ++k) {
    		temp += X[i * N + k] * Y[k * N + j];
  		}
    
    	A[i * N + j] = alpha * temp + beta * A[i * N + j];
	}
    
}

 
// CUDA DGEMM W/O SHARED MEMORY
__global__ void dgemm_cuda_shared (int N,       double alpha,
						    	   double *X,   double *Y,
								   double beta, double *A) {

	// Create Shared Memory Arrays
    __shared__ double Xshared[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ double Yshared[BLOCK_SIZE][BLOCK_SIZE];

    //Setup i and j
    int i = blockIdx.y * BLOCK_SIZE + threadIdx.y;
    int j = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    double temp = 0;

    for (int s = 0; s < gridDim.x; ++s) {

        int index = i * N + s * BLOCK_SIZE + threadIdx.x;
        if(index >= N*N) 
            Xshared[threadIdx.y][threadIdx.x] = 0;
        else 
            Xshared[threadIdx.y][threadIdx.x] = X[index];
        

        index = (s * BLOCK_SIZE + threadIdx.y) * N + j;
        if(index >= N*N) 
            Yshared[threadIdx.y][threadIdx.x] = 0;
        else 
            Yshared[threadIdx.y][threadIdx.x] = Y[index];
        
        __syncthreads();

        for (int k = 0; k < BLOCK_SIZE; ++k) 
            temp += Xshared[threadIdx.y][k] * Yshared[k][threadIdx.x];
        
        __syncthreads();
    }
    
    if(i < N && j < N) {
        A[i * N + j] = temp * alpha + beta * A[i * N + j];
    }

}


double* createMatrix(int M, int N) {
	double *A = new double[M*N];

	int i; 
	#pragma omp parallel for private(i)
	for(i = 0; i < M*N; i++) {
		A[i] = ((double) rand() / (RAND_MAX)); //Random number from 0 to 1
	}
		
	
	return A;
}

 double* createVector(int len) {
	double *A = new double[len];
	
	int i;
	#pragma omp parallel for private(i)
	for(i = 0; i < len; i++) {
		A[i] = ((double) rand() / (RAND_MAX));
	}
		
	
	return A;
}

dgemv_data* generateRandomData(int M, int N) {

	dgemv_data *data = new dgemv_data();
	data->M = M;
	data->N = N;
	data->A = new double[M*N];
	data->alpha = (double) (rand() % 10)  + ((double) rand() / (RAND_MAX)); 
	data->beta = (double) (rand() % 10)  + ((double) rand() / (RAND_MAX)); 
	data->lda = M; //Not used for dgemm
	data->X = createMatrix(N,N);
	data->Y = createMatrix(N,N);
	data->incX = 1; //Not used for dgemm
	data->incY = 1; //not used for dgemm

	return data;
}

dgemv_data* copyData(dgemv_data* data) {
	dgemv_data* copy = new dgemv_data();

	copy->M = data->N;
	copy->N = data->N;	
	copy->alpha = data->alpha;
	copy->beta = data->beta;
	copy->lda = data->lda;
	copy->incX = data->incX;
	copy->incY = data->incY;
	copy->X = new double[data->N * data->N];
	copy->Y = new double[data->N * data->N];
	copy->A = new double[data->N * data->N];
	memcpy (copy->X, data->X, data->N * data->N * sizeof(double));
	memcpy (copy->Y, data->Y, data->N * data->N * sizeof(double));
	memcpy (copy->A, data->A, data->N * data->N * sizeof(double));

	return copy;
}

bool compareMatrices(double *A,  double *B, int length) {
	int i;
	for(i = 0; i < length; i++) {
		//To account for doubleing pt error, check if greater than some epsilon
		if(abs(A[i] - B[i]) > 0.000001)  { 
			std::cout <<"i: " << i << " A[i]: " << A[i] << " B[i]: " << B[i] << std::endl;
			return false;
		}
	}

	return true;
}

void freeDataStruct(dgemv_data* data) {
	delete[](data->A);
	delete[](data->X);
	delete[](data->Y);
}

void testOutput(dgemv_data *data, dgemv_data* test_data) {
		dgemm(test_data->N,   test_data->alpha,
		       test_data->X,   test_data->Y,
		       test_data->beta,test_data->A);
		
		if(compareMatrices(data->A, test_data->A, data->N * data->N)) {
			std::cout << "Output: PASSED" << std::endl;
		} else {
			std::cout << "Output: FAILED" << std::endl;
		}
}

void printTimeTaken(unsigned long ns) {
	std::cout << std::fixed;
    std::cout << std::setprecision(10)
			  << "Time taken: " 
			  << ns 
			  << " ns or " 
			  << (double) ns/1000000000.0
			  << " s\n"
			  << std::endl;
}

int main(int argc, char **argv) {

	//Clock for C++
	if (argc != 3) {
		std::cout << "Invalid set of arguments.\n"
				  << "Usage: ./dgemv [Testing Off/On(0/1)] [size N]"
				  << std::endl;
		exit(-1);
	}

	//Get user arguments
	int M,N;
	bool test;

	//Square matrix
	if(argc == 3) {
		M = atoi(argv[2]);
		N = M;
	} 
	
	test = (atoi(argv[1]) > 0) ? true : false;


	//Feed random seed
	srand(time(NULL));

	//Generate the data
	dgemv_data *unModifiedData = generateRandomData(M,N); //DO not run functions on this
	


	/************************************************************************/
	//Run DGEMVT Serial Version
	
	//Get data sets to run with and test with (Since the arrays are modified)
	dgemv_data *data = copyData(unModifiedData);

	dgemv_data *serialTestData = copyData(unModifiedData);
	std::cout << "-----------------------------------------------" << std::endl;
	std::cout << "Running Serial Version of DGEMVT" << std::endl;
	auto start = Clock::now();
	dgemm(data->N, data->alpha, data->X,
		   data->Y, data->beta,  data->A);

	auto end = Clock::now();
	unsigned long ns = (unsigned long) std::chrono::duration_cast<nanoseconds>(end-start).count();
	printTimeTaken(ns);

	//Test the output
	if(test) {
		testOutput(data, serialTestData);
	}


	//Congifure CUDA blocksize and dim grid/block
    size_t gridR = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    size_t gridC = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
	dim3 dimGrid(gridC,gridR);
	dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);		



	/************************************************************************/
	//Run DGEMVT CUDA Version
	
	
	//Get data sets to run with and test with (Since the arrays are modified)
	dgemv_data *cuData = copyData(unModifiedData);
	dgemv_data *cuTestData = copyData(unModifiedData);
	
	//Double arrays for the GPU
	double *cuA, *cuX, *cuY;

	//Allocate space and copy data into GPU allocated arrays
	CUDA_WARN(cudaMalloc(&cuA, N * N * sizeof(double)));
	CUDA_WARN(cudaMalloc(&cuX, N * N * sizeof(double)));	
	CUDA_WARN(cudaMalloc(&cuY, N * N * sizeof(double)));	
	
	CUDA_WARN(cudaMemcpy(cuA, cuData->A, N*N*sizeof(double), cudaMemcpyHostToDevice));	
	CUDA_WARN(cudaMemcpy(cuX, cuData->X, N*N*sizeof(double), cudaMemcpyHostToDevice));	
	CUDA_WARN(cudaMemcpy(cuY, cuData->Y, N*N*sizeof(double), cudaMemcpyHostToDevice));	

	std::cout << "-----------------------------------------------" << std::endl;
	std::cout << "Running CUDA Version of DGEMVT" << std::endl;
	start = Clock::now();
	dgemm_cuda<<<dimGrid, dimBlock>>>
			   (N,cuData->alpha, cuX,
                cuY, cuData->beta, cuA);
	CUDA_WARN(cudaThreadSynchronize());
	end = Clock::now();
	ns = (unsigned long) std::chrono::duration_cast<nanoseconds>(end-start).count();

	CUDA_WARN(cudaGetLastError());

	
	//Copy the cude result back
	cudaMemcpy(cuData->A, cuA, N*N*sizeof(double), cudaMemcpyDeviceToHost);

	//Print the time taken
	printTimeTaken(ns);

	//Test the output
	if(test) {
		testOutput(cuData, cuTestData);
	}

	/************************************************************************/
	//Run DGEMVT CUDA Shared Memory Version	
	
	//Get data sets to run with and test with (Since the arrays are modified)
	dgemv_data *cuDataShared = copyData(unModifiedData);
	dgemv_data *cuTestDataShared = copyData(unModifiedData);
	
	//Double arrays for the GPU
	double *scuA, *scuX, *scuY;

	//Allocate space and copy data into GPU allocated arrays
	CUDA_WARN(cudaMalloc(&scuA, N * N * sizeof(double)));
	CUDA_WARN(cudaMalloc(&scuX, N * N * sizeof(double)));	
	CUDA_WARN(cudaMalloc(&scuY, N * N * sizeof(double)));	
	
	CUDA_WARN(cudaMemcpy(scuA, cuDataShared->A, N*N*sizeof(double), cudaMemcpyHostToDevice));	
	CUDA_WARN(cudaMemcpy(scuX, cuDataShared->X, N*N*sizeof(double), cudaMemcpyHostToDevice));	
	CUDA_WARN(cudaMemcpy(scuY, cuDataShared->Y, N*N*sizeof(double), cudaMemcpyHostToDevice));	

	std::cout << "-----------------------------------------------" << std::endl;
	std::cout << "Running CUDA Shared Memory Version of DGEMVT" << std::endl;
	start = Clock::now();
	dgemm_cuda_shared<<<dimGrid, dimBlock>>>
			   (N,cuDataShared->alpha, scuX,
                scuY, cuDataShared->beta, scuA);
	CUDA_WARN(cudaThreadSynchronize());
	end = Clock::now();
	ns = (unsigned long) std::chrono::duration_cast<nanoseconds>(end-start).count();

	CUDA_WARN(cudaGetLastError());

	
	//Copy the cude result back
	cudaMemcpy(cuDataShared->A, scuA, N*N*sizeof(double), cudaMemcpyDeviceToHost);

	//Print the time taken
	printTimeTaken(ns);

	//Test the output
	if(test) {
		testOutput(cuDataShared, cuTestDataShared);
	}


	return 0;
}
