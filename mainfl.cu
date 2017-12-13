#include <iostream>
#include <cstdlib>
#include <iomanip>
#include <cstring>
#include <cuda_runtime.h>
#include <cstdio>

//C++ timers
#include <chrono>

#define BLOCK_SIZE 16
#define N_STREAMS 8

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
	int alpha;
	int beta;
	
	//First dimension of A 
	int lda;

	//Increments for X and Y
	int incX;
	int incY;

	//Array A and vectors X, Y;
	int *A, *X, *Y;
} dgemv_data;


// BASIC DGEMM METHOD
void dgemm(int N,       int alpha,
		   int *X,   int *Y,
           int beta, int *A) {

 	
  	for (int i = 0; i < N; ++i) {
    	for (int j = 0; j < N; ++j) {
            int tmp = 0;
            for (int k = 0; k < N; ++k) {
                tmp += X[i * N + k] * Y[k * N + j];
            }
            A[i * N + j] = alpha * tmp + beta * A[i * N + j];
        }
    }
}

 
// CUDA DGEMM W/O SHARED MEMORY
__global__ void dgemm_cuda (int N,       int alpha,
						    int *X,   int *Y,
							int beta, int *A) {

  	int i = blockDim.y * blockIdx.y + threadIdx.y;
  	int j = blockDim.x * blockIdx.x + threadIdx.x;
  
  	if (i < N && j < N) {
  
  		int temp = 0;
  		for (int k = 0; k < N; ++k) {
    		temp += X[i * N + k] * Y[k * N + j];
  		}
    
    	A[i * N + j] = alpha * temp + beta * A[i * N + j];
	}
    
}

 
// CUDA DGEMM W/O SHARED MEMORY
__global__ void dgemm_cuda_shared (int N,       int alpha,
						    	   int *X,   int *Y,
								   int beta, int *A) {

	// Create Shared Memory Arrays
    __shared__ int Xshared[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ int Yshared[BLOCK_SIZE][BLOCK_SIZE];

    //Setup i and j
    int i = blockIdx.y * BLOCK_SIZE + threadIdx.y;
    int j = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    int temp = 0;

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


int* createMatrix(int M, int N) {
	int *A = new int[M*N];

	int i; 
	#pragma omp parallel for private(i)
	for(i = 0; i < M*N; i++) {
		A[i] = (int) (rand() % 10); //Random number from 0 to 1
	}
		
	
	return A;
}

 int* createVector(int len) {
	int *A = new int[len];
	
	int i;
	#pragma omp parallel for private(i)
	for(i = 0; i < len; i++) {
		A[i] = rand() % 10;
	}
		
	
	return A;
}

dgemv_data* generateRandomData(int M, int N) {

	dgemv_data *data = new dgemv_data();
	data->M = M;
	data->N = N;
	data->A = new int[M*N];
	data->alpha = (int) (rand() % 10);
	data->beta = (int) (rand() % 10);
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
	copy->X = new int[data->N * data->N];
	copy->Y = new int[data->N * data->N];
	copy->A = new int[data->N * data->N];
	memcpy (copy->X, data->X, data->N * data->N * sizeof(int));
	memcpy (copy->Y, data->Y, data->N * data->N * sizeof(int));
	memcpy (copy->A, data->A, data->N * data->N * sizeof(int));

	return copy;
}

bool compareMatrices(int *A,  int *B, int length) {
	int i;
	for(i = 0; i < length; i++) {
		//To account for inting pt error, check if greater than some epsilon
		if(abs(A[i] - B[i]) != 0)  { //was float but this still wr
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

// void printTimeTaken(unsigned long ms) {
// 	std::cout << std::fixed;
//     std::cout << std::setprecision(10)
// 			  << "Time taken: " 
// 			  << ms 
// 			  << " ms or " 
// 			  << (unsigned long) ms/1000.0
// 			  << " s\n"
// 			  << std::endl;
// }

void printTimeTakenFloat(float ms) {
	std::cout << std::fixed;
    std::cout << std::setprecision(10)
			  << "Time taken: " 
			  << ms 
			  << " ms or " 
			  << ms/1000.0
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
	std::cout << "Running Serial Version of DGEMM" << std::endl;
	auto start = Clock::now();
	//dgemm(data->N, data->alpha, data->X,
	//	   data->Y, data->beta,  data->A);

	auto stop = Clock::now();
	unsigned long ns = (unsigned long) std::chrono::duration_cast<nanoseconds>(stop-start).count();
	printTimeTakenFloat(ns/1000000.0);

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
	
	//int arrays for the GPU
	int *cuA, *cuX, *cuY;

	//Start time
	start = Clock::now();

	//Allocate space and copy data into GPU allocated arrays
	CUDA_WARN(cudaMalloc(&cuA, N * N * sizeof(int)));
	CUDA_WARN(cudaMalloc(&cuX, N * N * sizeof(int)));	
	CUDA_WARN(cudaMalloc(&cuY, N * N * sizeof(int)));	
	
	CUDA_WARN(cudaMemcpy(cuA, cuData->A, N*N*sizeof(int), cudaMemcpyHostToDevice));	
	CUDA_WARN(cudaMemcpy(cuX, cuData->X, N*N*sizeof(int), cudaMemcpyHostToDevice));	
	CUDA_WARN(cudaMemcpy(cuY, cuData->Y, N*N*sizeof(int), cudaMemcpyHostToDevice));	

	std::cout << "-----------------------------------------------" << std::endl;
	std::cout << "Running Basic CUDA Version of DGEMM" << std::endl;
	dgemm_cuda<<<dimGrid, dimBlock>>>
			   (N,cuData->alpha, cuX,
                cuY, cuData->beta, cuA);
	
	//Check if there was an error
	CUDA_WARN(cudaGetLastError());
	CUDA_WARN(cudaThreadSynchronize());
	
	//Copy the cude result back
	CUDA_WARN(cudaMemcpy(cuData->A, cuA, N*N*sizeof(int), cudaMemcpyDeviceToHost));

	//Print the time taken
	stop = Clock::now();
	ns = (unsigned long) std::chrono::duration_cast<nanoseconds>(stop-start).count();
	printTimeTakenFloat(ns/1000000.0);

	//Test the output
	if(test) {
		testOutput(cuData, cuTestData);
	}

	//free variables
	CUDA_WARN(cudaFree(cuA));
	CUDA_WARN(cudaFree(cuX));
	CUDA_WARN(cudaFree(cuY));

	/************************************************************************/
	//Run DGEMVT CUDA Shared Memory Version	
	
	//Get data sets to run with and test with (Since the arrays are modified)
	dgemv_data *cuDataShared = copyData(unModifiedData);
	dgemv_data *cuTestDataShared = copyData(unModifiedData);
	
	//int arrays for the GPU
	int *scuA, *scuX, *scuY;
	
	//Start time
	start = Clock::now();

	//Allocate space and copy data into GPU allocated arrays
	CUDA_WARN(cudaMalloc(&scuA, N * N * sizeof(int)));
	CUDA_WARN(cudaMalloc(&scuX, N * N * sizeof(int)));	
	CUDA_WARN(cudaMalloc(&scuY, N * N * sizeof(int)));	
	
	CUDA_WARN(cudaMemcpy(scuA, cuDataShared->A, N*N*sizeof(int), cudaMemcpyHostToDevice));	
	CUDA_WARN(cudaMemcpy(scuX, cuDataShared->X, N*N*sizeof(int), cudaMemcpyHostToDevice));	
	CUDA_WARN(cudaMemcpy(scuY, cuDataShared->Y, N*N*sizeof(int), cudaMemcpyHostToDevice));	

	std::cout << "-----------------------------------------------" << std::endl;
	std::cout << "Running Shared Memory CUDA Version of DGEMM" << std::endl;
	dgemm_cuda_shared<<<dimGrid, dimBlock>>>
			   (N,cuDataShared->alpha, scuX,
                scuY, cuDataShared->beta, scuA);
	
	//Check if there was an error
	CUDA_WARN(cudaGetLastError());
	CUDA_WARN(cudaThreadSynchronize());
	
	//Copy the cude result back
	CUDA_WARN(cudaMemcpy(cuDataShared->A, scuA, N*N*sizeof(int), cudaMemcpyDeviceToHost));

	//Print the time taken
	stop = Clock::now();
	ns = (unsigned long) std::chrono::duration_cast<nanoseconds>(stop-start).count();
	printTimeTakenFloat(ns/1000000.0);

	//Test the output
	if(test) {
		testOutput(cuDataShared, cuTestDataShared);
	}


	//free variables
	CUDA_WARN(cudaFree(scuA));
	CUDA_WARN(cudaFree(scuX));
	CUDA_WARN(cudaFree(scuY));


	/************************************************************************/
	//Run DGEMVT CUDA Shared Memory Version	w/ Driver Events/Streams
	
	//Get data sets to run with and test with (Since the arrays are modified)
	dgemv_data *cuDataShared2 = copyData(unModifiedData);
	dgemv_data *cuTestDataShared2 = copyData(unModifiedData);
	
	//int arrays for the GPU
	int *scuA2, *scuX2, *scuY2;

	//Events and streams
	cudaStream_t stream[N_STREAMS];
	cudaEvent_t begin, end;
	cudaEventCreate(&begin);
    cudaEventCreate(&end);
    for(int i = 0; i < N_STREAMS; i++) cudaStreamCreate(&stream[i]);
   
    //Start the event timer
    cudaEventRecord(begin, 0);
   
	//Allocate space and copy data into GPU allocated arrays
	CUDA_WARN(cudaMalloc(&scuA2, N * N * sizeof(int)));
	CUDA_WARN(cudaMalloc(&scuX2, N * N * sizeof(int)));	
	CUDA_WARN(cudaMalloc(&scuY2, N * N * sizeof(int)));	
	
	//COpy Memory to the GPU
	CUDA_WARN(cudaMemcpyAsync(scuA2, cuDataShared2->A, N*N*sizeof(int), cudaMemcpyHostToDevice, stream[0]));	
	CUDA_WARN(cudaMemcpyAsync(scuX2, cuDataShared2->X, N*N*sizeof(int), cudaMemcpyHostToDevice, stream[1]));	
	CUDA_WARN(cudaMemcpyAsync(scuY2, cuDataShared2->Y, N*N*sizeof(int), cudaMemcpyHostToDevice, stream[2]));	
	cudaDeviceSynchronize();

	std::cout << "-----------------------------------------------" << std::endl;
	std::cout << "Running CUDA Shared Memory Version of DGEMM" << std::endl;

	dgemm_cuda_shared<<<dimGrid, dimBlock>>>
			   (N,cuDataShared2->alpha, scuX2,
                scuY2, cuDataShared2->beta, scuA2);

	//Check if there was an error
	CUDA_WARN(cudaGetLastError());
	CUDA_WARN(cudaThreadSynchronize());
	
	//Streams
  	for (int i = 0; i < N_STREAMS; i++) {
    	cudaMemcpyAsync(cuDataShared2->A+(i*N*N)/N_STREAMS, scuA2+(i*N*N)/N_STREAMS, sizeof(int)*(N*N)/N_STREAMS, cudaMemcpyDeviceToHost, stream[i]);
  	}
	cudaDeviceSynchronize();

	//Stop the timer
    cudaEventRecord(end, 0);
    cudaEventSynchronize(end);

	//Print the time taken
	float ms;
	cudaEventElapsedTime(&ms, begin, end);
	printTimeTakenFloat(ms);

	//Test the output
	if(test) {
		testOutput(cuDataShared2, cuTestDataShared2);
	}


	//free variables
	CUDA_WARN(cudaFree(scuA2));
	CUDA_WARN(cudaFree(scuX2));
	CUDA_WARN(cudaFree(scuY2));

	return 0;
}
