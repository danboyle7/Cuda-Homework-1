all: main

main: main.cu
	nvcc  -g -std=c++11  "main.cu" -o main -L/usr/local/cuda/lib64 -lm -lcuda -Wno-deprecated-gpu-targets
	 

clean: 
	rm -rf *.o
	rm -rf main
