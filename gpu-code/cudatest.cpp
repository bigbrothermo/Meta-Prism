
#include "cuda_runtime.h"
#include <iostream>
#include <stdio.h>

using namespace std;
 
const int N = 10;
 
__global__ void add_Jeremy(int*a, int*b, int*c)
{
	int tid = blockIdx.x;
	if (tid < N)
	{
		c[tid] = a[tid] + b[tid];
	}
}


 

int main()
{
	int a[N], b[N], c[N];
	int *dev_a, *dev_b, *dev_c;
 
	cudaMalloc((void**)&dev_a, N*sizeof(int));
	cudaMalloc((void**)&dev_b, N*sizeof(int));
	cudaMalloc((void**)&dev_c, N*sizeof(int));
 
	for (int i = 0; i < N; i++)
	{
		a[i] = -i;
		b[i] = i * i;
	}




	size_t free;
    size_t total;
 
    cudaMemGetInfo(&free, &total);

    std::cout<<"free:"<<free<<"\t"<<"total:"<<total<<std::endl;



 
	cudaMemcpy(dev_a, a, N*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_b, b, N*sizeof(int), cudaMemcpyHostToDevice);
 
	add_Jeremy<<<N,1>>>(dev_a, dev_b, dev_c);
 
	cudaMemcpy(c, dev_c, N*sizeof(int), cudaMemcpyDeviceToHost);
 
	for (int i = 0; i < N; i++)
	{
		printf("%d + %d = %d\n", a[i], b[i], c[i]);
	}
 
	cudaFree(dev_a);
	cudaFree(dev_b);
	cudaFree(dev_c);
 
	return 0;
}
