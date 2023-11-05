#include <math.h>
#include <float.h>
#include <cuda.h>

__global__ void gpu_Heat (float *h, float *g, int N) {
	int y_inner = threadIdx.x + blockDim.x * blockIdx.x;
	for(int i =1; i< N-1; ++i) {
	int outer = (y_inner+1)*N +i;    
	g[outer] = h[outer -N] + h[outer + N] + h[outer -1]+ h[outer + 1];
	g[outer] /= 4;
	}
}
