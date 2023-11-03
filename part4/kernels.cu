#include <math.h>
#include <float.h>
#include <cuda.h>

__global__ void gpu_Heat (float *h, float *g, int N) {
	int i = threadIdx.x + blockDim.x * blockIdx.x;
	int w_inner = N -2;
	int x_inner = i%w_inner;
	int y_inner= i/w_inner;
	int outer = (y_inner+1) * N + x_inner + 1;
	g[outer] = h[outer -N] + h[outer + N] + h[outer -1]+ h[outer + 1];
	g[outer] /= 4;
}
