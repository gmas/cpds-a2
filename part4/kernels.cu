#include <math.h>
#include <float.h>
#include <cuda.h>

#ifndef NumElem
#define NumElem 512
#endif


__global__ void gpu_Heat (float *h, float *g,float *res, int N) {
	int num_elements_computed = N-2;
	int x_inner = threadIdx.x + blockDim.x * blockIdx.x;
	int y_inner = threadIdx.y + blockDim.y * blockIdx.y;
	int outer = (y_inner+1)*N +x_inner + 1;
	float computed_sum = (h[outer -N] + h[outer + N] + h[outer -1]+ h[outer + 1]) /4;
	g[outer] = computed_sum;
	float diff = h[outer] - computed_sum;
	res[x_inner + y_inner*num_elements_computed] = diff*diff;
}

// Directly adapted from the examples of hands on cuda
__global__ void Kernel07(float *g_idata, float *g_odata, int N) {
  __shared__ float sdata[NumElem];
  unsigned int s;

  // Cada thread realiza la suma parcial de los datos que le
  // corresponden y la deja en la memoria compartida
  unsigned int tid = threadIdx.x;
  unsigned int i = blockIdx.x*(blockDim.x*2) + threadIdx.x;
  unsigned int gridSize = blockDim.x*2*gridDim.x;
  sdata[tid] = 0;
  while (i < N) {
    sdata[tid] += g_idata[i] + g_idata[i+blockDim.x];
    i += gridSize;
  }
  __syncthreads();

  // Hacemos la reduccion en la memoria compartida
  for (s=blockDim.x/2; s>32; s>>=1) {
    if (tid < s)
      sdata[tid] += sdata[tid + s];
    __syncthreads();
  }
  // desenrrollamos el ultimo warp activo
  if (tid < 32) {
    volatile float *smem = sdata;

    smem[tid] += smem[tid + 32];
    smem[tid] += smem[tid + 16];
    smem[tid] += smem[tid + 8];
    smem[tid] += smem[tid + 4];
    smem[tid] += smem[tid + 2];
    smem[tid] += smem[tid + 1];
  }


  // El thread 0 escribe el resultado de este bloque en la memoria global
  if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}


__global__ void Kernel06(float *g_idata, float *g_odata) {
  __shared__ float sdata[NumElem];
  unsigned int s;

  // Cada thread carga 2 elementos desde la memoria global,
  // los suma y los deja en la memoria compartida
  unsigned int tid = threadIdx.x;
  unsigned int i = blockIdx.x*(blockDim.x*2) + threadIdx.x;
  sdata[tid] = g_idata[i] + g_idata[i+blockDim.x];
  __syncthreads();

  // Hacemos la reduccion en la memoria compartida
  for (s=blockDim.x/2; s>32; s>>=1) {
    if (tid < s)
      sdata[tid] += sdata[tid + s];
    __syncthreads();
  } 

 // desenrrollamos el ultimo warp activo
 if (tid < 32) {
   volatile float *smem = sdata;

   smem[tid] += smem[tid + 32];
   smem[tid] += smem[tid + 16];
   smem[tid] += smem[tid + 8];
   smem[tid] += smem[tid + 4];
   smem[tid] += smem[tid + 2];
   smem[tid] += smem[tid + 1];
 }

 // El thread 0 escribe el resultado de este bloque en la memoria global
 if (tid == 0) g_odata[blockIdx.x] = sdata[0];

}

__global__ void gpu_Residual (float *h, float *g, int N) {
	int num_elements_computed = N-2;
	int x_inner = threadIdx.x + blockDim.x * blockIdx.x;
	int y_inner = threadIdx.y + blockDim.y * blockIdx.y;
	int outer = (y_inner+1)*N +x_inner + 1;    
	g[outer] = (h[outer -N] + h[outer + N] + h[outer -1]+ h[outer + 1]) /4;	
}
