#include <stdio.h>
#include <iostream>
#include <cstdlib>
#include <algorithm>

using namespace std;

__device__ float atomicMaxFloat(float* addr, float val) {
  int *addrAsInt = (int *) addr;
  int old = *addrAsInt ;
  while(val > __int_as_float(old)) {
    old = atomicCAS(addrAsInt, old, __float_as_int(val));
  }

  return __int_as_float(old);
}

// Please note that __syncthreads() can only synchronize threads in a
// particular block
__global__ void getMax(float *d_in, float *d_max) {
  int gid = blockIdx.x * blockDim.x + threadIdx.x;

  // if (d_in[gid] > d_max[0])
    // d_max[0] = d_in[gid];
  atomicMaxFloat(d_max, d_in[gid]);
}
int main() {
  srand(time(NULL));

  const int ARRAY_SIZE = 10;
  float h_in[ARRAY_SIZE];
  for (int i = 0; i < ARRAY_SIZE; i++) h_in[i] = float(i)+0.2;
  random_shuffle(&h_in[0], &h_in[ARRAY_SIZE]);
  float h_max[1];
  //http://stackoverflow.com/questions/14720134/is-it-possible-to-random-shuffle-an-array-of-int-elements
  const dim3 blkDim(1, 1, 1);
  const dim3 grdDim(ARRAY_SIZE, 1, 1);
  size_t fsize = sizeof(float);

  float *d_in;
  cudaMalloc(&d_in, fsize*ARRAY_SIZE);
  cudaMemcpy(d_in, h_in, fsize*ARRAY_SIZE, cudaMemcpyHostToDevice);
  float *d_max;
  cudaMalloc(&d_max, fsize);
  cudaMemcpy(d_max, h_in, fsize, cudaMemcpyHostToDevice);

  // launch the kernel
  getMax<<<grdDim, blkDim>>>(d_in, d_max);

  cudaMemcpy(h_max, d_max, fsize, cudaMemcpyDeviceToHost);

  cudaFree(d_in);
  cudaFree(d_max);
  cout << h_max[0] << endl;
}
