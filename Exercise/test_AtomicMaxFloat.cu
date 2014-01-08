#include <iostream>
#include <cstdlib>
#include <algorithm>

using namespace std;

__global__ void getMax(float *d_in, float *d_max) {
  int tid = threadIdx.x;
  int gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid == 0) d_max[0] = d_in[gid];
  __syncthreads();

  if (d_in[gid] > d_max[0])
    d_max[0] = d_in[gid];
}
int main(int argc, char **argv) {
  srand(time(NULL));

  const int ARRAY_SIZE = 1024;
  float h_in[ARRAY_SIZE];
  for (int i = 0; i < ARRAY_SIZE; i++) h_in[i] = float(i);
  random_shuffle(&h_in[0], &h_in[ARRAY_SIZE]);
  //http://stackoverflow.com/questions/14720134/is-it-possible-to-random-shuffle-an-array-of-int-elements
  const dim3 blkDim(1, 1, 1);
  const dim3 grdDim(ARRAY_SIZE, 1, 1);
  size_t fsize = sizeof(float);

  float *d_in;
  cudaMalloc(&d_in, fsize*ARRAY_SIZE);
  cudaMemcpy(d_in, h_in, fsize*ARRAY_SIZE, cudaMemcpyDeviceToHost);
  float *d_max, *h_max;
  cudaMalloc(&d_max, fsize);
  cudaMalloc(&h_max, fsize);

  // launch the kernel
  getMax<<<grdDim, blkDim>>>(d_in, d_max);

  cudaMemcpy(h_max, d_max, fsize, cudaMemcpyHostToDevice);
  //cout << h_max[0] << endl;
}
