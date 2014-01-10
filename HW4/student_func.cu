//Udacity HW 4
//Radix Sorting

#include "utils.h"
//#include <thrust/host_vector.h>
#include "timer.h"

/* Red Eye Removal
   ===============

   For this assignment we are implementing red eye removal.  This is
   accomplished by first creating a score for every pixel that tells us how
   likely it is to be a red eye pixel.  We have already done this for you - you
   are receiving the scores and need to sort them in ascending order so that we
   know which pixels to alter to remove the red eye.

   Note: ascending order == smallest to largest

   Each score is associated with a position, when you sort the scores, you must
   also move the positions accordingly.

   Implementing Parallel Radix Sort with CUDA
   ==========================================

   The basic idea is to construct a histogram on each pass of how many of each
   "digit" there are.   Then we scan this histogram so that we know where to put
   the output of each digit.  For example, the first 1 must come after all the
   0s so we have to know how many 0s there are to be able to start moving 1s
   into the correct position.

   1) Histogram of the number of occurrences of each digit
   2) Exclusive Prefix Sum of Histogram
   3) Determine relative offset of each digit
   For example [0 0 1 1 0 0 1]
   ->  [0 1 0 1 2 3 2]
   4) Combine the results of steps 2 & 3 to determine the final
   output location for each element and move it there

   LSB Radix sort is an out-of-place sort and you will need to ping-pong values
   between the input and output buffers we have provided.  Make sure the final
   sorted results end up in the output buffer!  Hint: You may need to do a copy
   at the end.

*/

using namespace std;

__global__ void calcHisto(unsigned int *d_histo, unsigned int *d_in, unsigned int mask, int i, size_t numElems) {
  int gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid >= (int)numElems) return;
  unsigned int bin = (d_in[gid] & mask) >> i;
  atomicAdd(&d_histo[bin], 1);
}

__global__ void calcOneBefore(unsigned int *d_in, unsigned int *d_oneBefore, unsigned int mask, int i, size_t numElems) {
  int gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid >= (int)numElems) return;
  if (((d_in[gid] & mask) >> i) == 1) {
    for (size_t k = gid+1; k < numElems; k++)
      atomicAdd(&d_oneBefore[k], 1);
  }
}

__global__ void movePos(unsigned int *d_valSrc, unsigned int *d_posSrc,
                        unsigned int *d_valDst, unsigned int *d_posDst,
                        unsigned int *d_scan, unsigned int *d_oneBefore,
                        unsigned int mask, int i, size_t numElems) {
  int gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid >= (int)numElems) return;
  unsigned int bin = (d_valSrc[gid] & mask) >> i;
  if (bin == 0) {
    d_valDst[d_scan[bin] + gid - d_oneBefore[gid]] = d_valSrc[gid];
    d_posDst[d_scan[bin] + gid - d_oneBefore[gid]] = d_posSrc[gid];
  } else {
    d_valDst[d_scan[bin] + d_oneBefore[gid]] = d_valSrc[gid];
    d_posDst[d_scan[bin] + d_oneBefore[gid]] = d_posSrc[gid];
  }
}

__global__ void copy(unsigned int *d_inVal, unsigned int *d_inPos,
                     unsigned int * d_outVal, unsigned int *d_outPos,
                     size_t numElems) {
  int gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid >= (int)numElems) return;
  d_outVal[gid] = d_inVal[gid];
  d_outPos[gid] = d_inPos[gid];
}

void your_sort(unsigned int* const d_inputVals,
               unsigned int* const d_inputPos,
               unsigned int* const d_outputVals,
               unsigned int* const d_outputPos,
               const size_t numElems)
{
  //TODO
  //PUT YOUR SORT HERE
  std::cout << "numElems = " << numElems << std::endl;

  const int nBins = 2;
  const dim3 blkDim(512, 1, 1);
  const dim3 grdDim(ceil(numElems/(double)blkDim.x), 1, 1);
  cout << "blkDim.x=" << blkDim.x << "\tgrdDim.x=" << grdDim.x << endl;

  //unsigned int *h_oneBefore = new unsigned int[numElems];
  unsigned int *d_histo, *d_scan, *d_oneBefore;
  size_t bin_size = nBins * sizeof(unsigned int);
  size_t ele_size = numElems * sizeof(unsigned int);
  checkCudaErrors(cudaMalloc(&d_histo, bin_size));
  checkCudaErrors(cudaMalloc(&d_scan, bin_size));
  checkCudaErrors(cudaMalloc(&d_oneBefore, ele_size));

  unsigned int *d_valSrc = d_inputVals;
  unsigned int *d_posSrc = d_inputPos;
  unsigned int *d_valDst = d_outputVals;
  unsigned int *d_posDst = d_outputPos;

  for (size_t i = 0; i < 8 * sizeof(unsigned int); i++) {
    unsigned int mask = 1 << i;
    checkCudaErrors(cudaMemset(d_histo, 0, bin_size));
    checkCudaErrors(cudaMemset(d_scan, 0, bin_size));
    checkCudaErrors(cudaMemset(d_oneBefore, 0, ele_size));

    // calculate histogram
    calcHisto<<<grdDim, blkDim>>>(d_histo, d_valSrc, mask, i, numElems);
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

    // sum scan -- since there are only two bins, it can be done by memcpy
    cudaMemcpy(&d_scan[1], d_histo, sizeof(unsigned int), cudaMemcpyDeviceToDevice);

    // count d_oneBefore
    // GpuTimer timer;
    // timer.Start();
    calcOneBefore<<<grdDim, blkDim>>>(d_valSrc, d_oneBefore, mask, i, numElems);
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
    // timer.Stop();
    // cout << timer.Elapsed() << endl;

    // move to the right position
    movePos<<<grdDim, blkDim>>>(d_valSrc, d_posSrc, d_valDst, d_posDst, d_scan, d_oneBefore, mask, i, numElems);
    cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

    std::swap(d_valDst, d_valSrc);
    std::swap(d_posDst, d_posSrc);
  }

  copy<<<grdDim, blkDim>>>(d_inputVals, d_inputPos, d_outputVals, d_outputPos, numElems);

  checkCudaErrors(cudaFree(d_histo));
  checkCudaErrors(cudaFree(d_scan));
  checkCudaErrors(cudaFree(d_oneBefore));
}
