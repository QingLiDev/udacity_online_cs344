NVCC=nvcc

CUDA_INCLUDEPATH=/usr/local/cuda-5.0/include

NVCC_OPTS=-O3 -arch=sm_20 -Xcompiler -Wall -Xcompiler -Wextra -m64
GCC_OPTS=-O3 -Wall -Wextra -m64

OBJECTS = test_AtomicMaxFloat.o
EXES = test_AtomicMaxFloat.out

all: $(OBJECTS) $(EXES)

clean:
	rm *.o *.out

%.o: %.cu
	$(NVCC) $(NVCC_OPTS) -c $< -o $@

test_AtomicMaxFloat.out: test_AtomicMaxFloat.o
	$(NVCC) $(NVCC_OPTS) $< -o $@
