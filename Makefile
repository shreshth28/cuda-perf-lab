NVCC ?= nvcc
CXX ?= g++
ARCH ?= sm_86
NVFLAGS = -O3 -std=c++17 -arch=$(ARCH) -Icommon
LIBS = -lcublas
CXXFLAGS = -O3 -march=native -std=c++17 -fopenmp -Icommon
CLFLAGS = -O3 -std=c++17 -Icommon -Iopencl
CLLIBS = -lOpenCL

PROGS = bin/vecadd bin/reduce bin/transpose bin/matmul bin/matmul_cpu
CL_PROGS = bin/cl_vecadd bin/cl_reduce bin/cl_transpose

all: $(PROGS)

opencl: $(CL_PROGS)

bin:
	mkdir -p bin

bin/vecadd: vecadd/vecadd.cu common/common.h common/args.h | bin
	$(NVCC) $(NVFLAGS) -o $@ $< $(LIBS)

bin/reduce: reduce/reduce.cu common/common.h common/args.h | bin
	$(NVCC) $(NVFLAGS) -o $@ $< $(LIBS)

bin/transpose: transpose/transpose.cu common/common.h common/args.h | bin
	$(NVCC) $(NVFLAGS) -o $@ $< $(LIBS)

bin/matmul: matmul/matmul.cu common/common.h common/args.h | bin
	$(NVCC) $(NVFLAGS) -o $@ $< $(LIBS)

bin/matmul_cpu: matmul/matmul_cpu.cpp common/args.h | bin
	$(CXX) $(CXXFLAGS) -o $@ $<

bin/cl_%: opencl/%.cpp opencl/cl_common.h common/args.h | bin
	$(CXX) $(CLFLAGS) -o $@ $< $(CLLIBS)

clean:
	rm -rf bin

.PHONY: all opencl clean
