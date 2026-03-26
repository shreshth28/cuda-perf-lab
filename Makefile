NVCC ?= nvcc
ARCH ?= sm_86
NVFLAGS = -O3 -std=c++17 -arch=$(ARCH) -Icommon
LIBS = -lcublas

PROGS = bin/vecadd bin/reduce bin/transpose

all: $(PROGS)

bin:
	mkdir -p bin

bin/vecadd: vecadd/vecadd.cu common/common.h | bin
	$(NVCC) $(NVFLAGS) -o $@ $< $(LIBS)

bin/reduce: reduce/reduce.cu common/common.h | bin
	$(NVCC) $(NVFLAGS) -o $@ $< $(LIBS)

bin/transpose: transpose/transpose.cu common/common.h | bin
	$(NVCC) $(NVFLAGS) -o $@ $< $(LIBS)

clean:
	rm -rf bin

.PHONY: all clean
