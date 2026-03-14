NVCC ?= nvcc
ARCH ?= sm_86
NVFLAGS = -O3 -std=c++17 -arch=$(ARCH) -Icommon
LIBS = -lcublas

PROGS = bin/vecadd

all: $(PROGS)

bin:
	mkdir -p bin

bin/vecadd: vecadd/vecadd.cu common/common.h | bin
	$(NVCC) $(NVFLAGS) -o $@ $< $(LIBS)

clean:
	rm -rf bin

.PHONY: all clean
