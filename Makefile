NVCC ?= nvcc
ARCH ?= sm_86
NVFLAGS = -O3 -std=c++17 -arch=$(ARCH) -Icommon

PROGS =

all: $(PROGS)

bin:
	mkdir -p bin

clean:
	rm -rf bin

.PHONY: all clean
