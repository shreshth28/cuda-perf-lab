#!/bin/bash
set -e

make -s

run() {
	prog=$1
	shift
	sizes=$1
	shift
	for k in "$@"; do
		for n in $sizes; do
			./bin/$prog --kernel $k --n $n || echo "kernel=$k n=$n FAILED"
		done
	done
}

echo "# vecadd"
run vecadd "1048576 16777216 67108864" naive stride vec4 cublas
echo "# reduce"
run reduce "1048576 16777216 67108864" atomic tree shfl multi cublas
echo "# transpose"
run transpose "1024 2048 4096 8192" naive tiled tiled_pad cublas
echo "# matmul"
run matmul "512 1024 2048 4096" naive tiled reg1d reg2d vec4 cublas
echo "# matmul cpu"
run matmul_cpu "256 512 1024" naive ikj omp
