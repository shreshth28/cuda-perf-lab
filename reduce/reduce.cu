#include "common.h"

__global__ void sum_atomic(const float *x, float *out, long n)
{
	long i = blockIdx.x * (long)blockDim.x + threadIdx.x;
	if (i < n)
		atomicAdd(out, x[i]);
}

template <int BS>
__global__ void sum_tree(const float *x, float *out, long n)
{
	__shared__ float s[BS];
	long i = blockIdx.x * (long)BS + threadIdx.x;
	s[threadIdx.x] = x[i];
	__syncthreads();
	for (int k = BS / 2; k > 0; k >>= 1) {
		if (threadIdx.x < k)
			s[threadIdx.x] += s[threadIdx.x + k];
		__syncthreads();
	}
	if (threadIdx.x == 0)
		atomicAdd(out, s[0]);
}

__device__ float warp_sum(float v)
{
	for (int o = 16; o > 0; o >>= 1)
		v += __shfl_down_sync(0xffffffff, v, o);
	return v;
}

__device__ void block_sum_to(float v, float *out)
{
	__shared__ float w[32];
	v = warp_sum(v);
	int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
	if (lane == 0)
		w[wid] = v;
	__syncthreads();
	if (wid == 0) {
		v = lane < blockDim.x / 32 ? w[lane] : 0.f;
		v = warp_sum(v);
		if (lane == 0)
			atomicAdd(out, v);
	}
}

__global__ void sum_shfl(const float *x, float *out, long n)
{
	long i = blockIdx.x * (long)blockDim.x + threadIdx.x;
	block_sum_to(i < n ? x[i] : 0.f, out);
}

__global__ void sum_multi(const float *x, float *out, long n)
{
	long stride = (long)gridDim.x * blockDim.x;
	float v = 0.f;
	for (long i = blockIdx.x * (long)blockDim.x + threadIdx.x; i < n; i += stride)
		v += x[i];
	block_sum_to(v, out);
}

int main(int argc, char **argv)
{
	Args args = parse(argc, argv, 1L << 24);
	long n = args.n;

	std::vector<float> hx(n);
	fill(hx, 3);

	float *x, *out;
	CK(cudaMalloc(&x, n * sizeof(float)));
	CK(cudaMalloc(&out, sizeof(float)));
	CK(cudaMemcpy(x, hx.data(), n * sizeof(float), cudaMemcpyHostToDevice));

	int bs = 256;
	int grid = (n + bs - 1) / bs;

	int sms;
	CK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0));

	float ms;
	if (args.kernel == "atomic") {
		ms = time_ms(args, [&] {
			CK(cudaMemsetAsync(out, 0, sizeof(float)));
			sum_atomic<<<grid, bs>>>(x, out, n);
		});
	} else if (args.kernel == "tree") {
		ms = time_ms(args, [&] {
			CK(cudaMemsetAsync(out, 0, sizeof(float)));
			sum_tree<256><<<grid, 256>>>(x, out, n);
		});
	} else if (args.kernel == "shfl") {
		ms = time_ms(args, [&] {
			CK(cudaMemsetAsync(out, 0, sizeof(float)));
			sum_shfl<<<grid, bs>>>(x, out, n);
		});
	} else if (args.kernel == "multi") {
		ms = time_ms(args, [&] {
			CK(cudaMemsetAsync(out, 0, sizeof(float)));
			sum_multi<<<sms * 8, bs>>>(x, out, n);
		});
	} else if (args.kernel == "cublas") {
		cublasHandle_t h;
		CKB(cublasCreate(&h));
		CKB(cublasSetPointerMode(h, CUBLAS_POINTER_MODE_DEVICE));
		ms = time_ms(args, [&] { CKB(cublasSasum(h, n, x, 1, out)); });
		CK(cudaDeviceSynchronize());
		cublasDestroy(h);
	} else {
		fprintf(stderr, "unknown kernel %s\n", args.kernel.c_str());
		return 1;
	}

	bool ok = true;
	if (args.check) {
		float got;
		CK(cudaMemcpy(&got, out, sizeof(float), cudaMemcpyDeviceToHost));
		double want = 0;
		for (long i = 0; i < n; i++)
			want += hx[i];
		ok = fabs(got - want) <= 1e-3 * fabs(want);
		if (!ok)
			fprintf(stderr, "got %f want %f\n", got, want);
	}

	double gbps = 1.0 * n * sizeof(float) / (ms * 1e6);
	report(args, ms, "gbps", gbps, ok);

	cudaFree(x);
	cudaFree(out);
	return ok ? 0 : 1;
}
