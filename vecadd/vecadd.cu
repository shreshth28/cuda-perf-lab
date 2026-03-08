#include "common.h"

__global__ void add_naive(const float *a, const float *b, float *c, long n)
{
	long i = blockIdx.x * (long)blockDim.x + threadIdx.x;
	if (i < n)
		c[i] = a[i] + b[i];
}

__global__ void add_stride(const float *a, const float *b, float *c, long n)
{
	long stride = (long)gridDim.x * blockDim.x;
	for (long i = blockIdx.x * (long)blockDim.x + threadIdx.x; i < n; i += stride)
		c[i] = a[i] + b[i];
}

int main(int argc, char **argv)
{
	Args args = parse(argc, argv, 1L << 24);
	long n = args.n;

	std::vector<float> ha(n), hb(n), hc(n);
	fill(ha, 1);
	fill(hb, 2);

	float *a, *b, *c;
	CK(cudaMalloc(&a, n * sizeof(float)));
	CK(cudaMalloc(&b, n * sizeof(float)));
	CK(cudaMalloc(&c, n * sizeof(float)));
	CK(cudaMemcpy(a, ha.data(), n * sizeof(float), cudaMemcpyHostToDevice));
	CK(cudaMemcpy(b, hb.data(), n * sizeof(float), cudaMemcpyHostToDevice));

	int bs = 256;
	int grid = (n + bs - 1) / bs;

	int sms;
	CK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0));

	float ms;
	if (args.kernel == "naive") {
		ms = time_ms(args, [&] { add_naive<<<grid, bs>>>(a, b, c, n); });
	} else if (args.kernel == "stride") {
		ms = time_ms(args, [&] { add_stride<<<sms * 8, bs>>>(a, b, c, n); });
	} else {
		fprintf(stderr, "unknown kernel %s\n", args.kernel.c_str());
		return 1;
	}

	bool ok = true;
	if (args.check) {
		CK(cudaMemcpy(hc.data(), c, n * sizeof(float), cudaMemcpyDeviceToHost));
		std::vector<float> want(n);
		for (long i = 0; i < n; i++)
			want[i] = ha[i] + hb[i];
		ok = close_enough(hc, want, 1e-6f);
	}

	double gbps = 3.0 * n * sizeof(float) / (ms * 1e6);
	report(args, ms, "gbps", gbps, ok);

	cudaFree(a);
	cudaFree(b);
	cudaFree(c);
	return ok ? 0 : 1;
}
