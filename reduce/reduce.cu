#include "common.h"

__global__ void sum_atomic(const float *x, float *out, long n)
{
	long i = blockIdx.x * (long)blockDim.x + threadIdx.x;
	if (i < n)
		atomicAdd(out, x[i]);
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

	float ms;
	if (args.kernel == "atomic") {
		ms = time_ms(args, [&] {
			CK(cudaMemsetAsync(out, 0, sizeof(float)));
			sum_atomic<<<grid, bs>>>(x, out, n);
		});
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
