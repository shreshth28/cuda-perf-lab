#include "common.h"

__global__ void tr_naive(const float *in, float *out, int n)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x < n && y < n)
		out[(long)x * n + y] = in[(long)y * n + x];
}

int main(int argc, char **argv)
{
	Args args = parse(argc, argv, 4096);
	int n = args.n;
	long nn = (long)n * n;

	std::vector<float> hin(nn), hout(nn);
	fill(hin, 4);

	float *in, *out;
	CK(cudaMalloc(&in, nn * sizeof(float)));
	CK(cudaMalloc(&out, nn * sizeof(float)));
	CK(cudaMemcpy(in, hin.data(), nn * sizeof(float), cudaMemcpyHostToDevice));

	dim3 b(32, 8);
	dim3 g((n + 31) / 32, (n + 7) / 8);

	float ms;
	if (args.kernel == "naive") {
		ms = time_ms(args, [&] { tr_naive<<<g, b>>>(in, out, n); });
	} else {
		fprintf(stderr, "unknown kernel %s\n", args.kernel.c_str());
		return 1;
	}

	bool ok = true;
	if (args.check) {
		CK(cudaMemcpy(hout.data(), out, nn * sizeof(float), cudaMemcpyDeviceToHost));
		std::vector<float> want(nn);
		for (int y = 0; y < n; y++)
			for (int x = 0; x < n; x++)
				want[(long)x * n + y] = hin[(long)y * n + x];
		ok = close_enough(hout, want, 0.f);
	}

	double gbps = 2.0 * nn * sizeof(float) / (ms * 1e6);
	report(args, ms, "gbps", gbps, ok);

	cudaFree(in);
	cudaFree(out);
	return ok ? 0 : 1;
}
