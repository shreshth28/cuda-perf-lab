#include <algorithm>

#include "common.h"

__global__ void mm_naive(const float *a, const float *b, float *c, int n)
{
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	if (row < n && col < n) {
		float s = 0.f;
		for (int k = 0; k < n; k++)
			s += a[row * n + k] * b[k * n + col];
		c[row * n + col] = s;
	}
}

static void cpu_ref(const std::vector<float> &a, const std::vector<float> &b, std::vector<float> &c, int n)
{
	std::vector<double> acc(n);
	for (int i = 0; i < n; i++) {
		std::fill(acc.begin(), acc.end(), 0.0);
		for (int k = 0; k < n; k++) {
			double aik = a[(long)i * n + k];
			for (int j = 0; j < n; j++)
				acc[j] += aik * b[(long)k * n + j];
		}
		for (int j = 0; j < n; j++)
			c[(long)i * n + j] = acc[j];
	}
}

int main(int argc, char **argv)
{
	Args args = parse(argc, argv, 2048);
	int n = args.n;
	long nn = (long)n * n;

	std::vector<float> ha(nn), hb(nn), hc(nn);
	fill(ha, 5);
	fill(hb, 6);

	float *a, *b, *c;
	CK(cudaMalloc(&a, nn * sizeof(float)));
	CK(cudaMalloc(&b, nn * sizeof(float)));
	CK(cudaMalloc(&c, nn * sizeof(float)));
	CK(cudaMemcpy(a, ha.data(), nn * sizeof(float), cudaMemcpyHostToDevice));
	CK(cudaMemcpy(b, hb.data(), nn * sizeof(float), cudaMemcpyHostToDevice));

	float ms;
	if (args.kernel == "naive") {
		dim3 bl(32, 32);
		dim3 g((n + 31) / 32, (n + 31) / 32);
		ms = time_ms(args, [&] { mm_naive<<<g, bl>>>(a, b, c, n); });
	} else {
		fprintf(stderr, "unknown kernel %s\n", args.kernel.c_str());
		return 1;
	}

	bool ok = true;
	if (args.check) {
		CK(cudaMemcpy(hc.data(), c, nn * sizeof(float), cudaMemcpyDeviceToHost));
		std::vector<float> want(nn);
		cpu_ref(ha, hb, want, n);
		ok = close_enough(hc, want, 1e-3f);
	}

	double gflops = 2.0 * n * n * n / (ms * 1e6);
	report(args, ms, "gflops", gflops, ok);

	cudaFree(a);
	cudaFree(b);
	cudaFree(c);
	return ok ? 0 : 1;
}
