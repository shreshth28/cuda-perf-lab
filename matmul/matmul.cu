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

template <int T>
__global__ void mm_tiled(const float *a, const float *b, float *c, int n)
{
	__shared__ float as[T][T], bs[T][T];
	int ty = threadIdx.y, tx = threadIdx.x;
	int row = blockIdx.y * T + ty, col = blockIdx.x * T + tx;
	float s = 0.f;
	for (int t = 0; t < n; t += T) {
		as[ty][tx] = row < n && t + tx < n ? a[row * n + t + tx] : 0.f;
		bs[ty][tx] = t + ty < n && col < n ? b[(t + ty) * n + col] : 0.f;
		__syncthreads();
		for (int k = 0; k < T; k++)
			s += as[ty][k] * bs[k][tx];
		__syncthreads();
	}
	if (row < n && col < n)
		c[row * n + col] = s;
}

template <int BM, int BN, int BK, int TM>
__global__ void mm_reg1d(const float *a, const float *b, float *c, int n)
{
	__shared__ float as[BM * BK], bs[BK * BN];
	int tc = threadIdx.x % BN;
	int tr = threadIdx.x / BN;
	a += blockIdx.y * BM * n;
	b += blockIdx.x * BN;
	c += blockIdx.y * BM * n + blockIdx.x * BN;
	int ar = threadIdx.x / BK, ac = threadIdx.x % BK;
	int br = threadIdx.x / BN, bc = threadIdx.x % BN;
	float acc[TM] = {0.f};
	for (int t = 0; t < n; t += BK) {
		as[ar * BK + ac] = a[ar * n + ac];
		bs[br * BN + bc] = b[br * n + bc];
		__syncthreads();
		a += BK;
		b += BK * n;
		for (int k = 0; k < BK; k++) {
			float bv = bs[k * BN + tc];
			for (int m = 0; m < TM; m++)
				acc[m] += as[(tr * TM + m) * BK + k] * bv;
		}
		__syncthreads();
	}
	for (int m = 0; m < TM; m++)
		c[(tr * TM + m) * n + tc] = acc[m];
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
	} else if (args.kernel == "tiled") {
		dim3 bl(32, 32);
		dim3 g((n + 31) / 32, (n + 31) / 32);
		ms = time_ms(args, [&] { mm_tiled<32><<<g, bl>>>(a, b, c, n); });
	} else if (args.kernel == "reg1d") {
		dim3 g(n / 64, n / 64);
		ms = time_ms(args, [&] { mm_reg1d<64, 64, 8, 8><<<g, 64 * 64 / 8>>>(a, b, c, n); });
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
