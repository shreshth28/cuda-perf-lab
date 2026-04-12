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

template <int BM, int BN, int BK, int TM, int TN>
__global__ void __launch_bounds__((BM / TM) * (BN / TN)) mm_reg2d(const float *a, const float *b, float *c, int n)
{
	__shared__ float as[BM * BK], bs[BK * BN];
	const int nt = (BM / TM) * (BN / TN);
	int tc = threadIdx.x % (BN / TN);
	int tr = threadIdx.x / (BN / TN);
	a += blockIdx.y * BM * n;
	b += blockIdx.x * BN;
	c += blockIdx.y * BM * n + blockIdx.x * BN;
	int ar = threadIdx.x / BK, ac = threadIdx.x % BK, sa = nt / BK;
	int br = threadIdx.x / BN, bc = threadIdx.x % BN, sb = nt / BN;
	float acc[TM * TN] = {0.f};
	float ra[TM], rb[TN];
	for (int t = 0; t < n; t += BK) {
		for (int o = 0; o < BM; o += sa)
			as[(ar + o) * BK + ac] = a[(ar + o) * n + ac];
		for (int o = 0; o < BK; o += sb)
			bs[(br + o) * BN + bc] = b[(br + o) * n + bc];
		__syncthreads();
		a += BK;
		b += BK * n;
		for (int k = 0; k < BK; k++) {
			for (int m = 0; m < TM; m++)
				ra[m] = as[(tr * TM + m) * BK + k];
			for (int j = 0; j < TN; j++)
				rb[j] = bs[k * BN + tc * TN + j];
			for (int m = 0; m < TM; m++)
				for (int j = 0; j < TN; j++)
					acc[m * TN + j] += ra[m] * rb[j];
		}
		__syncthreads();
	}
	for (int m = 0; m < TM; m++)
		for (int j = 0; j < TN; j++)
			c[(tr * TM + m) * n + tc * TN + j] = acc[m * TN + j];
}

template <int BM, int BN, int BK, int TM, int TN>
__global__ void __launch_bounds__((BM / TM) * (BN / TN)) mm_vec4(const float *a, const float *b, float *c, int n)
{
	__shared__ float as[BK * BM], bs[BK * BN];
	int tc = threadIdx.x % (BN / TN);
	int tr = threadIdx.x / (BN / TN);
	a += blockIdx.y * BM * n;
	b += blockIdx.x * BN;
	c += blockIdx.y * BM * n + blockIdx.x * BN;
	int ar = threadIdx.x / (BK / 4), ac = threadIdx.x % (BK / 4);
	int br = threadIdx.x / (BN / 4), bc = threadIdx.x % (BN / 4);
	float acc[TM * TN] = {0.f};
	float ra[TM], rb[TN];
	for (int t = 0; t < n; t += BK) {
		float4 v = *reinterpret_cast<const float4 *>(&a[ar * n + ac * 4]);
		as[(ac * 4 + 0) * BM + ar] = v.x;
		as[(ac * 4 + 1) * BM + ar] = v.y;
		as[(ac * 4 + 2) * BM + ar] = v.z;
		as[(ac * 4 + 3) * BM + ar] = v.w;
		*reinterpret_cast<float4 *>(&bs[br * BN + bc * 4]) = *reinterpret_cast<const float4 *>(&b[br * n + bc * 4]);
		__syncthreads();
		a += BK;
		b += BK * n;
		for (int k = 0; k < BK; k++) {
			for (int m = 0; m < TM; m++)
				ra[m] = as[k * BM + tr * TM + m];
			for (int j = 0; j < TN; j++)
				rb[j] = bs[k * BN + tc * TN + j];
			for (int m = 0; m < TM; m++)
				for (int j = 0; j < TN; j++)
					acc[m * TN + j] += ra[m] * rb[j];
		}
		__syncthreads();
	}
	for (int m = 0; m < TM; m++)
		for (int j = 0; j < TN; j += 4)
			*reinterpret_cast<float4 *>(&c[(tr * TM + m) * n + tc * TN + j]) =
				make_float4(acc[m * TN + j], acc[m * TN + j + 1], acc[m * TN + j + 2], acc[m * TN + j + 3]);
}

static void need_multiple(int n, int m)
{
	if (n % m) {
		fprintf(stderr, "this kernel needs n to be a multiple of %d\n", m);
		exit(1);
	}
}

static void cublas_mm(cublasHandle_t h, const float *a, const float *b, float *c, int n)
{
	float one = 1.f, zero = 0.f;
	CKB(cublasSgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &one, b, n, a, n, &zero, c, n));
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

	cublasHandle_t h;
	CKB(cublasCreate(&h));

	float ms;
	if (args.kernel == "cublas") {
		ms = time_ms(args, [&] { cublas_mm(h, a, b, c, n); });
	} else if (args.kernel == "naive") {
		dim3 bl(32, 32);
		dim3 g((n + 31) / 32, (n + 31) / 32);
		ms = time_ms(args, [&] { mm_naive<<<g, bl>>>(a, b, c, n); });
	} else if (args.kernel == "tiled") {
		dim3 bl(32, 32);
		dim3 g((n + 31) / 32, (n + 31) / 32);
		ms = time_ms(args, [&] { mm_tiled<32><<<g, bl>>>(a, b, c, n); });
	} else if (args.kernel == "reg1d") {
		need_multiple(n, 64);
		dim3 g(n / 64, n / 64);
		ms = time_ms(args, [&] { mm_reg1d<64, 64, 8, 8><<<g, 64 * 64 / 8>>>(a, b, c, n); });
	} else if (args.kernel == "reg2d") {
		need_multiple(n, 128);
		dim3 g(n / 128, n / 128);
		ms = time_ms(args, [&] { mm_reg2d<128, 128, 8, 8, 8><<<g, 256>>>(a, b, c, n); });
	} else if (args.kernel == "vec4") {
		need_multiple(n, 128);
		dim3 g(n / 128, n / 128);
		ms = time_ms(args, [&] { mm_vec4<128, 128, 8, 8, 8><<<g, 256>>>(a, b, c, n); });
	} else {
		fprintf(stderr, "unknown kernel %s\n", args.kernel.c_str());
		return 1;
	}

	bool ok = true;
	if (args.check) {
		CK(cudaMemcpy(hc.data(), c, nn * sizeof(float), cudaMemcpyDeviceToHost));
		std::vector<float> want(nn);
		if (n <= 1024 || args.kernel == "cublas") {
			cpu_ref(ha, hb, want, n);
		} else {
			float *r;
			CK(cudaMalloc(&r, nn * sizeof(float)));
			cublas_mm(h, a, b, r, n);
			CK(cudaMemcpy(want.data(), r, nn * sizeof(float), cudaMemcpyDeviceToHost));
			cudaFree(r);
		}
		ok = close_enough(hc, want, 1e-3f);
	}

	double gflops = 2.0 * n * n * n / (ms * 1e6);
	report(args, ms, "gflops", gflops, ok);

	cublasDestroy(h);
	cudaFree(a);
	cudaFree(b);
	cudaFree(c);
	return ok ? 0 : 1;
}
