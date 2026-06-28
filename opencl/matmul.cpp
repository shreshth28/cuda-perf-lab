#include <algorithm>

#include "cl_common.h"

static const char *src = R"(
__kernel void mm_naive(__global const float *a, __global const float *b, __global float *c, int n)
{
	int col = get_global_id(0);
	int row = get_global_id(1);
	if (row < n && col < n) {
		float s = 0.f;
		for (int k = 0; k < n; k++)
			s += a[row * n + k] * b[k * n + col];
		c[row * n + col] = s;
	}
}

#define T 16
__kernel void mm_tiled(__global const float *a, __global const float *b, __global float *c, int n)
{
	__local float as[T][T], bs[T][T];
	int tx = get_local_id(0), ty = get_local_id(1);
	int row = get_group_id(1) * T + ty, col = get_group_id(0) * T + tx;
	float s = 0.f;
	for (int t = 0; t < n; t += T) {
		as[ty][tx] = row < n && t + tx < n ? a[row * n + t + tx] : 0.f;
		bs[ty][tx] = t + ty < n && col < n ? b[(t + ty) * n + col] : 0.f;
		barrier(CLK_LOCAL_MEM_FENCE);
		for (int k = 0; k < T; k++)
			s += as[ty][k] * bs[k][tx];
		barrier(CLK_LOCAL_MEM_FENCE);
	}
	if (row < n && col < n)
		c[row * n + col] = s;
}

#define BM 128
#define BN 128
#define BK 8
#define TM 8
#define TN 8
__kernel void mm_reg2d(__global const float *a, __global const float *b, __global float *c, int n)
{
	__local float as[BM * BK], bs[BK * BN];
	const int nt = (BM / TM) * (BN / TN);
	int tid = get_local_id(0);
	int tc = tid % (BN / TN);
	int tr = tid / (BN / TN);
	a += get_group_id(1) * BM * n;
	b += get_group_id(0) * BN;
	c += get_group_id(1) * BM * n + get_group_id(0) * BN;
	int ar = tid / BK, ac = tid % BK, sa = nt / BK;
	int br = tid / BN, bc = tid % BN, sb = nt / BN;
	float acc[TM * TN];
	for (int i = 0; i < TM * TN; i++)
		acc[i] = 0.f;
	float ra[TM], rb[TN];
	for (int t = 0; t < n; t += BK) {
		for (int o = 0; o < BM; o += sa)
			as[(ar + o) * BK + ac] = a[(ar + o) * n + ac];
		for (int o = 0; o < BK; o += sb)
			bs[(br + o) * BN + bc] = b[(br + o) * n + bc];
		barrier(CLK_LOCAL_MEM_FENCE);
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
		barrier(CLK_LOCAL_MEM_FENCE);
	}
	for (int m = 0; m < TM; m++)
		for (int j = 0; j < TN; j++)
			c[(tr * TM + m) * n + tc * TN + j] = acc[m * TN + j];
}
)";

static size_t round_up(size_t x, size_t m) { return (x + m - 1) / m * m; }

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

	Cl cl;
	cl_program prog = cl.build(src);
	cl_mem a = cl.buf(nn * sizeof(float), ha.data());
	cl_mem b = cl.buf(nn * sizeof(float), hb.data());
	cl_mem c = cl.buf(nn * sizeof(float));

	float ms;
	if (args.kernel == "naive" || args.kernel == "tiled") {
		cl_kernel k = cl.kernel(prog, ("mm_" + args.kernel).c_str());
		set_args(k, a, b, c, (cl_int)n);
		ms = cl_time_ms(cl, args, [&] {
			return std::vector<cl_event>{launch2d(cl, k, round_up(n, 16), round_up(n, 16), 16, 16)};
		});
		clReleaseKernel(k);
	} else if (args.kernel == "reg2d") {
		if (n % 128) {
			fprintf(stderr, "reg2d needs n to be a multiple of 128\n");
			return 1;
		}
		cl_kernel k = cl.kernel(prog, "mm_reg2d");
		set_args(k, a, b, c, (cl_int)n);
		ms = cl_time_ms(cl, args, [&] {
			return std::vector<cl_event>{launch2d(cl, k, (n / 128) * 256, n / 128, 256, 1)};
		});
		clReleaseKernel(k);
	} else {
		fprintf(stderr, "unknown kernel %s\n", args.kernel.c_str());
		return 1;
	}

	bool ok = true;
	if (args.check) {
		cl.read(c, hc.data(), nn * sizeof(float));
		std::vector<float> want(nn);
		cpu_ref(ha, hb, want, n);
		ok = close_enough(hc, want, 1e-3f);
	}

	double gflops = 2.0 * n * n * n / (ms * 1e6);
	report(args, ms, "gflops", gflops, ok);

	clReleaseMemObject(a);
	clReleaseMemObject(b);
	clReleaseMemObject(c);
	clReleaseProgram(prog);
	return ok ? 0 : 1;
}
