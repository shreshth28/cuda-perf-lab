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
