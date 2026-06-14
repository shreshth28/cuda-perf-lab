#include "cl_common.h"

static const char *src = R"(
__kernel void add_naive(__global const float *a, __global const float *b, __global float *c, long n)
{
	long i = get_global_id(0);
	if (i < n)
		c[i] = a[i] + b[i];
}

__kernel void add_vec4(__global const float4 *a, __global const float4 *b, __global float4 *c, long n4,
		       __global const float *as, __global const float *bs, __global float *cs, long n)
{
	long i = get_global_id(0);
	if (i < n4)
		c[i] = a[i] + b[i];
	long t = n4 * 4 + i;
	if (i < 4 && t < n)
		cs[t] = as[t] + bs[t];
}
)";

static size_t round_up(size_t x, size_t m) { return (x + m - 1) / m * m; }

int main(int argc, char **argv)
{
	Args args = parse(argc, argv, 1L << 24);
	long n = args.n;

	std::vector<float> ha(n), hb(n), hc(n);
	fill(ha, 1);
	fill(hb, 2);

	Cl cl;
	cl_program prog = cl.build(src);
	cl_mem a = cl.buf(n * sizeof(float), ha.data());
	cl_mem b = cl.buf(n * sizeof(float), hb.data());
	cl_mem c = cl.buf(n * sizeof(float));

	size_t ls = 256;
	float ms;
	if (args.kernel == "naive") {
		cl_kernel k = cl.kernel(prog, "add_naive");
		set_args(k, a, b, c, (cl_long)n);
		ms = cl_time_ms(cl, args, [&] { return launch1d(cl, k, round_up(n, ls), ls); });
		clReleaseKernel(k);
	} else if (args.kernel == "vec4") {
		cl_kernel k = cl.kernel(prog, "add_vec4");
		cl_long n4 = n / 4;
		set_args(k, a, b, c, n4, a, b, c, (cl_long)n);
		ms = cl_time_ms(cl, args, [&] { return launch1d(cl, k, round_up(n4 > 4 ? n4 : 4, ls), ls); });
		clReleaseKernel(k);
	} else {
		fprintf(stderr, "unknown kernel %s\n", args.kernel.c_str());
		return 1;
	}

	bool ok = true;
	if (args.check) {
		cl.read(c, hc.data(), n * sizeof(float));
		std::vector<float> want(n);
		for (long i = 0; i < n; i++)
			want[i] = ha[i] + hb[i];
		ok = close_enough(hc, want, 1e-6f);
	}

	double gbps = 3.0 * n * sizeof(float) / (ms * 1e6);
	report(args, ms, "gbps", gbps, ok);

	clReleaseMemObject(a);
	clReleaseMemObject(b);
	clReleaseMemObject(c);
	clReleaseProgram(prog);
	return ok ? 0 : 1;
}
