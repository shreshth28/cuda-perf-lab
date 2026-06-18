#include "cl_common.h"

static const char *src = R"(
__kernel void sum_tree(__global const float *x, __global float *part, long n, __local float *s)
{
	long i = get_global_id(0);
	int l = get_local_id(0);
	s[l] = i < n ? x[i] : 0.f;
	barrier(CLK_LOCAL_MEM_FENCE);
	for (int k = get_local_size(0) / 2; k > 0; k >>= 1) {
		if (l < k)
			s[l] += s[l + k];
		barrier(CLK_LOCAL_MEM_FENCE);
	}
	if (l == 0)
		part[get_group_id(0)] = s[0];
}

__kernel void sum_final(__global const float *part, __global float *out, int m, __local float *s)
{
	int l = get_local_id(0);
	float v = 0.f;
	for (int i = l; i < m; i += get_local_size(0))
		v += part[i];
	s[l] = v;
	barrier(CLK_LOCAL_MEM_FENCE);
	for (int k = get_local_size(0) / 2; k > 0; k >>= 1) {
		if (l < k)
			s[l] += s[l + k];
		barrier(CLK_LOCAL_MEM_FENCE);
	}
	if (l == 0)
		out[0] = s[0];
}
)";

static size_t round_up(size_t x, size_t m) { return (x + m - 1) / m * m; }

int main(int argc, char **argv)
{
	Args args = parse(argc, argv, 1L << 24);
	long n = args.n;

	std::vector<float> hx(n);
	fill(hx, 3);

	Cl cl;
	cl_program prog = cl.build(src);
	cl_mem x = cl.buf(n * sizeof(float), hx.data());
	cl_mem out = cl.buf(sizeof(float));

	size_t ls = 256;
	cl_kernel fin = cl.kernel(prog, "sum_final");

	float ms;
	cl_mem part = nullptr;
	if (args.kernel == "tree") {
		int groups = (n + ls - 1) / ls;
		part = cl.buf(groups * sizeof(float));
		cl_kernel k = cl.kernel(prog, "sum_tree");
		set_args(k, x, part, (cl_long)n);
		local_arg(k, 3, ls * sizeof(float));
		set_args(fin, part, out, (cl_int)groups);
		local_arg(fin, 3, ls * sizeof(float));
		ms = cl_time_ms(cl, args, [&] {
			return std::vector<cl_event>{launch1d(cl, k, round_up(n, ls), ls), launch1d(cl, fin, ls, ls)};
		});
		clReleaseKernel(k);
	} else {
		fprintf(stderr, "unknown kernel %s\n", args.kernel.c_str());
		return 1;
	}

	bool ok = true;
	if (args.check) {
		float got;
		cl.read(out, &got, sizeof(float));
		double want = 0;
		for (long i = 0; i < n; i++)
			want += hx[i];
		ok = fabs(got - want) <= 1e-3 * fabs(want);
		if (!ok)
			fprintf(stderr, "got %f want %f\n", got, want);
	}

	double gbps = 1.0 * n * sizeof(float) / (ms * 1e6);
	report(args, ms, "gbps", gbps, ok);

	clReleaseKernel(fin);
	clReleaseMemObject(x);
	clReleaseMemObject(out);
	clReleaseMemObject(part);
	clReleaseProgram(prog);
	return ok ? 0 : 1;
}
