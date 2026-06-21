#include "cl_common.h"

static const char *src = R"(
__kernel void tr_naive(__global const float *in, __global float *out, int n)
{
	int x = get_global_id(0);
	int y = get_global_id(1);
	if (x < n && y < n)
		out[(long)x * n + y] = in[(long)y * n + x];
}

__kernel void tr_tiled(__global const float *in, __global float *out, int n)
{
	__local float t[32][32];
	int tx = get_local_id(0), ty = get_local_id(1);
	int x = get_group_id(0) * 32 + tx;
	int y = get_group_id(1) * 32 + ty;
	for (int j = 0; j < 32; j += 8)
		if (x < n && y + j < n)
			t[ty + j][tx] = in[(long)(y + j) * n + x];
	barrier(CLK_LOCAL_MEM_FENCE);
	x = get_group_id(1) * 32 + tx;
	y = get_group_id(0) * 32 + ty;
	for (int j = 0; j < 32; j += 8)
		if (x < n && y + j < n)
			out[(long)(y + j) * n + x] = t[tx][ty + j];
}
)";

static size_t round_up(size_t x, size_t m) { return (x + m - 1) / m * m; }

int main(int argc, char **argv)
{
	Args args = parse(argc, argv, 4096);
	int n = args.n;
	long nn = (long)n * n;

	std::vector<float> hin(nn), hout(nn);
	fill(hin, 4);

	Cl cl;
	cl_program prog = cl.build(src);
	cl_mem in = cl.buf(nn * sizeof(float), hin.data());
	cl_mem out = cl.buf(nn * sizeof(float));

	float ms;
	if (args.kernel == "naive" || args.kernel == "tiled") {
		cl_kernel k = cl.kernel(prog, args.kernel == "naive" ? "tr_naive" : "tr_tiled");
		set_args(k, in, out, (cl_int)n);
		size_t gy = args.kernel == "naive" ? round_up(n, 8) : round_up(n, 32) / 4;
		ms = cl_time_ms(cl, args, [&] { return std::vector<cl_event>{launch2d(cl, k, round_up(n, 32), gy, 32, 8)}; });
		clReleaseKernel(k);
	} else {
		fprintf(stderr, "unknown kernel %s\n", args.kernel.c_str());
		return 1;
	}

	bool ok = true;
	if (args.check) {
		cl.read(out, hout.data(), nn * sizeof(float));
		std::vector<float> want(nn);
		for (int y = 0; y < n; y++)
			for (int x = 0; x < n; x++)
				want[(long)x * n + y] = hin[(long)y * n + x];
		ok = close_enough(hout, want, 0.f);
	}

	double gbps = 2.0 * nn * sizeof(float) / (ms * 1e6);
	report(args, ms, "gbps", gbps, ok);

	clReleaseMemObject(in);
	clReleaseMemObject(out);
	clReleaseProgram(prog);
	return ok ? 0 : 1;
}
