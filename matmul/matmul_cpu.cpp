#include <chrono>

#include "args.h"

static void mm_naive(const float *a, const float *b, float *c, int n)
{
	for (int i = 0; i < n; i++)
		for (int j = 0; j < n; j++) {
			float s = 0.f;
			for (int k = 0; k < n; k++)
				s += a[(long)i * n + k] * b[(long)k * n + j];
			c[(long)i * n + j] = s;
		}
}

static void mm_ikj(const float *a, const float *b, float *c, int n)
{
	for (long i = 0; i < (long)n * n; i++)
		c[i] = 0.f;
	for (int i = 0; i < n; i++)
		for (int k = 0; k < n; k++) {
			float aik = a[(long)i * n + k];
			for (int j = 0; j < n; j++)
				c[(long)i * n + j] += aik * b[(long)k * n + j];
		}
}

static void mm_omp(const float *a, const float *b, float *c, int n)
{
#pragma omp parallel for schedule(static)
	for (int i = 0; i < n; i++) {
		float *ci = c + (long)i * n;
		for (int j = 0; j < n; j++)
			ci[j] = 0.f;
		for (int k = 0; k < n; k++) {
			float aik = a[(long)i * n + k];
			const float *bk = b + (long)k * n;
			for (int j = 0; j < n; j++)
				ci[j] += aik * bk[j];
		}
	}
}

static void mm_ref(const std::vector<float> &a, const std::vector<float> &b, std::vector<float> &c, int n)
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
	Args args = parse(argc, argv, 1024, 3, 1);
	int n = args.n;
	long nn = (long)n * n;

	std::vector<float> a(nn), b(nn), c(nn);
	fill(a, 5);
	fill(b, 6);

	void (*f)(const float *, const float *, float *, int);
	if (args.kernel == "naive")
		f = mm_naive;
	else if (args.kernel == "ikj")
		f = mm_ikj;
	else if (args.kernel == "omp")
		f = mm_omp;
	else {
		fprintf(stderr, "unknown kernel %s\n", args.kernel.c_str());
		return 1;
	}

	for (int i = 0; i < args.warmup; i++)
		f(a.data(), b.data(), c.data(), n);
	auto t0 = std::chrono::steady_clock::now();
	for (int i = 0; i < args.iters; i++)
		f(a.data(), b.data(), c.data(), n);
	auto t1 = std::chrono::steady_clock::now();
	float ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / args.iters;

	bool ok = true;
	if (args.check) {
		std::vector<float> want(nn);
		mm_ref(a, b, want, n);
		ok = close_enough(c, want, 1e-3f);
	}

	double gflops = 2.0 * n * n * n / (ms * 1e6);
	report(args, ms, "gflops", gflops, ok);
	return ok ? 0 : 1;
}
