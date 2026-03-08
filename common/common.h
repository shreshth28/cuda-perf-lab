#pragma once

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#define CK(x)                                                                          \
	do {                                                                           \
		cudaError_t e_ = (x);                                                  \
		if (e_ != cudaSuccess) {                                               \
			fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); \
			exit(1);                                                       \
		}                                                                      \
	} while (0)

struct Args {
	std::string kernel = "naive";
	long n = 0;
	int iters = 20;
	int warmup = 3;
	bool check = true;
};

inline Args parse(int argc, char **argv, long default_n)
{
	Args a;
	a.n = default_n;
	for (int i = 1; i < argc; i++) {
		std::string s = argv[i];
		auto val = [&]() {
			if (i + 1 >= argc) {
				fprintf(stderr, "%s needs a value\n", s.c_str());
				exit(1);
			}
			return std::string(argv[++i]);
		};
		if (s == "--kernel")
			a.kernel = val();
		else if (s == "--n")
			a.n = std::stol(val());
		else if (s == "--iters")
			a.iters = std::stoi(val());
		else if (s == "--warmup")
			a.warmup = std::stoi(val());
		else if (s == "--no-check")
			a.check = false;
		else {
			fprintf(stderr, "unknown arg %s\n", s.c_str());
			exit(1);
		}
	}
	return a;
}

struct GpuTimer {
	cudaEvent_t a, b;
	GpuTimer()
	{
		CK(cudaEventCreate(&a));
		CK(cudaEventCreate(&b));
	}
	~GpuTimer()
	{
		cudaEventDestroy(a);
		cudaEventDestroy(b);
	}
	void start() { CK(cudaEventRecord(a)); }
	float stop()
	{
		CK(cudaEventRecord(b));
		CK(cudaEventSynchronize(b));
		float ms;
		CK(cudaEventElapsedTime(&ms, a, b));
		return ms;
	}
};

inline void fill(std::vector<float> &v, unsigned seed)
{
	std::mt19937 g(seed);
	std::uniform_real_distribution<float> d(0.f, 1.f);
	for (auto &x : v)
		x = d(g);
}

inline bool close_enough(const std::vector<float> &got, const std::vector<float> &want, float tol)
{
	for (size_t i = 0; i < got.size(); i++) {
		float d = fabsf(got[i] - want[i]);
		if (d > tol * fmaxf(1.f, fabsf(want[i]))) {
			fprintf(stderr, "mismatch at %zu: got %f want %f\n", i, got[i], want[i]);
			return false;
		}
	}
	return true;
}

template <class F>
float time_ms(const Args &a, F f)
{
	for (int i = 0; i < a.warmup; i++)
		f();
	CK(cudaDeviceSynchronize());
	GpuTimer t;
	t.start();
	for (int i = 0; i < a.iters; i++)
		f();
	float ms = t.stop();
	CK(cudaGetLastError());
	return ms / a.iters;
}

inline void report(const Args &a, float ms, const char *rate_name, double rate, bool ok)
{
	printf("kernel=%s n=%ld time_ms=%.4f %s=%.2f ok=%d\n", a.kernel.c_str(), a.n, ms, rate_name, rate, ok ? 1 : 0);
}
