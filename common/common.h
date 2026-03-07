#pragma once

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
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
