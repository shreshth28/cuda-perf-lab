#pragma once

#include "args.h"

#include <cublas_v2.h>
#include <cuda_runtime.h>

#define CK(x)                                                                          \
	do {                                                                           \
		cudaError_t e_ = (x);                                                  \
		if (e_ != cudaSuccess) {                                               \
			fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); \
			exit(1);                                                       \
		}                                                                      \
	} while (0)

#define CKB(x)                                                                         \
	do {                                                                           \
		cublasStatus_t s_ = (x);                                               \
		if (s_ != CUBLAS_STATUS_SUCCESS) {                                     \
			fprintf(stderr, "%s:%d cublas error %d\n", __FILE__, __LINE__, (int)s_); \
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
