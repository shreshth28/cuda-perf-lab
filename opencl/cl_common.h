#pragma once

#define CL_TARGET_OPENCL_VERSION 120
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#include <CL/cl.h>

#include "args.h"

#define CL(x)                                                                          \
	do {                                                                           \
		cl_int e_ = (x);                                                       \
		if (e_ != CL_SUCCESS) {                                                \
			fprintf(stderr, "%s:%d opencl error %d\n", __FILE__, __LINE__, e_); \
			exit(1);                                                       \
		}                                                                      \
	} while (0)

struct Cl {
	cl_platform_id plat;
	cl_device_id dev;
	cl_context ctx;
	cl_command_queue q;

	Cl()
	{
		cl_uint np;
		CL(clGetPlatformIDs(0, nullptr, &np));
		std::vector<cl_platform_id> ps(np);
		CL(clGetPlatformIDs(np, ps.data(), nullptr));
		bool found = false;
		for (cl_device_type type : {(cl_device_type)CL_DEVICE_TYPE_GPU, (cl_device_type)CL_DEVICE_TYPE_ALL}) {
			for (auto p : ps) {
				if (clGetDeviceIDs(p, type, 1, &dev, nullptr) == CL_SUCCESS) {
					plat = p;
					found = true;
					break;
				}
			}
			if (found)
				break;
		}
		if (!found) {
			fprintf(stderr, "no opencl device\n");
			exit(1);
		}
		cl_int e;
		ctx = clCreateContext(nullptr, 1, &dev, nullptr, nullptr, &e);
		CL(e);
		q = clCreateCommandQueue(ctx, dev, CL_QUEUE_PROFILING_ENABLE, &e);
		CL(e);
	}
	~Cl()
	{
		clReleaseCommandQueue(q);
		clReleaseContext(ctx);
	}

	cl_program build(const char *src)
	{
		cl_int e;
		cl_program p = clCreateProgramWithSource(ctx, 1, &src, nullptr, &e);
		CL(e);
		if (clBuildProgram(p, 1, &dev, "-cl-fast-relaxed-math", nullptr, nullptr) != CL_SUCCESS) {
			size_t len;
			clGetProgramBuildInfo(p, dev, CL_PROGRAM_BUILD_LOG, 0, nullptr, &len);
			std::string log(len, '\0');
			clGetProgramBuildInfo(p, dev, CL_PROGRAM_BUILD_LOG, len, &log[0], nullptr);
			fprintf(stderr, "kernel build failed:\n%s\n", log.c_str());
			exit(1);
		}
		return p;
	}

	cl_kernel kernel(cl_program p, const char *name)
	{
		cl_int e;
		cl_kernel k = clCreateKernel(p, name, &e);
		CL(e);
		return k;
	}

	cl_mem buf(size_t bytes, const void *host = nullptr)
	{
		cl_int e;
		cl_mem m = clCreateBuffer(ctx, CL_MEM_READ_WRITE | (host ? CL_MEM_COPY_HOST_PTR : 0), bytes,
					  const_cast<void *>(host), &e);
		CL(e);
		return m;
	}

	void read(cl_mem m, void *host, size_t bytes)
	{
		CL(clEnqueueReadBuffer(q, m, CL_TRUE, 0, bytes, host, 0, nullptr, nullptr));
	}
};

inline void cl_ok(cl_int e, const char *what)
{
	if (e != CL_SUCCESS) {
		fprintf(stderr, "%s failed: opencl error %d\n", what, e);
		exit(1);
	}
}

template <class... T>
void set_args(cl_kernel k, const T &...a)
{
	cl_uint i = 0;
	(cl_ok(clSetKernelArg(k, i++, sizeof(T), &a), "clSetKernelArg"), ...);
}

template <class F>
float cl_time_ms(Cl &cl, const Args &a, F launch)
{
	for (int i = 0; i < a.warmup; i++)
		for (cl_event e : launch())
			clReleaseEvent(e);
	CL(clFinish(cl.q));
	double total = 0;
	for (int i = 0; i < a.iters; i++) {
		std::vector<cl_event> evs = launch();
		CL(clWaitForEvents(evs.size(), evs.data()));
		cl_ulong t0, t1;
		CL(clGetEventProfilingInfo(evs.front(), CL_PROFILING_COMMAND_START, sizeof(t0), &t0, nullptr));
		CL(clGetEventProfilingInfo(evs.back(), CL_PROFILING_COMMAND_END, sizeof(t1), &t1, nullptr));
		total += (t1 - t0) * 1e-6;
		for (cl_event e : evs)
			clReleaseEvent(e);
	}
	return total / a.iters;
}

inline void local_arg(cl_kernel k, cl_uint i, size_t bytes)
{
	cl_ok(clSetKernelArg(k, i, bytes, nullptr), "clSetKernelArg (local)");
}

inline cl_event launch1d(Cl &cl, cl_kernel k, size_t global, size_t local)
{
	cl_event ev;
	CL(clEnqueueNDRangeKernel(cl.q, k, 1, nullptr, &global, &local, 0, nullptr, &ev));
	return ev;
}

inline cl_event launch2d(Cl &cl, cl_kernel k, size_t gx, size_t gy, size_t lx, size_t ly)
{
	cl_event ev;
	size_t g[2] = {gx, gy}, l[2] = {lx, ly};
	CL(clEnqueueNDRangeKernel(cl.q, k, 2, nullptr, g, l, 0, nullptr, &ev));
	return ev;
}
