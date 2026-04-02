#pragma once

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

struct Args {
	std::string kernel = "naive";
	long n = 0;
	int iters = 20;
	int warmup = 3;
	bool check = true;
};

inline Args parse(int argc, char **argv, long default_n, int default_iters = 20, int default_warmup = 3)
{
	Args a;
	a.n = default_n;
	a.iters = default_iters;
	a.warmup = default_warmup;
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

inline void report(const Args &a, float ms, const char *rate_name, double rate, bool ok)
{
	printf("kernel=%s n=%ld time_ms=%.4f %s=%.2f ok=%d\n", a.kernel.c_str(), a.n, ms, rate_name, rate, ok ? 1 : 0);
}
