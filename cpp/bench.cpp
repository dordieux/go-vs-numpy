// C++ side of the float64 kernel benchmark: naive loops compiled with -O3,
// so whatever vectorization happens is the compiler's own doing.
// Emits the same JSON-lines schema as the Go and Python harnesses.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <functional>
#include <string>
#include <vector>

static constexpr double kAlpha = 1.0000001;

// Knuth multiplicative hash of the index mapped into [-100, 100).
static std::vector<double> make_input(size_t n) {
  std::vector<double> out(n);
  for (size_t i = 0; i < n; i++) {
    uint64_t h = (static_cast<uint64_t>(i) * 2654435761ULL) & 0xFFFFFFFFULL;
    out[i] = static_cast<double>(h) / 4294967296.0 * 200 - 100;
  }
  return out;
}

static void clip_op(std::vector<double>& dst, const std::vector<double>& src) {
  for (size_t i = 0; i < src.size(); i++) dst[i] = src[i] < 0 ? 0 : src[i];
}
static void add_op(std::vector<double>& dst, const std::vector<double>& a,
                   const std::vector<double>& b) {
  for (size_t i = 0; i < a.size(); i++) dst[i] = a[i] + b[i];
}
static void axpy_op(std::vector<double>& dst, const std::vector<double>& x,
                    const std::vector<double>& y) {
  for (size_t i = 0; i < x.size(); i++) dst[i] = kAlpha * x[i] + y[i];
}
static void sqrt_op(std::vector<double>& dst, const std::vector<double>& src) {
  for (size_t i = 0; i < src.size(); i++) dst[i] = std::sqrt(src[i]);
}
static void exp_op(std::vector<double>& dst, const std::vector<double>& src) {
  for (size_t i = 0; i < src.size(); i++) dst[i] = std::exp(src[i]);
}
static double sum_op(const std::vector<double>& src) {
  double s = 0;
  for (double v : src) s += v;
  return s;
}

static volatile double g_sink;

static double digest(const std::vector<double>& dst) {
  double s = 0;
  for (double v : dst) s += v;
  return s;
}

struct Timing {
  double ns_min;
  double ns_median;
  int reps;
};

static Timing measure(const std::function<void()>& fn, double min_time_s,
                      int min_reps, int max_reps) {
  fn();  // warmup
  std::vector<double> times;
  double total = 0;
  while ((total < min_time_s || static_cast<int>(times.size()) < min_reps) &&
         static_cast<int>(times.size()) < max_reps) {
    auto start = std::chrono::steady_clock::now();
    fn();
    auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                  std::chrono::steady_clock::now() - start)
                  .count();
    total += static_cast<double>(ns) / 1e9;
    times.push_back(static_cast<double>(ns));
  }
  std::sort(times.begin(), times.end());
  return {times.front(), times[times.size() / 2],
          static_cast<int>(times.size())};
}

int main(int argc, char** argv) {
  std::string ops_arg = "clip,add,axpy,sqrt,exp,sum";
  std::string sizes_arg = "1000,10000,100000,1000000,10000000,100000000";
  double min_time_s = 0.3;
  int min_reps = 5, max_reps = 10000;
  for (int i = 1; i + 1 < argc; i += 2) {
    std::string flag = argv[i];
    if (flag == "--ops") ops_arg = argv[i + 1];
    else if (flag == "--sizes") sizes_arg = argv[i + 1];
    else if (flag == "--mintime") min_time_s = std::stod(argv[i + 1]);
    else if (flag == "--minreps") min_reps = std::stoi(argv[i + 1]);
    else if (flag == "--maxreps") max_reps = std::stoi(argv[i + 1]);
  }

  auto split = [](const std::string& s) {
    std::vector<std::string> out;
    size_t start = 0;
    while (start <= s.size()) {
      size_t comma = s.find(',', start);
      if (comma == std::string::npos) comma = s.size();
      out.push_back(s.substr(start, comma - start));
      start = comma + 1;
    }
    return out;
  };

  std::string meta = "cpp=" __VERSION__;
#if defined(__aarch64__)
  meta += " arch=arm64";
#elif defined(__x86_64__)
  meta += " arch=amd64";
#endif
  std::fprintf(stderr, "# %s\n", meta.c_str());

  for (const std::string& size_str : split(sizes_arg)) {
    size_t n = static_cast<size_t>(std::stoll(size_str));
    std::vector<double> x = make_input(n);
    std::vector<double> y = x;
    std::reverse(y.begin(), y.end());
    std::vector<double> dst(n);

    for (const std::string& op : split(ops_arg)) {
      std::vector<double> src = x;
      if (op == "sqrt") {
        for (size_t i = 0; i < n; i++) src[i] = std::fabs(x[i]);
      } else if (op == "exp") {
        for (size_t i = 0; i < n; i++) src[i] = x[i] / 100;
      }

      // The timed fn contains the kernel only; the checksum digest runs once
      // afterwards. g_sink defeats dead-code elimination for reductions.
      std::function<void()> fn;
      std::function<double()> check;
      if (op == "clip") { fn = [&] { clip_op(dst, src); }; check = [&] { return digest(dst); }; }
      else if (op == "add") { fn = [&] { add_op(dst, x, y); }; check = [&] { return digest(dst); }; }
      else if (op == "axpy") { fn = [&] { axpy_op(dst, x, y); }; check = [&] { return digest(dst); }; }
      else if (op == "sqrt") { fn = [&] { sqrt_op(dst, src); }; check = [&] { return digest(dst); }; }
      else if (op == "exp") { fn = [&] { exp_op(dst, src); }; check = [&] { return digest(dst); }; }
      else if (op == "sum") { fn = [&] { g_sink = sum_op(src); }; check = [&] { return static_cast<double>(g_sink); }; }
      else { std::fprintf(stderr, "unknown op: %s\n", op.c_str()); return 1; }

      Timing t = measure(fn, min_time_s, min_reps, max_reps);
      double checksum = check();
      std::printf(
          "{\"lang\":\"cpp\",\"impl\":\"scalar_O3\",\"op\":\"%s\",\"n\":%zu,"
          "\"ns_min\":%.0f,\"ns_median\":%.0f,\"reps\":%d,\"checksum\":%.17g,"
          "\"meta\":\"%s\"}\n",
          op.c_str(), n, t.ns_min, t.ns_median, t.reps, checksum, meta.c_str());
      std::fflush(stdout);
      std::fprintf(stderr, "cpp scalar_O3 %-5s n=%-10zu min=%12.0fns reps=%d\n",
                   op.c_str(), n, t.ns_min, t.reps);
    }
  }
  return 0;
}
