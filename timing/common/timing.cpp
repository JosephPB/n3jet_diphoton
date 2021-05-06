#include <array>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

#include "timing.hpp"

void row(const std::array<int, 6> &cw, int p, const std::string &name, double val,
         double err, long dur, double tr) {
  std::cout << ' ' << std::setw(cw[0]) << (p > 0 ? std::to_string(p) : "")
            << std::setw(cw[1]) << name << std::scientific << std::setprecision(16)
            << std::setw(cw[2]) << val << std::setprecision(1) << std::setw(cw[3])
            << err << std::fixed << std::setw(cw[4]) << dur << std::setw(cw[5]) << tr
            << '\n';
}

double mean(const std::vector<long> &data) {
  return static_cast<double>(std::accumulate(data.cbegin(), data.cend(), 0)) /
         data.size();
}

double std_err(const std::vector<long> &data, const double mean) {
  double err{};
  for (long datum : data) {
    const double term{datum - mean};
    err += term * term;
  }
  return std::sqrt(err) / data.size();
}
