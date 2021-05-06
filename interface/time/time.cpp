#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <vector>

#include "model_fns.hpp"

std::string rpt(int n, const std::string &strg) {
  std::string out;
  for (int i{0}; i < n; ++i) {
    out += strg;
  }
  return out;
}

template <std::size_t c>
std::string hline(const std::array<int, c> &cw, std::string out,
                  const std::string &middle, const std::string &right,
                  const std::string &spacer = "─") {
  for (std::size_t i{0}; i < cw.size() - 1; ++i) {
    out += rpt(cw[i], spacer) + middle;
  }
  return out + rpt(cw.back(), spacer) + right + "\n";
}

void run(const int num) {
  const int pspoints{2};
  constexpr int cols{6};
  const std::array<std::string, cols> titles{
      {"pt", "val f32", "val+err f32", "val f64", "val+err f64", "f64/f32"}};
  std::array<int, cols> cw;
  std::transform(titles.cbegin(), titles.cend(), cw.begin(),
                 [](std::string s) -> int { return s.size(); });

  std::cout << '\n'
            << "n3jet: benchmark pretrained neural network C++ inference timing" << '\n'
            << "       showing mean of " << num << " runs for " << pspoints << " points"
            << '\n'
            << "       there sometimes seems to be a warmup effect where val+err is "
               "faster if evaluated"
            << '\n'
            << "       second, but val is faster if evaluated second" << '\n'
            << "       computed in float (f32) and double (f64)" << '\n'
            << "       all times are in microseconds" << '\n'
            << '\n'
            << hline(cw, "┌", "┬", "┐");

  std::cout << "│";
  for (int i{0}; i < cols; ++i) {
    std::cout << std::setw(cw[i]) << titles[i] << "│";
  }
  std::cout << '\n' << hline(cw, "├", "┼", "┤");

  using TP = std::chrono::high_resolution_clock::time_point;
  TP t0, t1;

  const int legs{5};
  const int training_reruns{20};
  const double delta{0.02};
  const std::string model_dir{
      "../../models/3g2A/RAMBO/"
      "100k_unit_002/"};
  const std::string cut_dir{"cut_0.02/"};

  // raw momenta input
  std::vector<std::vector<std::vector<double>>> momenta_f64{{
      {
          {80.80323390148537, 0.0, 0.0, 80.8032339014854},
          {53.57113343015938, 0.0, 0.0, -53.571133430159385},
          {42.47353254764989, 26.671855251549694, 29.511816381880454,
           14.88844512898612},
          {52.68325739804988, -51.14726465398362, 1.1299491037855054,
           12.578002365534454},
          {39.21757738594496, 24.475409402433904, -30.641765485665946,
           -0.2343470231945908},
      },
      {
          {88.48800215647898, 0.0, 0.0, 88.48800215647898},
          {30.862002335592713, 0.0, 0.0, -30.862002335592702},
          {37.52777986769674, -20.535557255118455, 24.668353258793047,
           19.444729299207314},
          {58.90087277950353, 24.239611461766692, -45.272650711571636,
           28.846856811754705},
          {22.921351844871374, -3.704054206648225, 20.604297452778603,
           9.334413709924226},
      },
  }};

  std::vector<std::vector<std::vector<float>>> momenta_f32(
      pspoints, std::vector<std::vector<float>>(legs, std::vector<float>(4)));
  for (int k{0}; k < pspoints; ++k) {
    for (int j{0}; j < legs; ++j) {
      for (int i{0}; i < 4; ++i) {
        momenta_f32[k][j][i] = static_cast<float>(momenta_f64[k][j][i]);
      }
    }
  }

  nn::FKSEnsemble<float> ensemble_f32(legs, training_reruns, model_dir, delta, cut_dir);

  nn::FKSEnsemble<double> ensemble_f64(legs, training_reruns, model_dir, delta,
                                       cut_dir);

  std::cout << std::fixed;

  for (int i{0}; i < pspoints; ++i) {

    t0 = std::chrono::high_resolution_clock::now();
    for (int j{0}; j < num; ++j) {
      ensemble_f32.compute(momenta_f32[i]);
    };
    t1 = std::chrono::high_resolution_clock::now();
    const long int f32dur{
        std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count()};
    const double avf32{static_cast<double>(f32dur) / num};

    t0 = std::chrono::high_resolution_clock::now();
    for (int j{0}; j < num; ++j) {
      ensemble_f32.compute_with_error(momenta_f32[i]);
    };
    t1 = std::chrono::high_resolution_clock::now();
    const long int f32dur2{
        std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count()};
    const double avf32e{static_cast<double>(f32dur2) / num};

    t0 = std::chrono::high_resolution_clock::now();
    for (int j{0}; j < num; ++j) {
      ensemble_f64.compute(momenta_f64[i]);
    };
    t1 = std::chrono::high_resolution_clock::now();
    const long int f64dur{
        std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count()};
    const double avf64{static_cast<double>(f64dur) / num};

    t0 = std::chrono::high_resolution_clock::now();
    for (int j{0}; j < num; ++j) {
      ensemble_f64.compute_with_error(momenta_f64[i]);
    };
    t1 = std::chrono::high_resolution_clock::now();
    const long int f64dur2{
        std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count()};
    const double avf64e{static_cast<double>(f64dur2) / num};

    const double rto{avf64 / avf32};

    std::cout << std::setprecision(1) << "│" << std::setw(cw[0]) << i << "│"
              << std::setw(cw[1]) << avf32 << "│" << std::setw(cw[2]) << avf32e << "│"
              << std::setw(cw[3]) << avf64 << "│" << std::setw(cw[4]) << avf64e << "│"
              << std::setw(cw[5]) << std::setprecision(2) << rto << "│" << '\n';
  }

  std::cout << hline(cw, "└", "┴", "┘");
}

int main() {
  run(1000);
  run(10000);
}
