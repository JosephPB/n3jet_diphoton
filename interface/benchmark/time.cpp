#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <vector>

#include "ngluon2/Mom.h"
#include "ngluon2/refine.h"
#include "tools/PhaseSpace.h"

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
      "../../models/3g2a/RAMBO/"
      "events_100k_fks_all_legs_all_pairs_new_sherpa_cuts_pdf_njet_test/"};
  const std::string cut_dir{"cut_0.02/"};

  nn::FKSEnsemble<float> ensemble_f32(legs, training_reruns, model_dir, delta, cut_dir);

  nn::FKSEnsemble<double> ensemble_f64(legs, training_reruns, model_dir, delta,
                                       cut_dir);

  std::cout << std::fixed;

  const std::vector<double> scales2{{0}};

  for (int i{0}; i < pspoints; ++i) {

    const int rseed{i + 1};
    PhaseSpace<double> ps(legs, rseed);
    std::vector<MOM<double>> njet_mom{ps.getPSpoint()};
    refineM(njet_mom, njet_mom, scales2);

    std::vector<std::vector<double>> nn_mom(legs, std::vector<double>(4));
    for (int j{0}; j < legs; ++j) {
      nn_mom[j] = {njet_mom[j].x0, njet_mom[j].x1, njet_mom[j].x2, njet_mom[j].x3};
    }

    std::vector<std::vector<float>> f32_mom(legs, std::vector<float>(4));
    for (int j{0}; j < legs; ++j) {
      for (int k{0}; k < 4; ++k) {
        f32_mom[j][k] = static_cast<float>(nn_mom[j][k]);
      }
    }

    t0 = std::chrono::high_resolution_clock::now();
    for (int j{0}; j < num; ++j) {
      ensemble_f32.compute_with_error(f32_mom);
    };
    t1 = std::chrono::high_resolution_clock::now();
    const long int f32dur2{
        std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count()};
    const double avf32e{static_cast<double>(f32dur2) / num};

    t0 = std::chrono::high_resolution_clock::now();
    for (int j{0}; j < num; ++j) {
      ensemble_f64.compute_with_error(nn_mom);
    };
    t1 = std::chrono::high_resolution_clock::now();
    const long int f64dur2{
        std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count()};
    const double avf64e{static_cast<double>(f64dur2) / num};


    std::cout << std::setprecision(1) << "│" << std::setw(cw[0]) << i << "│"
              << std::setw(cw[1]) << avf32e << "│" << std::setw(cw[2]) << avf32e << "│"
              << std::setw(cw[3]) << avf64e << "│" << std::setw(cw[4]) << avf64e << "│"
              << std::setw(cw[5]) << std::setprecision(2) << 1. << "│" << '\n';
  }

  std::cout << hline(cw, "└", "┴", "┘");
}

int main() {
  run(1000);
  run(10000);
}
