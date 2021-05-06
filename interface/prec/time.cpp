#include <algorithm>
#include <array>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <vector>

#include "ngluon2/Mom.h"
#include "ngluon2/refine.h"
#include "tools/PhaseSpace.h"

#include "model_fns.hpp"

double d(double a, double b) { return 2 * std::abs((a - b) / (a + b)); }

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

void run(const int start, const int end) {
  constexpr int cols{4};
  const std::array<std::string, cols> titles{
      {"pt", "err f32", "err f64", "d(f64,f32)"}};
  std::array<int, cols> cw;
  std::transform(titles.cbegin(), titles.cend(), cw.begin(),
                 [](std::string s) -> int { return s.size(); });

  std::cout << '\n' << hline(cw, "┌", "┬", "┐");

  std::cout << "│";
  for (int i{0}; i < cols; ++i) {
    std::cout << std::setw(cw[i]) << titles[i] << "│";
  }
  std::cout << '\n' << hline(cw, "├", "┼", "┤");

  const std::vector<double> scales2{{0}};
  const int legs{5};
  const int training_reruns{20};
  const double delta{0.02};
  const std::string model_dir{"../../models/3g2A/RAMBO/"
                              "100k_unit_002/"};
  const std::string cut_dir{"cut_0.02/"};

  nn::FKSEnsemble<float> ensemble_f32(legs, training_reruns, model_dir, delta, cut_dir);

  nn::FKSEnsemble<double> ensemble_f64(legs, training_reruns, model_dir, delta,
                                       cut_dir);

  for (int p{start}; p < end; ++p) {

    PhaseSpace<double> ps(legs, p);
    std::vector<MOM<double>> momenta_njet{ps.getPSpoint()};
    refineM(momenta_njet, momenta_njet, scales2);

    std::vector<std::vector<double>> momenta_f64(legs, std::vector<double>(4));
    for (int j{0}; j < legs; ++j) {
      momenta_f64[j][0] = momenta_njet[j].x0;
      momenta_f64[j][1] = momenta_njet[j].x1;
      momenta_f64[j][2] = momenta_njet[j].x2;
      momenta_f64[j][3] = momenta_njet[j].x3;
    }

    std::vector<std::vector<float>> momenta_f32(legs, std::vector<float>(4));
    for (int j{0}; j < legs; ++j) {
      for (int i{0}; i < 4; ++i) {
        momenta_f32[j][i] = static_cast<float>(momenta_f64[j][i]);
      }
    }

    ensemble_f32.compute_with_error(momenta_f32);

    ensemble_f64.compute_with_error(momenta_f64);

    std::cout << "│" << std::setw(cw[0]) << p << "│" << std::setw(cw[1])
              << ensemble_f32.std_err << "│" << std::setw(cw[2]) << ensemble_f64.std_err
              << "│" << std::setw(cw[3]) << d(ensemble_f32.mean, ensemble_f64.mean)
              << "│" << '\n';
  }

  std::cout << hline(cw, "└", "┴", "┘");
}

int main(int argc, char *argv[]) {
  std::cout << std::scientific << std::setprecision(1);

  std::cout << '\n'
            << "Comparison of neural net error with inference at different numerical "
               "precisions"
            << '\n'
            << '\n';

  int start;
  int end;

  if (argc == 2) {
    start = std::atoi(argv[1]);
    end = start + 1;
  } else if (argc == 3) {
    start = std::atoi(argv[1]);
    end = std::atoi(argv[2]);
  } else {
    std::cerr
        << "Error: run as `./test <initial rseed> <final rseed (exclusive)>`, where "
           "rseed is the random number seed for the phase space point generator."
        << '\n'
        << "If <final seed> is omitted, only <initial seed> will be evaluated." << '\n';
    std::exit(EXIT_FAILURE);
  }

  if (start < 1) {
    std::cerr << "Error: start must be greater than zero!" << '\n';
    std::exit(EXIT_FAILURE);
  }

  run(start, end);

  std::cout << '\n';
}
