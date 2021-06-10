#include <array>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <vector>

#include "model_fns.hpp"

double d(double a, double b) { return 2 * std::abs((a - b) / (a + b)); }

int main() {
  const double cutoff{1e-15};

  std::cout << '\n'
            << "n3jet: test inference calls to single neural networks through C++ "
               "interface against reference Python implementation result"
            << '\n'
            << "       tests two phase space points" << '\n'
            << "       all computations in f64" << '\n'
            << "       tests relative difference d < " << cutoff << ", where" << '\n'
            << "           d(a, b) = 2*|(a-b)/(a+b)|" << '\n'
            << '\n';

  const int legs{5};
  const int pspoints{2};
  const int training_reruns{20};
  const double delta{0.02};
  const std::string model_dir{"../../models/3g2A/RAMBO/"
                              "100k_unit_002_fks_test/"};
  const std::string cut_dir{"cut_0.02/"};

  const std::vector<std::vector<std::vector<double>>> momenta_f64{{
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

  std::array<double, 2> python_outputs{
      {2.2266415334598327e-07, 1.4302586769827208e-06}};

  nn::FKSEnsemble<double> ensemble_f64(legs, training_reruns, model_dir, delta,
                                       cut_dir);

  for (int i{0}; i < pspoints; ++i) {
    std::cout << "==================== Test point " << i + 1
              << " ====================" << '\n';

    double res{};
    for (int a{0}; a < training_reruns; ++a) {
      res += ensemble_f64.compute_single(momenta_f64[i], a);
    }
    res /= training_reruns;

    const double diff{d(res, python_outputs[i])};

    const int cw{20};

    std::cout << std::scientific << std::setprecision(16) << std::setw(cw)
              << "Python = " << python_outputs[i] << '\n'
              << std::setw(cw) << "C++ = " << res << '\n'
              << std::setw(cw) << "d(C++, Python) = " << std::setprecision(0) << diff
              << '\n'
              << std::setw(cw) << "" << (diff < cutoff ? "Pass" : "Fail") << '\n';
  }
}
