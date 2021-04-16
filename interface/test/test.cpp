#include <array>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <vector>

#include "model_fns.hpp"

// template nn::Network<double>::~Network();

double d(double a, double b) { return 2 * std::abs((a - b) / (a + b)); }

int main() {
  std::cout.setf(std::ios_base::scientific, std::ios_base::floatfield);
  std::cout.precision(16);

  std::cout << '\n'
            << "n3jet: test pretrained neural network C++ inference result against "
               "reference Python implementation, showing relative difference d"
            << '\n'
            << '\n';

  const int legs{5};
  const int pspoints{2};
  const int training_reruns{20};
  const double delta{0.02};

  // raw momenta input
  std::vector<std::vector<std::vector<double>>> momenta{{
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

  std::array<double, 2> python_outputs{{2.2266408e-07, 1.430258598666967e-06}};

  nn::FKSEnsemble<double> ensemble(
      legs, training_reruns,
      "../../models/3g2a/RAMBO/"
      "events_100k_fks_all_legs_all_pairs_new_sherpa_cuts_pdf_njet_test/",
      delta, "cut_0.02/");

  for (int i{0}; i < pspoints; ++i) {
    std::cout << "==================== Test point " << i + 1
              << " ====================" << '\n';

    double average_output{ensemble.compute(momenta[i])};

    ensemble.compute_with_error(momenta[i]);

    std::cout << std::setw(17) << "Python = " << python_outputs[i] << '\n'
              << std::setw(17) << "C++ = " << average_output << '\n'
              << std::setw(17) << "C++ 2 = " << ensemble.mean << " +- "
              << ensemble.std_err << '\n'
              << std::setw(17)
              << "d(C++, Python) = " << d(average_output, python_outputs[i]) << '\n';
  }
}
