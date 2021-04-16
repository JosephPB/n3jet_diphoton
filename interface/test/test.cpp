#include <array>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <vector>

#include "model_fns.hpp"

// template nn::Network<double>::~Network();

double d(double a, double b) { return 2 * std::abs((a - b) / (a + b)); }

int main() {
  std::cout << '\n'
            << "n3jet: test pretrained neural network C++ inference result against "
               "reference Python implementation, showing relative difference d"
            << '\n'
            << '\n';

  const int legs{5};
  const int pspoints{2};
  const int training_reruns{20};
  const double delta{0.02};
  const std::string model_dir{
      "../../models/3g2a/RAMBO/"
      "events_100k_fks_all_legs_all_pairs_new_sherpa_cuts_pdf_njet_test/"};
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

  std::vector<std::vector<std::vector<float>>> momenta_f32(
      pspoints, std::vector<std::vector<float>>(legs, std::vector<float>(4)));
  for (int k{0}; k < pspoints; ++k) {
    for (int j{0}; j < legs; ++j) {
      for (int i{0}; i < 4; ++i) {
        momenta_f32[k][j][i] = static_cast<float>(momenta_f64[k][j][i]);
      }
    }
  }

  std::array<double, 2> python_outputs{{2.2266408e-07, 1.430258598666967e-06}};

  nn::FKSEnsemble<float> ensemble_f32(legs, training_reruns, model_dir, delta, cut_dir);

  nn::FKSEnsemble<double> ensemble_f64(legs, training_reruns, model_dir, delta,
                                       cut_dir);

  for (int i{0}; i < pspoints; ++i) {
    std::cout << "==================== Test point " << i + 1
              << " ====================" << '\n';

    ensemble_f32.compute_with_error(momenta_f32[i]);

    double average_output{ensemble_f64.compute(momenta_f64[i])};

    ensemble_f64.compute_with_error(momenta_f64[i]);

    const int cw{20};

    std::cout << std::scientific << std::setprecision(16) << std::setw(cw)
              << "Python = " << python_outputs[i] << '\n'
              << std::setw(cw) << "C++ f64 = " << average_output << '\n'
              << std::setw(cw) << "C++ f32 w/ err = " << ensemble_f32.mean << " ± "
              << std::setprecision(1) << ensemble_f32.std_err << " (" << std::fixed
              << 100 * ensemble_f32.std_err / ensemble_f32.mean << "%)" << '\n'
              << std::setw(cw) << "C++ f64 w/ err = " << std::scientific
              << std::setprecision(16) << ensemble_f64.mean << " ± "
              << std::setprecision(1) << ensemble_f64.std_err << " (" << std::fixed
              << 100 * ensemble_f64.std_err / ensemble_f64.mean << "%)" << '\n'
              << std::setw(cw) << "d(C++ f32, Python) = " << std::scientific
              << d(ensemble_f32.mean, python_outputs[i]) << '\n'
              << std::setw(cw)
              << "d(C++ f64, Python) = " << d(ensemble_f64.mean, python_outputs[i])
              << '\n';
  }
}
