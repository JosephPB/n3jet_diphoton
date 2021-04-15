#include <chrono>
#include <iostream>
#include <vector>

#include "model_fns.hpp"

int main() {
  std::cout.setf(std::ios_base::scientific, std::ios_base::floatfield);
  std::cout.precision(16);

  const int num{100};
  const int pspoints{2};

  std::cout << '\n'
            << "n3jet: benchmark pretrained neural network C++ inference timing" << '\n'
            << "       showing mean of " << num << " runs for " << pspoints << " points"
            << '\n'
            << '\n';

  using TP = const std::chrono::high_resolution_clock::time_point;

  const int legs{5};
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

  nn::FKSEnsemble networks(
      legs, training_reruns,
      "../../models/3g2a/RAMBO/"
      "events_100k_fks_all_legs_all_pairs_new_sherpa_cuts_pdf_njet_test/",
      delta, "cut_0.02/");

  for (int i{0}; i < pspoints; ++i) {
    TP nnt1{std::chrono::high_resolution_clock::now()};
    for (int j{0}; j < num; ++j) {
      networks.compute(momenta[i]);
    };
    TP nnt2{std::chrono::high_resolution_clock::now()};
    const long int nndur{
        std::chrono::duration_cast<std::chrono::microseconds>(nnt2 - nnt1).count()};

    std::cout << i << " " << nndur / num << "us" << '\n';
  }
}
