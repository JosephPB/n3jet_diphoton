#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <ios>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

#include "chsums/0q4gA.h"
#include "chsums/NJetAccuracy.h"
#include "ngluon2/Model.h"
#include "ngluon2/Mom.h"
#include "ngluon2/refine.h"
#include "tools/PhaseSpace.h"

#include "model_fns.hpp"
#include "timing.hpp"

void run(const int start, const int end) {
  const int num{end - start};
  std::vector<long> tme_num(num);
  std::vector<long> tme_nn_f32(num);
  std::vector<long> tme_nn_f64(num);

  constexpr int cols{6};
  const std::array<int, cols> cw{{4, 6, 24, 12, 11, 10}};
  const std::array<std::string, cols> titles{
      {"pt", "impl", "value", "rel error", "time (ns)", "t ratio"}};

  std::cout << "Running " << num << " point" << (num == 1 ? "" : "s") << '\n' << '\n';

  std::cout << std::left << ' ';
  for (int i{0}; i < cols; ++i) {
    std::cout << std::setw(cw[i]) << titles[i];
  }
  std::cout << '\n' << std::setfill('-');
  for (int d : cw) {
    std::cout << std::setw(d) << '|';
  }
  std::cout << '|' << std::setfill(' ') << '\n';

  using TP = std::chrono::time_point<std::chrono::high_resolution_clock>;

  TP t0;
  TP t1;

  const std::vector<double> scales2{{0}};
  const double sqrtS{10.};
  const double mur{sqrtS / 2.};
  const int d{4};
  const int Nc{3};
  const int Nf{5};
  const int legs{6};

  const Flavour<double> Ax{
      StandardModel::Ax(StandardModel::IL(), StandardModel::IL().C())};

  NJetAccuracy<double> *const amp_num{
      NJetAccuracy<double>::template create<Amp0q4gAA<double>>(Ax)};
  amp_num->setMuR2(mur * mur);
  amp_num->setNf(Nf);
  amp_num->setNc(Nc);

  nn::FKSEnsemble<float> ensemble_f32(legs, 20,
                                      "../../models/4g2A/RAMBO/"
                                      "100k_unit_0001_fks/",
                                      0.02, "cut_0.02/");

  nn::FKSEnsemble<double> ensemble_f64(legs, 20,
                                       "../../models/4g2A/RAMBO/"
                                       "100k_unit_0001_fks/",
                                       0.02, "cut_0.02/");

  std::ofstream o("result.csv", std::ios::app);
  o << std::scientific << std::setprecision(16);

  // rseed = p
  for (int a{0}; a < num; ++a) {
    const int p{a + start};
    PhaseSpace<double> ps(legs, p, sqrtS);
    std::vector<MOM<double>> moms{ps.getPSpoint()};
    refineM(moms, moms, scales2);

    std::vector<std::vector<float>> moms_f32(legs, std::vector<float>(d));
    for (int i{0}; i < legs; ++i) {
      moms_f32[i] = std::vector<float>({
          static_cast<float>(moms[i].x0),
          static_cast<float>(moms[i].x1),
          static_cast<float>(moms[i].x2),
          static_cast<float>(moms[i].x3),
      });
    }

    std::vector<std::vector<double>> moms_f64(legs, std::vector<double>(d));
    for (int i{0}; i < legs; ++i) {
      moms_f64[i] = std::vector<double>({
          moms[i].x0,
          moms[i].x1,
          moms[i].x2,
          moms[i].x3,
      });
    }

    amp_num->setMomenta(moms);

    t0 = std::chrono::high_resolution_clock::now();
    amp_num->virtsq();
    t1 = std::chrono::high_resolution_clock::now();
    const long dur_num{
        std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count()};
    tme_num[a] = dur_num;
    const double val_num{amp_num->virtsq_value().get0().real()};
    const double err_num{std::abs(amp_num->virtsq_error().get0().real()) / val_num};

    t0 = std::chrono::high_resolution_clock::now();
    ensemble_f32.compute_with_error(moms_f32);
    t1 = std::chrono::high_resolution_clock::now();
    const long dur_nn_f32{
        std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count()};
    tme_nn_f32[a] = dur_nn_f32;
    const double val_nn_f32{ensemble_f32.mean};
    const double err_nn_f32{ensemble_f32.std_err};

    t0 = std::chrono::high_resolution_clock::now();
    ensemble_f64.compute_with_error(moms_f64);
    t1 = std::chrono::high_resolution_clock::now();
    const long dur_nn_f64{
        std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count()};
    tme_nn_f64[a] = dur_nn_f64;
    const double val_nn_f64{ensemble_f64.mean};
    const double err_nn_f64{ensemble_f64.std_err};

    const double tr_num{static_cast<double>(dur_num) / dur_nn_f32};
    const double tr_nn_f32{static_cast<double>(dur_nn_f32) / dur_nn_f32};
    const double tr_nn_f64{static_cast<double>(dur_nn_f64) / dur_nn_f32};

    row(cw, p, "num", val_num, err_num, dur_num, tr_num);
    row(cw, 0, "nn_f32", val_nn_f32, err_nn_f32, dur_nn_f32, tr_nn_f32);
    row(cw, 0, "nn_f64", val_nn_f64, err_nn_f64, dur_nn_f64, tr_nn_f64);

    o << p << ' ' << val_num << ' ' << err_num << ' ' << dur_num << ' ' << val_nn_f32
      << ' ' << err_nn_f32 << ' ' << dur_nn_f32 << val_nn_f64 << ' ' << err_nn_f64
      << ' ' << dur_nn_f64 << '\n';
  }

  std::cout << std::setfill('-');
  for (int d : cw) {
    std::cout << std::setw(d) << '|';
  }
  std::cout << '|' << std::setfill(' ') << '\n';

  report(tme_num, tme_nn_f32, tme_nn_f64, "num", "nn_f32", "nn_f64");
}

int main(int argc, char *argv[]) {
  std::cout << '\n'
            << "Comparison of evaluation speeds for 4g2a implementations" << '\n'
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
