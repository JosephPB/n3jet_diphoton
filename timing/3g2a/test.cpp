#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <ios>
#include <iostream>
#include <string>
#include <vector>

#include "analytic/0q3g2A-analytic.h"
#include "chsums/NJetAccuracy.h"
#include "ngluon2/Model.h"
#include "ngluon2/Mom.h"
#include "ngluon2/refine.h"
#include "tools/PhaseSpace.h"

#include "model_fns.hpp"

void row(const std::array<int, 6> &cw, int p, const std::string &name, double val,
         double err, long dur, double tr) {
  std::cout << ' ' << std::setw(cw[0]) << (p > 0 ? std::to_string(p) : "")
            << std::setw(cw[1]) << name << std::scientific << std::setprecision(16)
            << std::setw(cw[2]) << val << std::setprecision(1) << std::setw(cw[3])
            << err << std::fixed << std::setw(cw[4]) << dur << std::setw(cw[5]) << tr
            << '\n';
}

void run(const int start, const int end) {
  constexpr int cols{6};
  const std::array<int, cols> cw{{4, 6, 24, 12, 11, 10}};
  const std::array<std::string, cols> titles{
      {"pt", "impl", "value", "rel error", "time (us)", "t ratio"}};

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
  const int Nf{0};
  const int legs{5};

  const Flavour<double> Ax{
      StandardModel::Ax(StandardModel::IL(), StandardModel::IL().C())};

  NJetAccuracy<double> *const amp_num{
      NJetAccuracy<double>::template create<Amp0q3gAA<double>>(Ax)};
  amp_num->setMuR2(mur * mur);
  amp_num->setNf(Nf);
  amp_num->setNc(Nc);

  NJetAccuracy<double> *const amp_ana{
      NJetAccuracy<double>::template create<Amp0q3g2A_a<double>>(Ax)};
  amp_ana->setMuR2(mur * mur);
  amp_ana->setNf(Nf);
  amp_ana->setNc(Nc);

  nn::FKSNetworks ensemble(
      legs, 20,
      "../../models/3g2a/RAMBO/"
      "events_100k_fks_all_legs_all_pairs_new_sherpa_cuts_pdf_njet_test/",
      0.02, "cut_0.02/");

  // rseed = p
  for (int p{start}; p < end; ++p) {
    PhaseSpace<double> ps(legs, p, sqrtS);
    std::vector<MOM<double>> moms{ps.getPSpoint()};
    refineM(moms, moms, scales2);

    std::vector<std::vector<double>> moms_alt(legs, std::vector<double>(d));
    for (int i{0}; i < legs; ++i) {
      moms_alt[i] =
          std::vector<double>({moms[i].x0, moms[i].x1, moms[i].x2, moms[i].x3});
    }

    amp_num->setMomenta(moms);
    amp_ana->setMomenta(moms);

    t0 = std::chrono::high_resolution_clock::now();
    amp_num->virtsq();
    t1 = std::chrono::high_resolution_clock::now();
    const long dur_num{
        std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count()};
    const double val_num{amp_num->virtsq_value().get0().real()};
    const double err_num{std::abs(amp_num->virtsq_error().get0().real()) / val_num};

    t0 = std::chrono::high_resolution_clock::now();
    amp_ana->virtsq();
    t1 = std::chrono::high_resolution_clock::now();
    const long dur_ana{
        std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count()};
    const double val_ana{amp_ana->virtsq_value().get0().real()};
    const double err_ana{std::abs(amp_ana->virtsq_error().get0().real() / val_ana)};

    t0 = std::chrono::high_resolution_clock::now();
    ensemble.compute(moms_alt);
    t1 = std::chrono::high_resolution_clock::now();
    const long dur_nn{
        std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count()};
    const double val_nn{amp_ana->virtsq_value().get0().real()};
    const double err_nn{0.};

    const double tr_num{static_cast<double>(dur_num) / dur_ana};
    const double tr_ana{1};
    const double tr_nn{static_cast<double>(dur_nn) / dur_ana};

    row(cw, p, "num", val_num, err_num, dur_num, tr_num);
    row(cw, 0, "ana", val_ana, err_ana, dur_ana, tr_ana);
    row(cw, 0, "nn", val_nn, err_nn, dur_nn, tr_nn);
  }
}

int main(int argc, char *argv[]) {
  std::cout << '\n'
            << "Comparison of evaluation speeds for 3g2a implementations" << '\n'
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

  const int num{end - start};
  std::cout << "Running " << num << " point" << (num == 1 ? "" : "s") << '\n' << '\n';

  run(start, end);

  std::cout << '\n';
}
