#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "analytic/0q3g2A-analytic.h"
#include "chsums/NJetAccuracy.h"
#include "ngluon2/Model.h"
#include "ngluon2/Mom.h"
#include "ngluon2/refine.h"
#include "tools/PhaseSpace.h"

void run(const int start, const int end) {
  std::cout << "pt  impl  value                   relative error          time (us)"
            << '\n';

  using TP = std::chrono::time_point<std::chrono::high_resolution_clock>;

  TP t0;
  TP t1;

  const std::vector<double> scales2{{0}};
  const double sqrtS{10.};
  const double mur{sqrtS / 2.};
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

  // rseed = p
  for (int p{start}; p < end; ++p) {
    std::cout << p << "   ";

    PhaseSpace<double> ps(legs, p, sqrtS);
    std::vector<MOM<double>> momenta{ps.getPSpoint()};
    refineM(momenta, momenta, scales2);

    amp_num->setMomenta(momenta);
    amp_ana->setMomenta(momenta);

    t0 = std::chrono::high_resolution_clock::now();
    amp_num->virtsq();
    t1 = std::chrono::high_resolution_clock::now();
    const long dur_num{
        std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count()};
    const double val_num{amp_num->virtsq_value().get0().real()};
    const double err_num{std::abs(amp_num->virtsq_error().get0().real()) / val_num};
    std::cout << "num   " << val_num << "  " << err_num << "  " << dur_num << '\n';

    t0 = std::chrono::high_resolution_clock::now();
    amp_ana->virtsq();
    t1 = std::chrono::high_resolution_clock::now();
    const long dur_ana{
        std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count()};
    const double val_ana{amp_ana->virtsq_value().get0().real()};
    const double err_ana{std::abs(amp_ana->virtsq_error().get0().real() / val_ana)};
    std::cout << "    ana   " << val_ana << "  " << err_ana << "  " << dur_ana << '\n';
  }
}

int main(int argc, char *argv[]) {
  std::cout.setf(std::ios_base::scientific, std::ios_base::floatfield);
  std::cout.precision(16);

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
