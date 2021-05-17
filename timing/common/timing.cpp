#include <array>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

#include "timing.hpp"

void row(const std::array<int, 6> &cw, int p, const std::string &name, double val,
         double err, long dur, double tr) {
  std::cout << ' ' << std::setw(cw[0]) << (p > 0 ? std::to_string(p) : "")
            << std::setw(cw[1]) << name << std::scientific << std::setprecision(16)
            << std::setw(cw[2]) << val << std::setprecision(1) << std::setw(cw[3])
            << err << std::setw(cw[4]) << static_cast<double>(dur) << std::fixed
            << std::setw(cw[5]) << tr << '\n';
}

double mean(const std::vector<long> &data) {
  return std::accumulate(data.cbegin(), data.cend(), 0.) / data.size();
}

double std_err(const std::vector<long> &data, const double mean) {
  double err{};
  for (long datum : data) {
    const double term{datum - mean};
    err += term * term;
  }
  return std::sqrt(err) / data.size();
}

void report(const std::vector<long> &tme_num, const std::vector<long> &tme_ana,
            const std::vector<long> &tme_nn, const std::string &title1,
            const std::string &title2, const std::string &title3) {
  const double mn_num{mean(tme_num)};
  const double abs_std_err_num{std_err(tme_num, mn_num)};
  const double rel_std_err_num{abs_std_err_num / mn_num};

  const double mn_ana{mean(tme_ana)};
  const double abs_std_err_ana{std_err(tme_ana, mn_ana)};
  const double rel_std_err_ana{abs_std_err_ana / mn_ana};

  const double mn_nn{mean(tme_nn)};
  const double abs_std_err_nn{std_err(tme_nn, mn_nn)};
  const double rel_std_err_nn{abs_std_err_nn / mn_nn};

  const int abs_prec{1};
  const int perc_prec{1};

  std::cout << '\n'
            << "Mean times (ns)" << '\n'
            << std::setw(5) << title1 << std::scientific << std::setprecision(abs_prec)
            << mn_num << " ± " << abs_std_err_num << " (" << std::fixed
            << std::setprecision(perc_prec) << 100 * rel_std_err_num << "%)" << '\n'
            << std::setw(5) << title2 << std::scientific << std::setprecision(abs_prec)
            << mn_ana << " ± " << abs_std_err_ana << " (" << std::fixed
            << std::setprecision(perc_prec) << 100 * rel_std_err_ana << "%)" << '\n'
            << std::setw(5) << title3 << std::scientific << std::setprecision(abs_prec)
            << mn_nn << " ± " << abs_std_err_nn << " (" << std::fixed
            << std::setprecision(perc_prec) << 100 * rel_std_err_nn << "%)" << '\n';

  std::cout << '\n'
            << "Mean time ratios" << '\n'
            << std::setw(5) << title1 << std::fixed << std::setprecision(abs_prec)
            << mn_num / mn_nn << '\n'
            << std::setw(5) << title2 << mn_ana / mn_nn << '\n'
            << std::setw(5) << title3 << mn_nn / mn_nn << '\n';
}

void report(const std::vector<long> &tme_num, const std::vector<long> &tme_nn) {
  const double mn_num{mean(tme_num)};
  const double abs_std_err_num{std_err(tme_num, mn_num)};
  const double rel_std_err_num{abs_std_err_num / mn_num};

  const double mn_nn{mean(tme_nn)};
  const double abs_std_err_nn{std_err(tme_nn, mn_nn)};
  const double rel_std_err_nn{abs_std_err_nn / mn_nn};

  const int abs_prec{1};
  const int perc_prec{1};

  std::cout << '\n'
            << "Mean times (ns)" << '\n'
            << std::setw(5) << "num" << std::setprecision(abs_prec) << mn_num << " ± "
            << abs_std_err_num << " (" << std::setprecision(perc_prec)
            << 100 * rel_std_err_num << "%)" << '\n'
            << std::setw(5) << "nn" << std::setprecision(abs_prec) << mn_nn << " ± "
            << abs_std_err_nn << " (" << std::setprecision(perc_prec)
            << 100 * rel_std_err_nn << "%)" << '\n';

  std::cout << '\n'
            << "Mean time ratios" << '\n'
            << std::setw(5) << "num" << std::setprecision(abs_prec) << 1. << '\n'
            << std::setw(5) << "nn" << mn_nn / mn_num << '\n';
}
