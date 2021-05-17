#include <array>
#include <string>
#include <vector>

void row(const std::array<int, 6> &cw, int p, const std::string &name, double val,
         double err, long dur, double tr);

double mean(const std::vector<long> &data);

double std_err(const std::vector<long> &data, const double mean);

void report(const std::vector<long> &tme_num, const std::vector<long> &tme_ana,
            const std::vector<long> &tme_nn, const std::string &title1 = "num",
            const std::string &title2 = "ana", const std::string &title3 = "nn");

void report(const std::vector<long> &tme_num, const std::vector<long> &tme_nn);
