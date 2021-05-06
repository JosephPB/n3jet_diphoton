#include <array>
#include <string>
#include <vector>

void row(const std::array<int, 6> &cw, int p, const std::string &name, double val,
         double err, long dur, double tr);

double mean(const std::vector<long> &data);

double std_err(const std::vector<long> &data, const double mean);
