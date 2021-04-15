#include "model_fns.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

// utility functions
// ~~~~~~~~~~~~~~~~~

std::vector<double> nn::read_metadata_from_file(const std::string &fname) {
  std::ifstream fin(fname.c_str());
  int n_x_mean{4};
  int n_x_std{4};
  int n_y_mean{1};
  int n_y_std{1};

  std::vector<double> metadata(10);

  for (int i{0}; i < n_x_mean; ++i) {
    fin >> metadata[i];
  }
  for (int i{0}; i < n_x_std; ++i) {
    fin >> metadata[n_x_mean + i];
  }
  for (int i{0}; i < n_y_mean; ++i) {
    fin >> metadata[n_x_mean + n_x_std + i];
  }
  for (int i{0}; i < n_y_std; ++i) {
    fin >> metadata[n_x_mean + n_x_std + n_y_mean + i];
  }

  return metadata;
}

double nn::standardise(double value, double mean, double stnd) {
  return (value - mean) / stnd;
}

double nn::destandardise(double value, double mean, double stnd) {
  return value * stnd + mean;
}

// Layers
// ~~~~~~

nn::LayerDense::LayerDense(std::ifstream &fin) {
  fin >> input_node_count >> output_weights;
  layer_weights = std::vector<std::vector<double>>(
      output_weights, std::vector<double>(input_node_count));

  double tmp_double;
  char tmp_char;

  for (int i{0}; i < input_node_count; ++i) {
    fin >> tmp_char; // for '['
    for (int j{0}; j < output_weights; ++j) {
      fin >> tmp_double;
      layer_weights[j][i] = tmp_double;
    }
    fin >> tmp_char; // for ']'
  }

  fin >> tmp_char; // for '['
  for (int i{0}; i < output_weights; i++) {
    fin >> tmp_double;
    bias.push_back(tmp_double);
  }
  fin >> tmp_char; // for ']'
}

std::vector<double> nn::LayerDense::compute_output(std::vector<double> test_input) {
  std::vector<double> out(output_weights);
  for (int i{0}; i < output_weights; ++i) {
    out[i] = std::inner_product(test_input.begin(), test_input.end(),
                                layer_weights[i].begin(), bias[i]);
  }
  return out;
}

nn::LayerActivation::LayerActivation(std::ifstream &fin) {
  std::string tmp_type;
  fin >> tmp_type;

  if (tmp_type == "tanh") {
    activation_type = Tanh;
  } else if (tmp_type == "linear") {
    activation_type = Linear;
  } else {
    std::cerr << "Error: Activation type " << activation_type << " not defined!" << '\n'
              << "Please add its implementation before use." << '\n';
    std::exit(EXIT_FAILURE);
  }
}

std::vector<double>
nn::LayerActivation::compute_output(std::vector<double> test_input) {
  switch (activation_type) {
  case Tanh:
    std::transform(test_input.cbegin(), test_input.cend(), test_input.begin(),
                   [](double a) -> double { return std::tanh(a); });
    break;
  case ReLU:
    for (double &d : test_input) {
      if (d < 0) {
        d = 0;
      }
    }
    break;
  case Linear:
    break;
  }
  return test_input;
}

// } else if (activation_type == "softmax") {
//   double sum = 0.0;
//   for (std::size_t k{0}; k < test_input.size(); ++k) {
//     test_input[k] = std::exp(test_input[k]);
//     sum += test_input[k];
//   }
//   for (std::size_t k{0}; k < test_input.size(); ++k) {
//     test_input[k] /= sum;
//   }
// } else if (activation_type == "sigmoid") {
//   double denominator = 0.0;
//   for (std::size_t k{0}; k < test_input.size(); ++k) {
//     denominator = 1 + std::exp(-(test_input[k]));
//     test_input[k] = 1 / denominator;
//   }
// } else if (activation_type == "softplus") {
//   for (std::size_t k{0}; k < test_input.size(); ++k) {
//     // log1p = natural logarithm (to base e) of 1 plus the given number
//     (ln(1+x))
//     test_input[k] = std::log1p(std::exp(test_input[k]));
//   }
// } else if (activation_type == "softsign") {
//   for (std::size_t k{0}; k < test_input.size(); ++k) {
//     test_input[k] = test_input[k] / (1 + abs(test_input[k]));
//   }

// Network
// ~~~~~~~

nn::KerasModel::~KerasModel() {
  for (Layer *layer : layers) {
    delete layer;
  }
}

void nn::KerasModel::load_weights(std::string &input_fname) {
  std::ifstream fin(input_fname.c_str(), std::ifstream::in);

  if (!fin.good()) {
    std::cerr << "Error: no nnet file `" << input_fname << "`!\n";
    std::exit(EXIT_FAILURE);
  }

  std::string tmp_str;
  std::string tmp_layer_type;
  int tmp_layer_id;
  if (fin.is_open()) {
    // get layers count in layers_count var
    fin >> tmp_str >> layers_count;
    // Now iterate over each layer
    for (int i{0}; i < layers_count; ++i) {
      fin >> tmp_str >> tmp_layer_id >> tmp_layer_type;
      // pointer to layer
      Layer *layer = 0L;
      if (tmp_layer_type == "Dense") {
        layer = new LayerDense(fin);
      } else if (tmp_layer_type == "Activation") {
        layer = new LayerActivation(fin);
      } else {
        std::cerr << "Error: Layer type " << tmp_layer_type << " is not defined!"
                  << '\n'
                  << "Please add its implementation before use." << '\n';
        std::exit(EXIT_FAILURE);
      }
      layers.push_back(layer);
    }
  }
  fin.close();
}

std::vector<double> nn::KerasModel::compute_output(std::vector<double> test_input) {
  for (Layer *layer : layers) {
    test_input = layer->compute_output(test_input);
  }
  return test_input;
}

nn::Networks::Networks(const int legs_, const int runs_, const std::string &model_path,
                       const double delta_, const std::string &cut_dirs_)
    // n.b. there is an additional FKS pair for the cut network (for non-divergent
    // regions)
    : n_choose_2({{0, 0, 1, 3, 6, 10, 15, 21, 28, 36, 45}}), legs(legs_), runs(runs_),
      pairs(n_choose_2[legs] - 1), delta(delta_), cut_dirs(cut_dirs_),
      model_base(model_path), model_dirs(runs), pair_dirs(pairs) {
  std::generate(model_dirs.begin(), model_dirs.end(),
                [n = 0]() mutable { return std::to_string(n++) + "/"; });
  std::generate(pair_dirs.begin(), pair_dirs.end(),
                [n = 0]() mutable { return "pair_0.02_" + std::to_string(n++) + "/"; });
}

double nn::Networks::dot(const std::vector<std::vector<double>> &point, const int k,
                         const int j) const {
  return point[j][0] * point[k][0] -
         (point[j][1] * point[k][1] + point[j][2] * point[k][2] +
          point[j][3] * point[k][3]);
}

// Ensemble
// ~~~~~~~~

nn::NaiveNetworks::NaiveNetworks(const int legs, const int runs,
                                 const std::string &model_path, const double delta_,
                                 const std::string &cut_dirs_)
    : Networks(legs, runs, model_path, delta_, cut_dirs_), kerasModels(runs),
      metadatas(runs, std::vector<double>(10)), model_dir_models(runs) {
  for (int i{0}; i < runs; ++i) {
    // Naive networks
    const std::string metadata_file{model_base + model_dirs[i] +
                                    "dataset_metadata.dat"};
    const std::vector<double> metadata{nn::read_metadata_from_file(metadata_file)};
    std::copy(metadata.cbegin(), metadata.cend(), metadatas[i].begin());
    model_dir_models[i] = model_base + model_dirs[i] + "model.nnet";
    kerasModels[i].load_weights(model_dir_models[i]);
  }
}

double nn::NaiveNetworks::compute(const std::vector<std::vector<double>> &point) {
  // moms is an vector of runs results, each of which is an vector of flattened momenta
  // std::array<std::array<double, legs * d>, runs> moms;
  std::vector<std::vector<double>> moms(runs, std::vector<double>(legs * d));

  // flatten momenta
  for (int p{0}; p < legs; ++p) {
    for (int mu{0}; mu < d; ++mu) {
      // standardise input
      for (int k{0}; k < runs; ++k) {
        moms[k][p * d + mu] =
            nn::standardise(point[p][mu], metadatas[k][mu], metadatas[k][d + mu]);
      }
    }
  }

  // inference
  double results_sum{0.};
  for (int j{0}; j < runs; ++j) {
    const double result{kerasModels[j].compute_output(moms[j])[0]};
    results_sum += nn::destandardise(result, metadatas[j][8], metadatas[j][9]);
  }

  return results_sum / runs;
}

nn::FKSNetworks::FKSNetworks(const int legs, const int runs,
                             const std::string &model_path, const double delta_,
                             const std::string &cut_dirs_)
    : Networks(legs, runs, model_path, delta_, cut_dirs_),
      kerasModels(runs, std::vector<nn::KerasModel>(pairs + 1)),
      metadatas(runs,
                std::vector<std::vector<double>>(pairs + 1, std::vector<double>(10))),
      model_dir_models(runs, std::vector<std::string>(pairs + 1)) {
  for (int i{0}; i < runs; ++i) {
    // Near networks
    for (int j{0}; j < pairs; ++j) {
      const std::vector<double> metadata{nn::read_metadata_from_file(
          model_base + model_dirs[i] + pair_dirs[j] + "dataset_metadata.dat")};
      std::copy(metadata.cbegin(), metadata.cend(), metadatas[i][j].begin());
      model_dir_models[i][j] = model_base + model_dirs[i] + pair_dirs[j] + "model.nnet";
      kerasModels[i][j].load_weights(model_dir_models[i][j]);
    };
    // Cut networks
    const std::vector<double> metadata{nn::read_metadata_from_file(
        model_base + model_dirs[i] + cut_dirs + "dataset_metadata.dat")};
    std::copy(metadata.cbegin(), metadata.cend(), metadatas[i][pairs].begin());
    model_dir_models[i][pairs] = model_base + model_dirs[i] + cut_dirs + "model.nnet";
    kerasModels[i][pairs].load_weights(model_dir_models[i][pairs]);
  }
}

double nn::FKSNetworks::compute(const std::vector<std::vector<double>> &point) {
  // moms is an vector of runs results, each of which is an vector of FKS pairs results,
  // each of which is an vector of flattened momenta
  std::vector<std::vector<std::vector<double>>> moms(
      runs, std::vector<std::vector<double>>(pairs + 1, std::vector<double>(legs * d)));

  // NN compute_output accepts vectors - could edit model_fns
  // std::array<std::array<std::array<double, NN2A::legs * NN2A::d>, pairs + 1>, runs>
  // moms;

  // flatten momenta
  for (int p{0}; p < legs; ++p) {
    for (int mu{0}; mu < d; ++mu) {
      // standardise input
      for (int k{0}; k < runs; ++k) {
        for (int j{0}; j <= pairs; ++j) {
          moms[k][j][p * d + mu] = nn::standardise(point[p][mu], metadatas[k][j][mu],
                                                   metadatas[k][j][d + mu]);
        }
      }
    }
  }

  const double s_com{dot(point, 0, 1)};

  // cut/near check
  int cut_near{0};
  for (int j{0}; j < legs - 1; ++j) {
    for (int k{j + 1}; k < legs; ++k) {
      if (dot(point, j, k) / s_com < delta) {
        cut_near += 1;
      }
    }
  }

  // inference
  double results_sum{0.};
  for (int j{0}; j < runs; ++j) {
    if (cut_near > 0) {
      // the point is near an IR singularity
      // infer over all FKS pairs
      for (int k{0}; k < pairs; ++k) {
        const double result{kerasModels[j][k].compute_output(moms[j][k])[0]};
        results_sum +=
            nn::destandardise(result, metadatas[j][k][8], metadatas[j][k][9]);
      }
    } else {
      // the point is in a non-divergent region
      // use the 'cut' network which is the final entry in the pair network
      const double result{kerasModels[j][pairs].compute_output(moms[j][pairs])[0]};
      results_sum +=
          nn::destandardise(result, metadatas[j][pairs][8], metadatas[j][pairs][9]);
    }
  }

  return results_sum / runs;
}
