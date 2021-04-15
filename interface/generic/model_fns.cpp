#include "model_fns.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

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

// KerasModel destructor
nn::KerasModel::~KerasModel() {
  for (std::size_t i{0}; i < layers.size(); ++i) {
    delete layers[i]; // deallocate memory
  }
}

// load weights for all layers
void nn::KerasModel::load_weights(std::string &input_fname) {
#ifdef DEBUG
  std::cout << "###################################" << '\n';
  std::cout << "Reading weights from file " << input_fname << '\n';
#endif
  std::ifstream fin(input_fname.c_str(), std::ifstream::in);

  if (!fin.good()) {
    std::cerr << "Error: no nnet file `" << input_fname << "`!\n";
    std::exit(EXIT_FAILURE);
  }

  std::string tmp_str{""};
  std::string layer_type{""};
  int layer_id{0};
  if (fin.is_open()) {
    // get layers count in layers_count var
    fin >> tmp_str >> layers_count;
#ifdef DEBUG
    std::cout << "Getting layers and count: " << tmp_str << layers_count << '\n';
#endif

    // Now iterate over  each layer
#ifdef DEBUG
    std::cout << "Iterating over layers..." << '\n';
#endif
    for (int layer_index{0}; layer_index < layers_count; ++layer_index) {
      fin >> tmp_str >> layer_id >> layer_type;
#ifdef DEBUG
      std::cout << tmp_str << layer_id << layer_type << '\n';
#endif
      // pointer to layer
      Layer *layer = 0L;
      if (layer_type == "Dense") {
        layer = new LayerDense();
      } else if (layer_type == "Activation") {
        layer = new LayerActivation();
      }
      // if none of above case is true, means layer not-defined
      if (layer == 0L) {
#ifdef DEBUG
        std::cout << "Layer is empty, maybe layer " << layer_type
                  << " is not defined? Cannot define network." << '\n';
#endif
        return;
      }
      layer->load_weights(fin);
      layers.push_back(layer);
#ifdef DEBUG
      std::cout << "Layer pushed back!" << '\n';
#endif
    }
  }
#ifdef DEBUG
  std::cout << "Closing file " << input_fname << '\n';
#endif
  fin.close();
}

std::vector<double> nn::KerasModel::compute_output(std::vector<double> test_input) {
#ifdef DEBUG
  std::cout << "###################################" << '\n';
  std::cout << "KerasModel compute output" << '\n';
  std::cout << "for test input " << test_input[0] << ", " << test_input[1] << '\n';
  std::cout << "Layer count: " << layers_count << '\n';
#endif
  std::vector<double> response;
  for (int i{0}; i < layers_count; ++i) {
#ifdef DEBUG
    std::cout << "Processing layer to compute output " << layers[i]->layer_name << '\n';
#endif
    response = layers[i]->compute_output(test_input);
    test_input = response;
#ifdef DEBUG
    std::cout << "Response size " << response.size() << '\n';
#endif
  }
  return response;
}

// load weights and bias from input file for Dense layer
void nn::LayerDense::load_weights(std::ifstream &fin) {
#ifdef DEBUG
  std::cout << "Loading weights for Dense layer" << '\n';
#endif
  fin >> input_node_count >> output_weights;
#ifdef DEBUG
  std::cout << "Input node count " << input_node_count << " with output weights "
            << output_weights << '\n';
#endif
  double tmp_double;
  // read weights for all the input nodes
#ifdef DEBUG
  std::cout << "Now read weights of all input modes..." << '\n';
#endif
  char tmp_char{' '};
  for (int i{0}; i < input_node_count; ++i) {
    fin >> tmp_char; // for '['
#ifdef DEBUG
    std::cout << "Input node " << i << '\n';
#endif
    std::vector<double> tmp_weights;
    for (int j{0}; j < output_weights; ++j) {
      fin >> tmp_double;
#ifdef DEBUG
      std::cout << tmp_double << '\n';
#endif
      tmp_weights.push_back(tmp_double);
    }
    fin >> tmp_char; // for ']'
    layer_weights.push_back(tmp_weights);
  }
  // read and save bias values
#ifdef DEBUG
  std::cout << "Saving biases..." << '\n';
#endif
  fin >> tmp_char; // for '['
  for (int output_node_index{0}; output_node_index < output_weights;
       output_node_index++) {
    fin >> tmp_double;
#ifdef DEBUG
    std::cout << tmp_double << '\n';
#endif
    bias.push_back(tmp_double);
  }
  fin >> tmp_char; // for ']'
}

std::vector<double> nn::LayerDense::compute_output(std::vector<double> test_input) {
#ifdef DEBUG
  std::cout << "Inside dense layer compute output" << '\n';
  std::cout << "weights: input size " << layer_weights.size() << '\n';
  std::cout << "weights: neurons size " << layer_weights[0].size() << '\n';
  std::cout << "bias size " << bias.size() << '\n';
#endif
  std::vector<double> out(output_weights);
  for (int i{0}; i < output_weights; ++i) {
    double weighted_term{0};
    for (int j{0}; j < input_node_count; ++j) {
      weighted_term += (test_input[j] * layer_weights[j][i]);
    }
    out[i] = weighted_term + bias[i];
#ifdef DEBUG
    std::cout << "...out[i]: " << out[i] << '\n';
#endif
  }
  return out;
}

// std::vector<double>
// nn::LayerActivation::compute_output(std::vector<double> test_input) {
//   if (activation_type == "linear") {
//     return test_input;
//   } else if (activation_type == "relu") {
//     for (std::size_t i{0}; i < test_input.size(); ++i) {
//       if (test_input[i] < 0) {
//         test_input[i] = 0;
//       }
//     }
//   // } else if (activation_type == "softmax") {
//   //   double sum = 0.0;
//   //   for (std::size_t k{0}; k < test_input.size(); ++k) {
//   //     test_input[k] = std::exp(test_input[k]);
//   //     sum += test_input[k];
//   //   }
//   //   for (std::size_t k{0}; k < test_input.size(); ++k) {
//   //     test_input[k] /= sum;
//   //   }
//   // } else if (activation_type == "sigmoid") {
//   //   double denominator = 0.0;
//   //   for (std::size_t k{0}; k < test_input.size(); ++k) {
//   //     denominator = 1 + std::exp(-(test_input[k]));
//   //     test_input[k] = 1 / denominator;
//   //   }
//   // } else if (activation_type == "softplus") {
//   //   for (std::size_t k{0}; k < test_input.size(); ++k) {
//   //     // log1p = natural logarithm (to base e) of 1 plus the given number
//   (ln(1+x))
//   //     test_input[k] = std::log1p(std::exp(test_input[k]));
//   //   }
//   // } else if (activation_type == "softsign") {
//   //   for (std::size_t k{0}; k < test_input.size(); ++k) {
//   //     test_input[k] = test_input[k] / (1 + abs(test_input[k]));
//   //   }
//   } else if (activation_type == "tanh") {
//     for (std::size_t k{0}; k < test_input.size(); ++k) {
//       test_input[k] = std::tanh(test_input[k]);
//     }
//   } else {
//     missing_activation_impl(activation_type);
//   }
//   return test_input;
// }

std::vector<double>
nn::LayerActivation::compute_output(std::vector<double> test_input) {
  switch (activation_type) {
  case Tanh:
    std::transform(test_input.cbegin(), test_input.cend(), test_input.begin(),
                   [](double a) -> double { return std::tanh(a); });
    break;
  case Linear:
    break;
  }
  return test_input;
}

void nn::LayerActivation::load_weights(std::ifstream &fin) {
#ifdef DEBUG
  std::cout << "Loading weights for Activation layer" << '\n';
#endif
  std::string tmp_type;
  fin >> tmp_type;

  if (tmp_type == "tanh") {
    activation_type = Tanh;
  } else if (tmp_type == "linear") {
    activation_type = Linear;
  } else {
    std::cout << "Activation " << activation_type << " not defined!" << '\n'
              << "Please add its implementation before use." << '\n';
    std::exit(EXIT_FAILURE);
  }
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
