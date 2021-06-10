#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <numeric>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include <Eigen/Dense>

// ====================================================================================
// declaration
// ====================================================================================

namespace nn {

// Layers
// ~~~~~~

template <typename T> struct Layer {
  virtual ~Layer(){};

  virtual void compute_output(Eigen::VectorX<T> &test_input) = 0;
};

template <typename T> struct LayerDense : public Layer<T> {
  LayerDense(std::ifstream &fin);

  int input_node_count;
  int output_weights;
  std::vector<Eigen::VectorX<T>> layer_weights;
  Eigen::VectorX<T> bias;

  virtual void compute_output(Eigen::VectorX<T> &test_input) override;
};

template <typename T> struct LayerActivationTanh : public Layer<T> {
  virtual void compute_output(Eigen::VectorX<T> &test_input) override;
};

template <typename T> struct LayerActivationLinear : public Layer<T> {
  virtual void compute_output(Eigen::VectorX<T> & /* test_input */) override;
};

template <typename T> struct LayerActivationReLU : public Layer<T> {
  virtual void compute_output(Eigen::VectorX<T> &test_input) override;
};

// Network
// ~~~~~~~

template <typename T> class Network {
public:
  ~Network();
  void load_weights(std::string &input_fname);
  T compute_output(Eigen::VectorX<T> test_input);

private:
  int layers_count;
  std::vector<Layer<T> *> layers;
};

// Ensemble
// ~~~~~~~~

template <typename T> class Ensemble {
public:
  Ensemble(int legs_, int runs_, const std::string &model_path,
           const std::string &cut_dirs_);

  virtual T compute_single(const std::vector<std::vector<T>> &point,
                           int ensemble_index) = 0;
  virtual T compute(const std::vector<std::vector<T>> &point) = 0;
  virtual void compute_with_error(const std::vector<std::vector<T>> &point) = 0;

  T mean() const;
  T std_dev() const;
  T std_err() const;

protected:
  T _mean;
  T _std_dev;
  T _std_err;

  static constexpr int d{4};
  const int legs;
  const int runs;
  const std::string cut_dirs;
  const std::string model_base;

  std::vector<std::string> model_dirs;

  T dot(const std::vector<std::vector<T>> &point, int k, int j) const;
  T standardise(T value, T mean_, T stnd) const;
  T destandardise(T value, T mean_, T stnd) const;
  std::vector<T> read_metadata_from_file(const std::string &fname);
};

template <typename T> class NaiveEnsemble : public Ensemble<T> {
  using Base = Ensemble<T>;

public:
  NaiveEnsemble(const int legs, const int runs, const std::string &model_path,
                const std::string &cut_dirs_);

  virtual T compute_single(const std::vector<std::vector<T>> &point,
                           int index) override;
  virtual T compute(const std::vector<std::vector<T>> &point) override;
  virtual void compute_with_error(const std::vector<std::vector<T>> &point) override;

  using Base::mean;
  using Base::std_dev;
  using Base::std_err;

private:
  std::vector<nn::Network<T>> kerasModels;
  std::vector<std::vector<T>> metadatas;
  std::vector<std::string> model_dir_models;

  std::vector<Eigen::VectorX<T>> std_mom(const std::vector<std::vector<T>> &point);

  T infer(const std::vector<Eigen::VectorX<T>> &point, int ensemble_index);

  using Base::_mean;
  using Base::_std_dev;
  using Base::_std_err;
  using Base::d;
  using Base::destandardise;
  using Base::dot;
  using Base::legs;
  using Base::model_base;
  using Base::model_dirs;
  using Base::read_metadata_from_file;
  using Base::runs;
  using Base::standardise;
};

template <typename T> class FKSEnsemble : public Ensemble<T> {
  using Base = Ensemble<T>;

public:
  FKSEnsemble(const int legs, const int runs, const std::string &model_path, T delta_,
              const std::string &cut_dirs_);

  virtual T compute_single(const std::vector<std::vector<T>> &point,
                           int ensemble_index) override;
  virtual T compute(const std::vector<std::vector<T>> &point) override;
  virtual void compute_with_error(const std::vector<std::vector<T>> &point) override;

  using Base::mean;
  using Base::std_dev;
  using Base::std_err;

private:
  const std::array<int, 11> n_choose_2;
  const int pairs;
  const T delta;
  std::vector<std::string> pair_dirs;
  std::vector<std::vector<std::string>> model_dir_models;
  std::vector<std::vector<nn::Network<T>>> kerasModels;
  std::vector<std::vector<std::vector<T>>> metadatas;

  std::vector<std::vector<Eigen::VectorX<T>>>
  std_mom(const std::vector<std::vector<T>> &point) const;

  bool check_div(const std::vector<std::vector<T>> &point) const;

  T infer(const std::vector<std::vector<Eigen::VectorX<T>>> &point, int ensemble_index,
          int pair_index);

  using Base::_mean;
  using Base::_std_dev;
  using Base::_std_err;
  using Base::cut_dirs;
  using Base::d;
  using Base::destandardise;
  using Base::dot;
  using Base::legs;
  using Base::model_base;
  using Base::model_dirs;
  using Base::read_metadata_from_file;
  using Base::runs;
  using Base::standardise;
};

} // namespace nn

// ====================================================================================
// implementation
// ====================================================================================

// Layers
// ~~~~~~

template <typename T> nn::LayerDense<T>::LayerDense(std::ifstream &fin) {
  fin >> input_node_count >> output_weights;
  layer_weights = std::vector<Eigen::VectorX<T>>(output_weights,
                                                 Eigen::VectorX<T>(input_node_count));
  bias = Eigen::VectorX<T>(output_weights);

  T tmp_double;
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
    bias[i] = tmp_double;
  }
  fin >> tmp_char; // for ']'
}

template <typename T>
void nn::LayerDense<T>::compute_output(Eigen::VectorX<T> &test_input) {
  Eigen::VectorX<T> out(output_weights);
  for (int i{0}; i < output_weights; ++i) {
    out[i] = test_input.dot(layer_weights[i]) + bias[i];
  }
  test_input = out;
}

template <typename T>
void nn::LayerActivationTanh<T>::compute_output(Eigen::VectorX<T> &test_input) {
  test_input = test_input.array().tanh();
}

template <typename T>
void nn::LayerActivationLinear<T>::compute_output(
    Eigen::VectorX<T> & /* test_input */) {}

template <typename T>
void nn::LayerActivationReLU<T>::compute_output(Eigen::VectorX<T> &test_input) {
  test_input = test_input.unaryExpr([](T a) -> T { return std::max(T(), a); });
}

// } else if (activation_type == "softmax") {
//   T sum = 0.0;
//   for (std::size_t k{0}; k < test_input.size(); ++k) {
//     test_input[k] = std::exp(test_input[k]);
//     sum += test_input[k];
//   }
//   for (std::size_t k{0}; k < test_input.size(); ++k) {
//     test_input[k] /= sum;
//   }
// } else if (activation_type == "sigmoid") {
//   T denominator = 0.0;
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

template <typename T> nn::Network<T>::~Network() {
  for (Layer<T> *layer : layers) {
    delete layer;
  }
}

template <typename T> void nn::Network<T>::load_weights(std::string &input_fname) {
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
      Layer<T> *layer = 0L;
      if (tmp_layer_type == "Dense") {
        layer = new LayerDense<T>(fin);
      } else if (tmp_layer_type == "Activation") {
        std::string activation_type;
        fin >> activation_type;

        if (activation_type == "tanh") {
          layer = new LayerActivationTanh<T>;
        } else if (activation_type == "linear") {
          layer = new LayerActivationLinear<T>;
        } else if (activation_type == "relu") {
          layer = new LayerActivationReLU<T>;
        } else {
          std::cerr << "Error: Activation type " << activation_type << " not defined!"
                    << '\n'
                    << "Please add its implementation before use." << '\n';
          std::exit(EXIT_FAILURE);
        }
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

template <typename T> T nn::Network<T>::compute_output(Eigen::VectorX<T> test_input) {
  for (Layer<T> *layer : layers) {
    layer->compute_output(test_input);
  }
  return test_input[0];
}

// Ensemble Base
// ~~~~~~~~~~~~~

template <typename T>
nn::Ensemble<T>::Ensemble(const int legs_, const int runs_,
                          const std::string &model_path, const std::string &cut_dirs_)
    // n.b. there is an additional FKS pair for the cut network (for non-divergent
    // regions)
    : _mean(), _std_dev(), _std_err(), legs(legs_), runs(runs_), cut_dirs(cut_dirs_),
      model_base(model_path), model_dirs(runs) {
  std::generate(model_dirs.begin(), model_dirs.end(),
                [n = 0]() mutable { return std::to_string(n++) + "/"; });
}

template <typename T>
T nn::Ensemble<T>::dot(const std::vector<std::vector<T>> &point, const int k,
                       const int j) const {
  return point[j][0] * point[k][0] -
         (point[j][1] * point[k][1] + point[j][2] * point[k][2] +
          point[j][3] * point[k][3]);
}

template <typename T> T nn::Ensemble<T>::mean() const { return _mean; }

template <typename T> T nn::Ensemble<T>::std_dev() const { return _std_dev; }

template <typename T> T nn::Ensemble<T>::std_err() const { return _std_err; }

template <typename T> T nn::Ensemble<T>::standardise(T value, T mean_, T stnd) const {
  return (value - mean_) / stnd;
}

template <typename T> T nn::Ensemble<T>::destandardise(T value, T mean_, T stnd) const {
  return value * stnd + mean_;
}

template <typename T>
std::vector<T> nn::Ensemble<T>::read_metadata_from_file(const std::string &fname) {
  std::ifstream fin(fname.c_str());
  int n_x_mean{4};
  int n_x_std{4};
  int n_y_mean{1};
  int n_y_std{1};

  std::vector<T> metadata(10);

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

// Naive Ensemble
// ~~~~~~~~~~~~~~

template <typename T>
nn::NaiveEnsemble<T>::NaiveEnsemble(const int legs, const int runs,
                                    const std::string &model_path,
                                    const std::string &cut_dirs_)
    : Ensemble<T>(legs, runs, model_path, cut_dirs_), kerasModels(runs),
      metadatas(runs, std::vector<T>(10)), model_dir_models(runs) {
  for (int i{0}; i < runs; ++i) {
    // Naive networks
    const std::string metadata_file{model_base + model_dirs[i] +
                                    "dataset_metadata.dat"};
    const std::vector<T> metadata{read_metadata_from_file(metadata_file)};
    std::copy(metadata.cbegin(), metadata.cend(), metadatas[i].begin());
    model_dir_models[i] = model_base + model_dirs[i] + "model.nnet";
    kerasModels[i].load_weights(model_dir_models[i]);
  }
}

template <typename T>
std::vector<Eigen::VectorX<T>>
nn::NaiveEnsemble<T>::std_mom(const std::vector<std::vector<T>> &point) {
  // moms is an vector of runs results, each of which is an vector of flattened momenta
  // std::array<std::array<T, legs * d>, runs> moms;
  std::vector<Eigen::VectorX<T>> moms(runs, Eigen::VectorX<T>(legs * d));

  // flatten momenta
  for (int p{0}; p < legs; ++p) {
    for (int mu{0}; mu < d; ++mu) {
      // standardise input
      for (int k{0}; k < runs; ++k) {
        moms[k][p * d + mu] =
            standardise(point[p][mu], metadatas[k][mu], metadatas[k][d + mu]);
      }
    }
  }

  return moms;
}

template <typename T>
T nn::NaiveEnsemble<T>::infer(const std::vector<Eigen::VectorX<T>> &point,
                              const int i) {
  return destandardise(kerasModels[i].compute_output(point[i]), metadatas[i][8],
                       metadatas[i][9]);
}

template <typename T>
T nn::NaiveEnsemble<T>::compute_single(const std::vector<std::vector<T>> &point,
                                       const int index) {
  std::vector<Eigen::VectorX<T>> moms{std_mom(point)};

  return infer(moms, index);
}

template <typename T>
T nn::NaiveEnsemble<T>::compute(const std::vector<std::vector<T>> &point) {
  std::vector<Eigen::VectorX<T>> moms{std_mom(point)};

  // inference
  _mean = T();
  for (int j{0}; j < runs; ++j) {
    _mean += infer(moms, j);
  }
  _mean /= runs;

  return _mean;
}

template <typename T>
void nn::NaiveEnsemble<T>::compute_with_error(
    const std::vector<std::vector<T>> &point) {
  std::vector<Eigen::VectorX<T>> moms{std_mom(point)};

  // inference
  std::vector<T> results(runs);
  for (int j{0}; j < runs; ++j) {
    results[j] = infer(moms, j);
  }

  _mean = std::accumulate(results.cbegin(), results.cend(), T()) / runs;

  _std_dev = T();
  for (const T result : results) {
    const T term{result - _mean};
    _std_dev += term * term;
  }
  _std_dev = std::sqrt(_std_dev / runs);

  _std_err = _std_dev / std::sqrt(runs);
}

// FKS Ensemble
// ~~~~~~~~~~~~

template <typename T>
nn::FKSEnsemble<T>::FKSEnsemble(const int legs, const int runs,
                                const std::string &model_path, const T delta_,
                                const std::string &cut_dirs_)
    : Ensemble<T>(legs, runs, model_path, cut_dirs_),
      n_choose_2({{0, 0, 1, 3, 6, 10, 15, 21, 28, 36, 45}}),
      pairs(n_choose_2[legs] - 1), delta(delta_), pair_dirs(pairs),
      model_dir_models(runs, std::vector<std::string>(pairs + 1)),
      kerasModels(runs, std::vector<nn::Network<T>>(pairs + 1)),
      metadatas(runs, std::vector<std::vector<T>>(pairs + 1, std::vector<T>(10))) {
  std::generate(pair_dirs.begin(), pair_dirs.end(),
                [n = 0]() mutable { return "pair_0.02_" + std::to_string(n++) + "/"; });

  for (int i{0}; i < runs; ++i) {
    // Near networks
    for (int j{0}; j < pairs; ++j) {
      const std::vector<T> metadata{read_metadata_from_file(
          model_base + model_dirs[i] + pair_dirs[j] + "dataset_metadata.dat")};
      std::copy(metadata.cbegin(), metadata.cend(), metadatas[i][j].begin());
      model_dir_models[i][j] = model_base + model_dirs[i] + pair_dirs[j] + "model.nnet";
      kerasModels[i][j].load_weights(model_dir_models[i][j]);
    };
    // Cut networks
    const std::vector<T> metadata{read_metadata_from_file(
        model_base + model_dirs[i] + cut_dirs + "dataset_metadata.dat")};
    std::copy(metadata.cbegin(), metadata.cend(), metadatas[i][pairs].begin());
    model_dir_models[i][pairs] = model_base + model_dirs[i] + cut_dirs + "model.nnet";
    kerasModels[i][pairs].load_weights(model_dir_models[i][pairs]);
  }
}

template <typename T>
std::vector<std::vector<Eigen::VectorX<T>>>
nn::FKSEnsemble<T>::std_mom(const std::vector<std::vector<T>> &point) const {
  // moms is an vector of runs results, each of which is an vector of FKS pairs results,
  // each of which is an vector of flattened momenta
  std::vector<std::vector<Eigen::VectorX<T>>> moms(
      runs, std::vector<Eigen::VectorX<T>>(pairs + 1, Eigen::VectorX<T>(legs * d)));

  // flatten momenta
  for (int p{0}; p < legs; ++p) {
    for (int mu{0}; mu < d; ++mu) {
      // standardise input
      for (int k{0}; k < runs; ++k) {
        for (int j{0}; j <= pairs; ++j) {
          moms[k][j][p * d + mu] =
              standardise(point[p][mu], metadatas[k][j][mu], metadatas[k][j][d + mu]);
        }
      }
    }
  }

  return moms;
}

template <typename T>
bool nn::FKSEnsemble<T>::check_div(const std::vector<std::vector<T>> &point) const {
  const T s_com{dot(point, 0, 1)};

  // cut/near check
  for (int j{0}; j < legs - 1; ++j) {
    for (int k{j + 1}; k < legs; ++k) {
      if (dot(point, j, k) / s_com < delta) {
        return true;
      }
    }
  }

  return false;
}

template <typename T>
T nn::FKSEnsemble<T>::infer(const std::vector<std::vector<Eigen::VectorX<T>>> &point,
                            const int i, const int j) {
  return destandardise(kerasModels[i][j].compute_output(point[i][j]),
                       metadatas[i][j][8], metadatas[i][j][9]);
}

template <typename T>
T nn::FKSEnsemble<T>::compute_single(const std::vector<std::vector<T>> &point,
                                     const int index) {
  std::vector<std::vector<Eigen::VectorX<T>>> moms{std_mom(point)};

  // inference
  if (check_div(point)) {
    // the point is near an IR singularity
    // infer over all FKS pairs
    T result{};
    for (int k{0}; k < pairs; ++k) {
      result += infer(moms, index, k);
    }
    return result;
  } else {
    // the point is in a non-divergent region
    // use the 'cut' network which is the final entry in the pair network
    return infer(moms, index, pairs);
  }
}

template <typename T>
T nn::FKSEnsemble<T>::compute(const std::vector<std::vector<T>> &point) {
  std::vector<std::vector<Eigen::VectorX<T>>> moms{std_mom(point)};
  const bool div{check_div(point)};

  // inference
  _mean = T();
  for (int j{0}; j < runs; ++j) {
    if (div) {
      // the point is near an IR singularity
      // infer over all FKS pairs
      for (int k{0}; k < pairs; ++k) {
        _mean += infer(moms, j, k);
      }
    } else {
      // the point is in a non-divergent region
      // use the 'cut' network which is the final entry in the pair network
      _mean += infer(moms, j, pairs);
    }
  }
  _mean /= runs;

  return _mean;
}

template <typename T>
void nn::FKSEnsemble<T>::compute_with_error(const std::vector<std::vector<T>> &point) {
  std::vector<std::vector<Eigen::VectorX<T>>> moms{std_mom(point)};
  const bool div{check_div(point)};

  // inference
  std::vector<T> results(runs);
  for (int j{0}; j < runs; ++j) {
    if (div) {
      // the point is near an IR singularity
      // infer over all FKS pairs
      for (int k{0}; k < pairs; ++k) {
        results[j] += infer(moms, j, k);
      }
    } else {
      // the point is in a non-divergent region
      // use the 'cut' network which is the final entry in the pair network
      results[j] = infer(moms, j, pairs);
    }
  }

  _mean = std::accumulate(results.cbegin(), results.cend(), T()) / runs;

  _std_dev = T();
  for (const T result : results) {
    const T term{result - _mean};
    _std_dev += term * term;
  }
  _std_dev = std::sqrt(_std_dev / runs);

  _std_err = _std_dev / std::sqrt(runs);
}
