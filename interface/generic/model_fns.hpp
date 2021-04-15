#pragma once

#include <array>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace nn {

std::vector<double> read_metadata_from_file(const std::string &fname);
double standardise(double value, double mean, double stnd);
double destandardise(double value, double mean, double stnd);

enum ActivationType { Tanh, ReLU, Linear };

struct Layer {
  virtual ~Layer(){};

  virtual std::vector<double> compute_output(std::vector<double> test_input) = 0;
};

struct LayerDense : public Layer {
  LayerDense(std::ifstream &fin);

  int input_node_count;
  int output_weights;
  std::vector<std::vector<double>> layer_weights;
  std::vector<double> bias;

  std::vector<double> compute_output(std::vector<double> test_input);
};

struct LayerActivation : public Layer {
  LayerActivation(std::ifstream &fin);

  ActivationType activation_type;
  std::vector<double> compute_output(std::vector<double> test_input);
};

class Network {
public:
  ~Network();
  void load_weights(std::string &input_fname);
  double compute_output(std::vector<double> test_input);

private:
  int layers_count;
  std::vector<Layer *> layers;
};

class Ensemble {
public:
  Ensemble(int legs_, int runs_, const std::string &model_path, double delta_,
           const std::string &cut_dirs_);

private:
  // binomial coefficients
  const std::array<int, 11> n_choose_2;

protected:
  static constexpr int d{4};
  const int legs;
  const int runs;
  const int pairs;
  const double delta;
  const std::string cut_dirs;
  const std::string model_base;

  std::vector<std::string> model_dirs;
  std::vector<std::string> pair_dirs;

  double dot(const std::vector<std::vector<double>> &point, int k, int j) const;
};

class NaiveEnsemble : public Ensemble {
public:
  NaiveEnsemble(const int legs, const int runs, const std::string &model_path,
                double delta_, const std::string &cut_dirs_);

  double compute(const std::vector<std::vector<double>> &point);

  std::vector<nn::Network> kerasModels;
  std::vector<std::vector<double>> metadatas;

private:
  std::vector<std::string> model_dir_models;
};

class FKSEnsemble : public Ensemble {
public:
  FKSEnsemble(const int legs, const int runs, const std::string &model_path,
              double delta_, const std::string &cut_dirs_);

  double compute(const std::vector<std::vector<double>> &point);

  std::vector<std::vector<nn::Network>> kerasModels;
  std::vector<std::vector<std::vector<double>>> metadatas;

private:
  std::vector<std::vector<std::string>> model_dir_models;
};

} // namespace nn
