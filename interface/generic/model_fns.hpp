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

enum ActivationType { Tanh, Linear };

struct Layer {
  virtual ~Layer(){};

  virtual void load_weights(std::ifstream &input_fname) = 0;
  virtual std::vector<double> compute_output(std::vector<double> test_input) = 0;
};

struct LayerDense : public Layer {
  int input_node_count;
  int output_weights;
  std::vector<std::vector<double>> layer_weights;
  std::vector<double> bias;

  void load_weights(std::ifstream &fin);
  std::vector<double> compute_output(std::vector<double> test_input);
};

struct LayerActivation : public Layer {
  ActivationType activation_type;

  void load_weights(std::ifstream &fin);
  std::vector<double> compute_output(std::vector<double> test_input);
};

class KerasModel {
public:
  ~KerasModel();
  void load_weights(std::string &input_fname);
  std::vector<double> compute_output(std::vector<double> test_input);

private:
  int layers_count;
  std::vector<Layer *> layers;
};

class Networks {
public:
  Networks(int legs_, int runs_, const std::string &model_path, double delta_,
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

class NaiveNetworks : public Networks {
public:
  NaiveNetworks(const int legs, const int runs, const std::string &model_path,
                double delta_, const std::string &cut_dirs_);

  double compute(const std::vector<std::vector<double>> &point);

  std::vector<nn::KerasModel> kerasModels;
  std::vector<std::vector<double>> metadatas;

private:
  std::vector<std::string> model_dir_models;
};

class FKSNetworks : public Networks {
public:
  FKSNetworks(const int legs, const int runs, const std::string &model_path,
              double delta_, const std::string &cut_dirs_);

  double compute(const std::vector<std::vector<double>> &point);

  std::vector<std::vector<nn::KerasModel>> kerasModels;
  std::vector<std::vector<std::vector<double>>> metadatas;

private:
  std::vector<std::vector<std::string>> model_dir_models;
};

} // namespace nn
