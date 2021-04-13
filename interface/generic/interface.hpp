#pragma once

#include <array>
#include <chrono>
#include <complex>
#include <iostream>
#include <string>
#include <vector>

#include "ATOOLS/Org/Run_Parameter.H"
#include "MODEL/Main/Model_Base.H"
#include "MODEL/Main/Running_AlphaS.H"
#include "PHASIC++/Process/ME_Generator_Base.H"
#include "PHASIC++/Process/Tree_ME2_Base.H"

#include "model_fns.h"

// Define LEGS with compiler flags

namespace NN2A {

constexpr int d { 4 };

#ifndef LEGS
static_assert(false, "You must choose the number of legs with -DLEGS=#!");
#endif

constexpr int legs { LEGS };

#if (defined(NN) || defined(BOTH))
#ifndef RUNS
static_assert(false, "You must choose the number of NN runs to average over with -DRUNS=#!");
#endif
#if (RUNS == 1) && !defined(A)
static_assert(false, "You must set the index of the NN run with -DA=#!");
#endif
#ifndef NN_MODEL
static_assert(false, "You must choose the path to the model directory with -DNN_MODEL=\"\\\"/path/to/model\\\"\"!");
#endif
#endif

#if !defined(NN) && !defined(NJET)
static_assert(false, "You must choose one of -DNN and -DNJET!");
#endif

#if defined(NN) && defined(NJET)
static_assert(false, "You cannot use -DNN and -DNJET at the same time!");
#endif

class SquaredMatrixElement {
public:
    SquaredMatrixElement();
    ~SquaredMatrixElement();
    void PrintSummary() const;
    double Calculate(const ATOOLS::Vec4D_Vector& point);

private:
    using TP = const std::chrono::high_resolution_clock::time_point;

    static constexpr int n { 5 }; // momenta fifth entry is mass

    // binomial_coefficient
    // static constexpr std::array<int, 11> n_choose_2 { { 0, 0, 1, 3, 6, 10, 15, 21, 28, 36, 45 } };

    const double zero, m_alpha, m_alphas, m_mur, delta, x;

#ifdef RUNS
    static constexpr int training_reruns { RUNS };
// #else
//     static constexpr int training_reruns { 0 };
#endif

    // const std::string cut_dirs;
    // const std::string model_base;

    // std::array<std::string, training_reruns> model_dirs;

    // n.b. there is an additional FKS pair for the cut network (for non-divergent regions)
    // static constexpr int pairs { n_choose_2[legs] - 1 };

    // std::array<std::string, pairs> pair_dirs;

#if (defined(NN) || defined(BOTH))
#ifdef NAIVE
    nn::NaiveNetworks networks;
    // std::vector<std::vector<double>> metadatas;
    // std::array<std::string, training_reruns> model_dir_models;
    // std::vector<nn::KerasModel> kerasModels;
#else
    nn::FKSNetworks networks;
    // std::vector<std::vector<std::vector<double>>> metadatas;
    // std::array<std::array<std::string, pairs + 1>, training_reruns> model_dir_models;
    // std::vector<std::vector<nn::KerasModel>> kerasModels;
#endif
#endif

    const std::string resfile;

    std::vector<std::vector<double>> results_buffer;

    double dot(const ATOOLS::Vec4D_Vector& point, int k, int j) const;
};

class Interface : public PHASIC::ME_Generator_Base {
public:
    Interface();

    virtual bool Initialize(
        const std::string& path,
        const std::string& file,
        MODEL::Model_Base* const model,
        BEAM::Beam_Spectra_Handler* const beam,
        PDF::ISR_Handler* const isr) override;

    virtual PHASIC::Process_Base*
    InitializeProcess(const PHASIC::Process_Info& pi, bool add) override;

    virtual int PerformTests() override;

    virtual bool NewLibraries() override;

    virtual void SetClusterDefinitions(PDF::Cluster_Definitions_Base* const defs) override;
};

class Process : public PHASIC::Tree_ME2_Base {
protected:
    SquaredMatrixElement m_me;

public:
    Process(
        const PHASIC::Process_Info& pi,
        const ATOOLS::Flavour_Vector& flavs,
        const bool swap,
        const bool anti);

    virtual double Calc(const ATOOLS::Vec4D_Vector& p) override;

    virtual int OrderQCD(const int& id) override;

    virtual int OrderEW(const int& id) override;
};

} // End namespace NN2A
