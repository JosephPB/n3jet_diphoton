// TODO
// 3. Naive vs FKS NN implementations
//    https://github.com/JosephPB/n3jet/blob/master/n3jet/c%2B%2B_calls/ex_3g2A_multiple_single.cpp
// 4. Pass average of N training reruns to Sherpa or running each one individually
//    set training_reruns AND training index I with compile flag to 1, then bash script for loop (same rseed, diff -A.I)
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

constexpr int legs { LEGS };

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
    static constexpr std::array<int, 11> n_choose_2 { { 0, 0, 1, 3, 6, 10, 15, 21, 28, 36, 45 } };

    const double zero, m_alpha, m_alphas, m_mur, delta, x;

    // n.b. there is an additional FKS pair for the cut network (for non-divergent regions)
    static constexpr int pairs { n_choose_2[legs] - 1 };
    static constexpr int training_reruns { 20 };

    const std::string cut_dirs;
    const std::string model_base;

    std::array<std::string, training_reruns> model_dirs;
    const std::array<std::string, pairs> pair_dirs;

    std::vector<std::vector<std::vector<double>>> metadatas;
    std::array<std::array<std::string, pairs + 1>, training_reruns> model_dir_models;
    std::vector<std::vector<nn::KerasModel>> kerasModels;

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
