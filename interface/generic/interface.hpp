#pragma once

#include <array>
#include <complex>
#include <iostream>
#include <string>
#include <vector>

#include "ATOOLS/Org/Run_Parameter.H"
#include "MODEL/Main/Model_Base.H"
#include "MODEL/Main/Running_AlphaS.H"
#include "PHASIC++/Process/ME_Generator_Base.H"
#include "PHASIC++/Process/Tree_ME2_Base.H"

#include "model_fns.h" // TODO

// Define LEGS with compiler flags

namespace NN2A {

constexpr int d { 4 };

class SquaredMatrixElement {
public:
    SquaredMatrixElement();
    ~SquaredMatrixElement();
    void PrintSummary() const;
    double Calculate(const ATOOLS::Vec4D_Vector& point);

private:
    static constexpr int n { 5 }; // momenta fifth entry is mass

    const double zero, m_alpha, m_alphas, m_mur, delta, x;

    // n.b. there is an additional FKS pair for the cut network (for non-divergent regions)
    static constexpr int pairs { 5 : 9, 6 : 14 };
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
};

class Interface : public PHASIC::ME_Generator_Base {
public:
    Interface();

    bool Initialize(
        const std::string& path,
        const std::string& file,
        MODEL::Model_Base* const model,
        BEAM::Beam_Spectra_Handler* const beam,
        PDF::ISR_Handler* const isr);

    PHASIC::Process_Base*
    InitializeProcess(const PHASIC::Process_Info& pi, bool add);

    int PerformTests();

    bool NewLibraries();

    void SetClusterDefinitions(PDF::Cluster_Definitions_Base* const defs);

    ATOOLS::Cluster_Amplitude*
    ClusterConfiguration(PHASIC::Process_Base* const proc, const size_t& mode);
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

    double Calc(const ATOOLS::Vec4D_Vector& p);

    int OrderQCD(const int& id);

    int OrderEW(const int& id);
};

} // End namespace NN2A
