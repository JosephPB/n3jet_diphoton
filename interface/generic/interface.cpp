#include <algorithm>
#include <array>
#include <cassert>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstddef>
#include <ctime>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

#if defined(NJET) || defined(BOTH)
#include "njet.h"
#endif

#include "model_fns.h"

#include "interface.hpp"

#define NN_MODEL "100k_new_sherpa"

// Use (void) to silent unused warnings.
#define assertm(exp, msg) assert(((void)msg, exp))

// SquaredMatrixElement class constructor and member implementations

NN2A::SquaredMatrixElement::SquaredMatrixElement()
    : zero(0.)
    , m_alpha(1. / 137.035999084)
    , m_alphas(0.118)
    , m_mur(91.188)
    , delta(2e-2)
    , x(1e-2)
    , cut_dirs("cut_0.02/")
    , model_base("./models/parallel_fixed/" + std::string(NN_MODEL) + "/")
    , model_dirs()
    , pair_dirs({
          "pair_0.02_0/",
          "pair_0.02_1/",
          "pair_0.02_2/",
          "pair_0.02_3/",
          "pair_0.02_4/",
          "pair_0.02_5/",
          "pair_0.02_6/",
          "pair_0.02_7/",
          "pair_0.02_8/",
          "pair_0.02_9/",
          "pair_0.02_10/",
          "pair_0.02_11/",
          "pair_0.02_12/",
          "pair_0.02_13/",
      })
    , metadatas(training_reruns, std::vector<std::vector<double>>(pairs + 1, std::vector<double>(10)))
    , kerasModels(training_reruns, std::vector<nn::KerasModel>(pairs + 1))
    , resfile("res")
    , results_buffer()
{
    std::generate(model_dirs.begin(), model_dirs.end(), [n = 0]() mutable { return std::to_string(n++) + "/"; });
#ifndef NJET
    for (int i { 0 }; i < training_reruns; ++i) {
        // Near networks
        for (int j { 0 }; j < pairs; ++j) {
            std::string metadata_file { model_base + model_dirs[i] + pair_dirs[j] + "dataset_metadata.dat" };
            std::vector<double> metadata { nn::read_metadata_from_file(metadata_file) };
            for (int k { 0 }; k < 10; ++k) {
                metadatas[i][j][k] = metadata[k];
            };
            model_dir_models[i][j] = model_base + model_dirs[i] + pair_dirs[j] + "model.nnet";
            kerasModels[i][j].load_weights(model_dir_models[i][j]);
        };
        // Cut networks
        std::string metadata_file { model_base + model_dirs[i] + cut_dirs + "dataset_metadata.dat" };
        std::vector<double> metadata { nn::read_metadata_from_file(metadata_file) };
        for (int k { 0 }; k < 10; ++k) {
            metadatas[i][pairs][k] = metadata[k];
        };
        model_dir_models[i][pairs] = model_base + model_dirs[i] + cut_dirs + "model.nnet";
        kerasModels[i][pairs].load_weights(model_dir_models[i][pairs]);
    }
#endif
#if defined(NJET) || defined(BOTH)
    const std::string f { "OLE_contract_" + std::to_string(NN2A::legs - 2) + "g2A.lh" };
    const char* contract { f.c_str() };
    int status;
    OLP_Start(contract, &status);
    assertm(status, "There seems to be a problem with the contract file.");
#endif
}

NN2A::SquaredMatrixElement::~SquaredMatrixElement()
{
#ifdef REC
    int a { 0 };
    std::string name;
    std::ifstream file;
    do {
        name = resfile + "." + std::to_string(a++) + ".data";
        file = std::ifstream(name);
    } while (file.is_open());

    std::ofstream o(name, std::ios::trunc);
    o.setf(std::ios_base::scientific);
    o.precision(16);

    for (const std::vector<double>& v : results_buffer) {
        for (const double e : v) {
            o << e << ' ';
        }
        o << '\n';
    }
#endif
    std::cout << "\nGoodbye :)\n";
}

void NN2A::SquaredMatrixElement::PrintSummary() const
{
    msg_Info() << "Using NN2A Interface with the following parameters:"
               << '\n'
               << "  1/alpha = " << 1. / m_alpha << '\n'
               << "  ----------------------------------------"
               << '\n';
}

// code to compute amplitude here
//double NN2A::SquaredMatrixElement::Calculate(const double point[NN2A::legs][NN2A::d]) const
double NN2A::SquaredMatrixElement::Calculate(const ATOOLS::Vec4D_Vector& point)
{
#ifdef TEST
    return 1.;
#endif

#ifdef UNIT
    return 1.;
#endif

    //long double s_23 = point[2][0] * point[3][0] - (point[2][1] * point[3][1] + point[2][2] * point[3][2] + point[2][3] * point[3][3]);

#ifndef NJET
    std::array<double, training_reruns> results;

    // moms is an vector of training_reruns results, each of which is an vector of FKS pairs results, each of which is an vector of flattened momenta
    std::vector<std::vector<std::vector<double>>> moms(training_reruns, std::vector<std::vector<double>>(pairs + 1, std::vector<double>(NN2A::legs * NN2A::d)));

    // flatten momenta
    for (int p { 0 }; p < NN2A::legs; ++p) {
        for (int mu { 0 }; mu < NN2A::d; ++mu) {
            // standardise input
            for (int k { 0 }; k < training_reruns; ++k) {
                for (int j { 0 }; j <= pairs; ++j) {
                    moms[k][j][p * NN2A::d + mu] = nn::standardise(point[p][mu], metadatas[k][j][mu], metadatas[k][j][NN2A::d + mu]);
                }
                moms[k][pairs][p * NN2A::d + mu] = nn::standardise(point[p][mu], metadatas[k][pairs][mu], metadatas[k][pairs][NN2A::d + mu]);
            }
        }
    }

    // cut/near check
    int cut_near { 0 };
    for (int j { 0 }; j < NN2A::legs - 1; ++j) {
        for (int k { j + 1 }; k < NN2A::legs; ++k) {
            const double prod { point[j][0] * point[k][0] - (point[j][1] * point[k][1] + point[j][2] * point[k][2] + point[j][3] * point[k][3]) };
            const double dist { prod / s_com };
            if (dist < delta) {
                cut_near += 1;
            }
        }
    }

    // inference
    //bool cut_network = true;
    //int pair_chosen = 8;

    for (int j { 0 }; j < training_reruns; ++j) {
        if (cut_near >= 1) {
            // the point is near an IR singularity
            // infer over all FKS pairs
            results[j] = 0;
            for (int k { 0 }; k < pairs; ++k) {
                //if (cut_network == false && pair_chosen == k) {
                const double result { kerasModels[j][k].compute_output(moms[j][k])[0] };
                const double result_pair { nn::destandardise(result, metadatas[j][k][8], metadatas[j][k][9]) };
                results[j] += result_pair;
                //} else {
                //const double result_pair = 0.;
                //results[j] += result_pair;
                //}
            }
        } else {
            // the point is in a non-divergent region
            // use the 'cut' network which is the final entry in the pair network
            //if (cut_network == true){
            const double result { kerasModels[j][pairs].compute_output(moms[j][pairs])[0] };
            results[j] = nn::destandardise(result, metadatas[j][pairs][8], metadatas[j][pairs][9]);
            //} else {
            //results[j] = 0.;
            //}
        }
    }

    const double mean { std::accumulate(results.cbegin(), results.cend(), 0.) / training_reruns };

#endif

#if defined(NJET) || defined(BOTH)
    double LHMomenta[NN2A::legs * NN2A::n];
    for (std::size_t p { 0 }; p < NN2A::legs; ++p) {
        for (std::size_t mu { 0 }; mu < NN2A::d; ++mu) {
            LHMomenta[mu + p * NN2A::n] = point[p][mu];
        }
        // Set masses
        LHMomenta[d + p * NN2A::n] = 0.;
    }

    int alphasReturnStatus;
    OLP_SetParameter("alphas", &m_alphas, &zero, &alphasReturnStatus);
    assert(alphasReturnStatus == 1);

    // set alpha QED (answer changes if this changed, so does something)
    int alphaReturnStatus;
    OLP_SetParameter("alpha", &m_alpha, &zero, &alphaReturnStatus);
    assert(alphaReturnStatus == 1);

    double out[11];
    double acc { 0. };
    const int channel { 1 };
    OLP_EvalSubProcess2(&channel, LHMomenta, &m_mur, out, &acc);

    double njet_ans { out[4] };

#endif

#ifdef REC
    {
        std::vector<double> results_vector;
        for (int l { 0 }; l < NN2A::legs; ++l) {
            for (int mu { 0 }; mu < NN2A::d; ++mu) {
                results_vector.push_back(point[l][mu]);
            }
        }
        // std::ofstream o(resfile, std::ios::app);
        // o.setf(std::ios_base::scientific);
        // o.precision(16);
        // for (int l { 0 }; l < NN2A::legs; ++l) {
        //     for (int mu { 0 }; mu < NN2A::d; ++mu) {
        //         o << point[l][mu] << " ";
        //     }
        // }
#ifndef NJET
        results_vector.push_back(mean);
        // o << mean << " ";
#endif
#if defined(NJET) || defined(BOTH)
        results_vector.push_back(njet_ans);
        // o << njet_ans << " ";
#endif
        results_buffer.push_back(results_vector);
        // o << '\n';
    }
#endif

#ifdef NJET
    return njet_ans;
#else
    return mean;
#endif
}

// Interface class constructor and member implementations

NN2A::Interface::Interface()
    : ME_Generator_Base("NN2A")
{
}

bool NN2A::Interface::Initialize(
    const std::string& path, const std::string& file,
    MODEL::Model_Base* const model,
    BEAM::Beam_Spectra_Handler* const beam,
    PDF::ISR_Handler* const isr)
{
    return true;
}

PHASIC::Process_Base*
NN2A::Interface::InitializeProcess(const PHASIC::Process_Info& pi, bool add)
{
    return NULL;
}

int NN2A::Interface::PerformTests()
{
    return 1;
}

bool NN2A::Interface::NewLibraries()
{
    return false;
}

void NN2A::Interface::SetClusterDefinitions(PDF::Cluster_Definitions_Base* const defs) { }

ATOOLS::Cluster_Amplitude*
NN2A::Interface::ClusterConfiguration(PHASIC::Process_Base* const proc, const size_t& mode)
{
    return NULL;
}

// Process class constructor and member implementations

NN2A::Process::Process(
    const PHASIC::Process_Info& pi,
    const ATOOLS::Flavour_Vector& flavs,
    const bool swap, const bool anti)
    : Tree_ME2_Base(pi, flavs)
    , m_me()
{
    //m_me.SetParameter("alpha", AlphaQED());
    //m_me.RecalcDependentParameters();

    //Data_Reader reader(" ", ";", "#", "=");

    m_me.PrintSummary();

    //rpa->gen.AddCitation(1, string("<Description of calculation> from \\cite{xxx:2019yy}"));
}

double NN2A::Process::Calc(const ATOOLS::Vec4D_Vector& p)
{
    if (p.size() != NN2A::legs)
        THROW(fatal_error, "Wrong process.");
    //double moms[NN2A::legs][NN2A::d];
    //for (size_t i { 0 }; i < NN2A::legs; ++i)
    //    for (size_t j { 0 }; j < NN2A::d; ++j)
    //        moms[i][j] = p[i][j];
    return m_me.Calculate(p);
}

int NN2A::Process::OrderQCD(const int& id)
{
    return NN2A::legs - 2;
}

int NN2A::Process::OrderEW(const int& id)
{
    return 2;
}

// End class member implementations

DECLARE_GETTER(NN2A::Interface, "NN2A", PHASIC::ME_Generator_Base, PHASIC::ME_Generator_Key);

PHASIC::ME_Generator_Base*
ATOOLS::Getter<PHASIC::ME_Generator_Base, PHASIC::ME_Generator_Key, NN2A::Interface>::
operator()(const PHASIC::ME_Generator_Key& key) const
{
    return new NN2A::Interface();
}

void ATOOLS::Getter<PHASIC::ME_Generator_Base, PHASIC::ME_Generator_Key, NN2A::Interface>::
    PrintInfo(std::ostream& str, const size_t width) const
{
    str << "Interface to the NN2A calculation";
}

using namespace PHASIC;

DECLARE_TREEME2_GETTER(NN2A::Process, "NN2A::Process")

PHASIC::Tree_ME2_Base*
ATOOLS::Getter<PHASIC::Tree_ME2_Base, PHASIC::Process_Info, NN2A::Process>::
operator()(const PHASIC::Process_Info& pi) const
{
    assert(pi.m_loopgenerator == "NN2A");
    assert(MODEL::s_model->Name() == std::string("SM"));
    assert(pi.m_fi.m_nloewtype == nlo_type::lo);
    assert(pi.m_fi.m_nloqcdtype == nlo_type::lo);
    Flavour_Vector fl(pi.ExtractFlavours());

    // TODO if n=5
    // // check for g g  -> g a a
    // assert(fl[0].Kfcode() == kf_gluon && fl[1].Kfcode() == kf_gluon && fl[4].Kfcode() == kf_gluon
    //     && fl[2].Kfcode() == kf_photon && fl[3].Kfcode() == kf_photon);

    // check for g g  -> g g a a
    assert(fl[0].Kfcode() == kf_gluon && fl[1].Kfcode() == kf_gluon && fl[4].Kfcode() == kf_gluon && fl[5].Kfcode() == kf_gluon
        && fl[2].Kfcode() == kf_photon && fl[3].Kfcode() == kf_photon);
    return new NN2A::Process(pi, fl, 0, 0);
}
