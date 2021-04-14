#include <cassert>
#include <chrono>
#include <cstddef>
#include <iostream>
#include <string>
#include <vector>

#if defined(NJET) || defined(BOTH)
#include "njet.h"
#endif

#include "model_fns.h"

#include "interface.hpp"

// Use (void) to silent unused warnings.
#define assertm(exp, msg) assert(((void)msg, exp))

// SquaredMatrixElement class constructor and member implementations

NN2A::SquaredMatrixElement::SquaredMatrixElement()
    : zero(0.), m_alpha(1. / 137.035999084), m_alphas(0.118), m_mur(91.188),
      delta(2e-2), x(1e-2)
#if (defined(NN) || defined(BOTH))
      ,
      networks(NN2A::legs, training_reruns, NN_MODEL, delta, "cut_0.02/")
#endif
#if (RUNS == 1)
      ,
      resfile("res-" + std::to_string(A))
#else
      ,
      resfile("res")
#endif
      ,
      results_buffer() {
#if defined(NJET) || defined(BOTH)
  const std::string f{"OLE_contract_" + std::to_string(NN2A::legs - 2) + "g2A.lh"};
  const char *contract{f.c_str()};
  int status;
  OLP_Start(contract, &status);
  assertm(status, "There seems to be a problem with the contract file.");
#endif
}

NN2A::SquaredMatrixElement::~SquaredMatrixElement() {
#ifdef REC
  int a{0};
  std::string name;
  std::ifstream file;
  do {
    name = resfile + "." + std::to_string(a++) + ".data";
    file = std::ifstream(name);
  } while (file.is_open());

  std::cout << "\nWriting out results to new file `" << name << "`...\n";

  std::ofstream o(name, std::ios::trunc);
  o.setf(std::ios_base::scientific);
  o.precision(16);

  for (const std::vector<double> &v : results_buffer) {
    for (const double e : v) {
      o << e << ' ';
    }
    o << '\n';
  }
  std::cout << "    Done.\n";
#endif
}

void NN2A::SquaredMatrixElement::PrintSummary() const {
  msg_Info() << "Using NJet/NN " << NN2A::legs - 2
             << "G2A interface to Sherpa with the following parameters:" << '\n'
             << "  mur = " << m_mur << '\n'
             << "  alpha_s = " << m_alphas << '\n'
             << "  alpha = " << m_alpha << '\n'
             << "  1/alpha = " << 1. / m_alpha << '\n'
             << "  ----------------------------------------" << '\n';
}

double NN2A::SquaredMatrixElement::dot(const ATOOLS::Vec4D_Vector &point, int k,
                                       int j) const {
  return point[j][0] * point[k][0] -
         (point[j][1] * point[k][1] + point[j][2] * point[k][2] +
          point[j][3] * point[k][3]);
}

// double NN2A::SquaredMatrixElement::Calculate(const double point[NN2A::legs][NN2A::d])
// const
double NN2A::SquaredMatrixElement::Calculate(const ATOOLS::Vec4D_Vector &point) {
#ifdef UNIT
  return 1.;
#endif

#if defined(NN) || defined(BOTH)
#ifdef TIMING
  TP nnt1{std::chrono::high_resolution_clock::now()};
#endif
  double results_sum{0.};
#ifdef NAIVE
  // moms is an vector of training_reruns results, each of which is an vector of
  // flattened momenta std::array<std::array<double, NN2A::legs * NN2A::d>,
  // training_reruns> moms;
  std::vector<std::vector<double>> moms(training_reruns,
                                        std::vector<double>(NN2A::legs * NN2A::d));

  // flatten momenta
  for (int p{0}; p < NN2A::legs; ++p) {
    for (int mu{0}; mu < NN2A::d; ++mu) {
      // standardise input
      for (int k{0}; k < training_reruns; ++k) {
        moms[k][p * NN2A::d + mu] =
            nn::standardise(point[p][mu], networks.metadatas[k][mu],
                            networks.metadatas[k][NN2A::d + mu]);
      }
    }
  }

  // inference
  for (int j{0}; j < training_reruns; ++j) {
    const double result{networks.kerasModels[j].compute_output(moms[j])[0]};
    results_sum +=
        nn::destandardise(result, networks.metadatas[j][8], networks.metadatas[j][9]);
  }
#else
  // moms is an vector of training_reruns results, each of which is an vector of FKS
  // pairs results, each of which is an vector of flattened momenta
  std::vector<std::vector<std::vector<double>>> moms(
      training_reruns, std::vector<std::vector<double>>(
                           pairs + 1, std::vector<double>(NN2A::legs * NN2A::d)));

  // NN compute_output accepts vectors - could edit model_fns
  // std::array<std::array<std::array<double, NN2A::legs * NN2A::d>, pairs + 1>,
  // training_reruns> moms;

  // flatten momenta
  for (int p{0}; p < NN2A::legs; ++p) {
    for (int mu{0}; mu < NN2A::d; ++mu) {
      // standardise input
      for (int k{0}; k < training_reruns; ++k) {
        for (int j{0}; j <= pairs; ++j) {
          moms[k][j][p * NN2A::d + mu] =
              nn::standardise(point[p][mu], networks.metadatas[k][j][mu],
                              networks.metadatas[k][j][NN2A::d + mu]);
        }
        moms[k][pairs][p * NN2A::d + mu] =
            nn::standardise(point[p][mu], networks.metadatas[k][pairs][mu],
                            networks.metadatas[k][pairs][NN2A::d + mu]);
      }
    }
  }

  const double s_com{dot(point, 0, 1)};

  // cut/near check
  int cut_near{0};
  for (int j{0}; j < NN2A::legs - 1; ++j) {
    for (int k{j + 1}; k < NN2A::legs; ++k) {
      const double prod{dot(point, j, k)};
      const double dist{prod / s_com};
      if (dist < delta) {
        cut_near += 1;
      }
    }
  }

  // inference
  for (int j{0}; j < training_reruns; ++j) {
    if (cut_near >= 1) {
      // the point is near an IR singularity
      // infer over all FKS pairs
      for (int k{0}; k < pairs; ++k) {
        const double result{networks.kerasModels[j][k].compute_output(moms[j][k])[0]};
        results_sum += nn::destandardise(result, networks.metadatas[j][k][8],
                                         networks.metadatas[j][k][9]);
      }
    } else {
      // the point is in a non-divergent region
      // use the 'cut' network which is the final entry in the pair network
      const double result{
          networks.kerasModels[j][pairs].compute_output(moms[j][pairs])[0]};
      results_sum += nn::destandardise(result, networks.metadatas[j][pairs][8],
                                       networks.metadatas[j][pairs][9]);
    }
  }
#endif
  const double mean{results_sum / training_reruns};
#ifdef TIMING
  TP nnt2{std::chrono::high_resolution_clock::now()};
  const long int nndur{
      std::chrono::duration_cast<std::chrono::nanoseconds>(nnt2 - nnt1).count()};
#endif
#endif

#if defined(NJET) || defined(BOTH)
#ifdef TIMING
  TP njett1{std::chrono::high_resolution_clock::now()};
#endif
  double LHMomenta[NN2A::legs * n];
  for (std::size_t p{0}; p < NN2A::legs; ++p) {
    for (std::size_t mu{0}; mu < NN2A::d; ++mu) {
      LHMomenta[mu + p * n] = point[p][mu];
    }
    // Set masses
    LHMomenta[d + p * n] = 0.;
  }

  int alphasReturnStatus;
  OLP_SetParameter("alphas", &m_alphas, &zero, &alphasReturnStatus);
  assert(alphasReturnStatus == 1);

  int alphaReturnStatus;
  OLP_SetParameter("alpha", &m_alpha, &zero, &alphaReturnStatus);
  assert(alphaReturnStatus == 1);

  double out[11];
  double njet_acc{0.};
  const int channel{1};
  OLP_EvalSubProcess2(&channel, LHMomenta, &m_mur, out, &njet_acc);

  double njet_ans{out[4]};
#ifdef TIMING
  TP njett2{std::chrono::high_resolution_clock::now()};
  const long int njetdur{
      std::chrono::duration_cast<std::chrono::nanoseconds>(njett2 - njett1).count()};
#endif
#endif

#ifdef REC
  {
    std::vector<double> results_vector;
    for (int l{0}; l < NN2A::legs; ++l) {
      for (int mu{0}; mu < NN2A::d; ++mu) {
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

#if defined(NN) || defined(BOTH)
    results_vector.push_back(mean);
    // o << mean << " ";
#endif

#if defined(NJET) || defined(BOTH)
    results_vector.push_back(njet_ans);
    // o << njet_ans << " ";
#endif

#if (defined(NN) || defined(BOTH)) && defined(TIMING)
    results_vector.push_back(nndur);
#endif

#if (defined(NJET) || defined(BOTH)) && defined(TIMING)
    results_vector.push_back(njetdur);
#endif

    results_buffer.push_back(results_vector);
    // o << '\n';
  }
#endif

#ifdef NJET
  return njet_ans;
#endif

#ifdef NN
  return mean;
#endif
}

// Interface
// ~~~~~~~~~

NN2A::Interface::Interface()
    : ME_Generator_Base("NN" + std::to_string(NN2A::legs - 2) + "G2A") {}

bool NN2A::Interface::Initialize(const std::string & /* path */,
                                 const std::string & /* file */,
                                 MODEL::Model_Base *const /* model */,
                                 BEAM::Beam_Spectra_Handler *const /* beam */,
                                 PDF::ISR_Handler *const /* isr */) {
  return true;
}

PHASIC::Process_Base *
NN2A::Interface::InitializeProcess(const PHASIC::Process_Info & /* pi */,
                                   bool /* add */) {
  return NULL;
}

int NN2A::Interface::PerformTests() { return 1; }

bool NN2A::Interface::NewLibraries() { return false; }

void NN2A::Interface::SetClusterDefinitions(
    PDF::Cluster_Definitions_Base *const /* defs */) {}

// Process
// ~~~~~~~

NN2A::Process::Process(const PHASIC::Process_Info &pi,
                       const ATOOLS::Flavour_Vector &flavs, const bool /* swap */,
                       const bool /* anti */)
    : Tree_ME2_Base(pi, flavs), m_me() {
  m_me.PrintSummary();

  // rpa->gen.AddCitation(1, string("<Description of calculation> from
  // \\cite{xxx:2019yy}"));
}

double NN2A::Process::Calc(const ATOOLS::Vec4D_Vector &p) {
  if (p.size() != NN2A::legs)
    THROW(fatal_error, "Wrong process.");
  // double moms[NN2A::legs][NN2A::d];
  // for (size_t i { 0 }; i < NN2A::legs; ++i)
  //    for (size_t j { 0 }; j < NN2A::d; ++j)
  //        moms[i][j] = p[i][j];
  return m_me.Calculate(p);
}

int NN2A::Process::OrderQCD(const int & /* id */) { return NN2A::legs - 2; }

int NN2A::Process::OrderEW(const int & /* id */) { return 2; }

// End class member implementations

DECLARE_GETTER(NN2A::Interface, "NN2A", PHASIC::ME_Generator_Base,
               PHASIC::ME_Generator_Key);

PHASIC::ME_Generator_Base *
ATOOLS::Getter<PHASIC::ME_Generator_Base, PHASIC::ME_Generator_Key,
               NN2A::Interface>::operator()(const PHASIC::ME_Generator_Key & /* key */)
    const {
  return new NN2A::Interface();
}

void ATOOLS::Getter<PHASIC::ME_Generator_Base, PHASIC::ME_Generator_Key,
                    NN2A::Interface>::PrintInfo(std::ostream &str,
                                                const std::size_t /* width */) const {
  str << "Interface to the NN/NJet diphoton+gluons calculation";
}

using namespace PHASIC;

DECLARE_TREEME2_GETTER(NN2A::Process, "NN2A::Process")

PHASIC::Tree_ME2_Base *
ATOOLS::Getter<PHASIC::Tree_ME2_Base, PHASIC::Process_Info, NN2A::Process>::operator()(
    const PHASIC::Process_Info &pi) const {
  assert(pi.m_loopgenerator == "NN2A");
  assert(MODEL::s_model->Name() == std::string("SM"));
  assert(pi.m_fi.m_nloewtype == nlo_type::lo);
  assert(pi.m_fi.m_nloqcdtype == nlo_type::lo);
  Flavour_Vector fl(pi.ExtractFlavours());

#if LEGS == 5
  // // check for g g  -> g a a
  assert(fl[0].Kfcode() == kf_gluon && fl[1].Kfcode() == kf_gluon &&
         fl[4].Kfcode() == kf_gluon && fl[2].Kfcode() == kf_photon &&
         fl[3].Kfcode() == kf_photon);
#elif LEGS == 5
  // check for g g  -> g g a a
  assert(fl[0].Kfcode() == kf_gluon && fl[1].Kfcode() == kf_gluon &&
         fl[4].Kfcode() == kf_gluon && fl[5].Kfcode() == kf_gluon &&
         fl[2].Kfcode() == kf_photon && fl[3].Kfcode() == kf_photon);
#endif
  return new NN2A::Process(pi, fl, 0, 0);
}
