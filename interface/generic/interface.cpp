#include <cassert>
#include <chrono>
#include <cstddef>
#include <iostream>
#include <string>
#include <vector>

#if defined(NJET) || defined(BOTH)
#include "njet.h"
#endif

#include "interface.hpp"

// Use (void) to silent unused warnings.
#define assertm(exp, msg) assert(((void)msg, exp))

// SquaredMatrixElement class constructor and member implementations

NN2A::SquaredMatrixElement::SquaredMatrixElement()
    : zero(0.), m_alpha(1. / 137.035999084), m_alphas(0.118), m_mur(91.188), x(1e-2),
#if (defined(NN) || defined(BOTH))
#ifdef NAIVE
      networks(NN2A::legs, training_reruns, NN_MODEL, "cut_0.02/"),
#else
      networks(NN2A::legs, training_reruns, NN_MODEL, delta, "cut_0.02/"),
#endif
#endif
#ifdef INDEX
      resfile("res-" + std::to_string(INDEX)),
#else
      resfile("res"),
#endif
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

double NN2A::SquaredMatrixElement::Calculate(const ATOOLS::Vec4D_Vector &point) {
#ifdef UNIT
  return 1.;
#endif

#if defined(NN) || defined(BOTH)
#ifdef TIMING
  TP nnt1{std::chrono::high_resolution_clock::now()};
#endif
  std::vector<std::vector<double>> momenta(legs, std::vector<double>(d));
  for (int i{0}; i < legs; ++i) {
    for (int j{0}; j < d; ++j) {
      momenta[i][j] = point[i][j];
    }
  }
#ifdef INDEX
  const double nn_ans{networks.compute_single(momenta, INDEX)};
#else
  const double nn_ans{networks.compute(momenta)};
#endif
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
  double LHMomenta[legs * n];
  for (int p{0}; p < legs; ++p) {
    for (int mu{0}; mu < d; ++mu) {
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
    for (int l{0}; l < legs; ++l) {
      for (int mu{0}; mu < d; ++mu) {
        results_vector.push_back(point[l][mu]);
      }
    }
#if defined(NN) || defined(BOTH)
    results_vector.push_back(nn_ans);
#endif

#if defined(NJET) || defined(BOTH)
    results_vector.push_back(njet_ans);
#endif

#if (defined(NN) || defined(BOTH)) && defined(TIMING)
    results_vector.push_back(nndur);
#endif

#if (defined(NJET) || defined(BOTH)) && defined(TIMING)
    results_vector.push_back(njetdur);
#endif

    results_buffer.push_back(results_vector);
  }
#endif

#ifdef NJET
  return njet_ans;
#endif

#ifdef NN
  return nn_ans;
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
#elif LEGS == 6
  // check for g g  -> g g a a
  assert(fl[0].Kfcode() == kf_gluon && fl[1].Kfcode() == kf_gluon &&
         fl[4].Kfcode() == kf_gluon && fl[5].Kfcode() == kf_gluon &&
         fl[2].Kfcode() == kf_photon && fl[3].Kfcode() == kf_photon);
#elif LEGS == 4
  // check for g g  -> a a
  assert(fl[0].Kfcode() == kf_gluon && fl[1].Kfcode() == kf_gluon &&
         fl[2].Kfcode() == kf_photon && fl[3].Kfcode() == kf_photon);
#endif
  return new NN2A::Process(pi, fl, 0, 0);
}
