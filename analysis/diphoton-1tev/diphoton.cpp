// -*- C++ -*-

// created by editing this standard analysis
// https://rivet.hepforge.org/analyses/ATLAS_2017_I1591327.html

#include "Rivet/Analysis.hh"
#include "Rivet/Projections/FastJets.hh"
#include "Rivet/Projections/FinalState.hh"
#include "Rivet/Projections/IdentifiedFinalState.hh"

namespace Rivet {

/// Isolated diphoton + X differential cross-sections
class diphoton : public Analysis {
public:
  // Constructor
  DEFAULT_RIVET_ANALYSIS_CTOR(diphoton);

  // Book histograms and initialise projections before the run
  void init() {

    // Particle parton(93, FourMomentum(0,0,0,0));

    // AnalysisHandler::setRunBeams(ParticlePair(parton, parton));

    const FinalState fs;
    declare(fs, "FS");

    FastJets fj(fs, FastJets::ANTIKT, 0.4);
    _area_def = new fastjet::AreaDefinition(fastjet::VoronoiAreaSpec());
    fj.useJetArea(_area_def);
    declare(fj, "AntiKtJetsD05");

    // PT CUT
    IdentifiedFinalState photonfs(Cuts::abseta < 2.37 && Cuts::pT > 30 * GeV);
    photonfs.acceptId(PID::PHOTON);
    declare(photonfs, "Photon");

    int num_bin{100};

    // Histograms
    _h_M = bookHisto1D("mass", num_bin, 0, 1000);
    _h_pT = bookHisto1D("pt", num_bin, 0, 500);
    _h_at = bookHisto1D("at", num_bin, 0, 200);
    _h_phistar = bookHisto1D("phistar", num_bin, 0.001, 10);
    _h_costh = bookHisto1D("costh", num_bin, 0, 1);
    _h_dphiyy = bookHisto1D("dphiyy", num_bin, 0, 3.14);
    _h_dphijj = bookHisto1D("dphijj", num_bin, 0, 3.14);
    _h_rsepjy = bookHisto1D("rsepjy", num_bin, 0, 5);
    _h_j1pt = bookHisto1D("j1pt", num_bin, 0, 600);
    _h_j2pt = bookHisto1D("j2pt", num_bin, 0, 600);
    _h_etayy = bookHisto1D("etayy", num_bin, 0, 3.14);
    _h_etajj = bookHisto1D("etajj", num_bin, 0, 6);
  }

  // Perform the per-event analysis
  void analyze(const Event &event) {

    // Require at least 2 photons in final state
    const Particles photons =
        apply<IdentifiedFinalState>(event, "Photon").particlesByPt();
    if (photons.size() < 2)
      vetoEvent;

    // Compute the median energy density
    _ptDensity.clear();
    _sigma.clear();
    _Njets.clear();
    vector<vector<double>> ptDensities;
    vector<double> emptyVec;
    ptDensities.assign(ETA_BINS.size() - 1, emptyVec);

    // Get jets, and corresponding jet areas
    const shared_ptr<fastjet::ClusterSequenceArea> clust_seq_area =
        applyProjection<FastJets>(event, "AntiKtJetsD05").clusterSeqArea();
    for (const fastjet::PseudoJet &jet :
         apply<FastJets>(event, "AntiKtJetsD05").pseudoJets(0.0 * GeV)) {
      const double aeta = fabs(jet.eta());
      const double pt = jet.perp();
      const double area = clust_seq_area->area(jet);
      if (area < 1e-3)
        continue;
      const int ieta = binIndex(aeta, ETA_BINS);
      if (ieta != -1)
        ptDensities[ieta].push_back(pt / area);
    }

    // Compute median jet properties over the jets in the event
    for (size_t b = 0; b < ETA_BINS.size() - 1; ++b) {
      double median = 0.0, sigma = 0.0;
      int Njets = 0;
      if (ptDensities[b].size() > 0) {
        std::sort(ptDensities[b].begin(), ptDensities[b].end());
        int nDens = ptDensities[b].size();
        median = (nDens % 2 == 0)
                     ? (ptDensities[b][nDens / 2] + ptDensities[b][(nDens - 2) / 2]) / 2
                     : ptDensities[b][(nDens - 1) / 2];
        sigma = ptDensities[b][(int)(.15865 * nDens)];
        Njets = nDens;
      }
      _ptDensity.push_back(median);
      _sigma.push_back(sigma);
      _Njets.push_back(Njets);
    }

    // Select two hardest jets
    const std::vector<fastjet::PseudoJet> jets = fastjet::sorted_by_E(
        apply<FastJets>(event, "AntiKtJetsD05").pseudoJets(0.0 * GeV));
    const fastjet::PseudoJet j1 = jets[0];
    const fastjet::PseudoJet j2 = jets[1];

    // Loop over photons and fill vector of isolated ones
    Particles isolated_photons;
    for (const Particle &photon : photons) {
      // Check if it's a prompt photon (needed for SHERPA 2->5 sample, otherwise I also
      // get photons from hadron decays in jets)
      if (!photon.isPrompt())
        continue;

      const double eta_P = photon.eta();
      const double phi_P = photon.phi();

      // Compute isolation via particles within an R=0.4 cone of the photon
      const Particles fs = apply<FinalState>(event, "FS").particles();
      FourMomentum mom_in_EtCone;
      for (const Particle &p : fs) {
        // Reject if not in cone
        if (deltaR(photon.momentum(), p.momentum()) > 0.4)
          continue;
        // Sum momentum
        mom_in_EtCone += p.momentum();
      }

      // Add isolated photon to list
      isolated_photons.push_back(photon);
    }

    // Require at least two isolated photons
    if (isolated_photons.size() < 2) {
      vetoEvent;
    }

    // Select leading pT pair
    sortByPt(isolated_photons);
    const FourMomentum y1 = isolated_photons[0];
    const FourMomentum y2 = isolated_photons[1];

    // Select hardest photon
    sortByE(isolated_photons);
    const FourMomentum yh = isolated_photons[0];

    // Leading photon should have pT > 40 GeV, subleading > 30 GeV
    if (y1.pT() < 40. * GeV) {
      vetoEvent;
    }
    if (y2.pT() < 30. * GeV) {
      vetoEvent;
    }

    // Require the two photons to be separated (dR>0.4)
    if (deltaR(y1, y2) < 0.4) {
      vetoEvent;
    }

    // myy
    const FourMomentum yy = y1 + y2;
    const double Myy = yy.mass();

    // pTyy
    const double pTyy = yy.pT();

    // diphoton rapidity
    const double etayy = yy.eta();

    // dphiyy
    const double dphiyy = mapAngle0ToPi(y1.phi() - y2.phi());

    // dphijj
    const double dphijj = mapAngle0ToPi(j1.phi() - j2.phi());

    // phi*
    const double costhetastar_ = fabs(tanh((y1.eta() - y2.eta()) / 2.));
    const double sinthetastar_ = sqrt(1. - pow(costhetastar_, 2));
    const double phistar = tan(0.5 * (PI - dphiyy)) * sinthetastar_;

    // a_t
    const Vector3 t_hat(y1.x() - y2.x(), y1.y() - y2.y(), 0.);
    const double factor = t_hat.mod();
    const Vector3 t_hatx(t_hat.x() / factor, t_hat.y() / factor, t_hat.z() / factor);
    const Vector3 At(y1.x() + y2.x(), y1.y() + y2.y(), 0.);
    // Compute a_t transverse component with respect to t_hat
    const double at = At.cross(t_hatx).mod();

    // R-separation of hardest jet, j1, and hardest photon, yh
    const double rsepjy{deltaR(FourMomentum(j1.E(), j1.px(), j1.py(), j1.pz()), yh)};

    // leading jet pT
    const double j1pt{j1.pt()};

    // next to leading jet pT
    const double j2pt{j2.pt()};

    // rapidity of two hardest jets
    const fastjet::PseudoJet jj{j1 + j2};
    const double etajj{jj.rap()};

    // Fill histograms
    _h_M->fill(Myy);
    _h_pT->fill(pTyy);
    _h_dphiyy->fill(dphiyy);
    _h_dphijj->fill(dphijj);
    _h_costh->fill(costhetastar_);
    _h_phistar->fill(phistar);
    _h_at->fill(at);
    _h_rsepjy->fill(rsepjy);
    _h_j1pt->fill(j1pt);
    _h_j2pt->fill(j2pt);
    _h_etayy->fill(etayy);
    _h_etajj->fill(etajj);
  }

  // Normalise histograms etc., after the run
  void finalize() {
    const double sf = 1.; // crossSection() / femtobarn / sumOfWeights();
    scale(_h_M, sf);
    scale(_h_pT, sf);
    scale(_h_dphiyy, sf);
    scale(_h_dphijj, sf);
    scale(_h_costh, sf);
    scale(_h_phistar, sf);
    scale(_h_at, sf);
    scale(_h_rsepjy, sf);
    scale(_h_j1pt, sf);
    scale(_h_j2pt, sf);
    scale(_h_etayy, sf);
    scale(_h_etajj, sf);
  }

private:
  Histo1DPtr _h_M, _h_pT, _h_dphiyy, _h_dphijj, _h_costh, _h_phistar, _h_at, _h_rsepjy,
      _h_j1pt, _h_j2pt, _h_etayy, _h_etajj;

  fastjet::AreaDefinition *_area_def;

  const vector<double> ETA_BINS = {0.0, 1.5, 3.0};
  vector<double> _ptDensity, _sigma, _Njets;
};

DECLARE_RIVET_PLUGIN(diphoton);

} // namespace Rivet
