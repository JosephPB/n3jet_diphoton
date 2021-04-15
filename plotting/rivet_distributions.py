import argparse
import matplotlib

matplotlib.use("Agg")

from rivet_plotter import RivetPlotter


def parse():
    """
    Parse arguments
    """

    parser = argparse.ArgumentParser(
        description="Plotting differential distributions from a Rivet .dat file"
    )

    parser.add_argument(
        "--rivet_dir",
        dest="rivet_dir",
        help="Location of Rivet .dat files",
        type=str,
    )

    parser.add_argument(
        "--save_dir",
        dest="save_dir",
        help="Location of where to save the png files",
        type=str,
        default="./paper_plots/",
    )

    parser.add_argument(
        "--rescaling",
        dest="rescaling",
        help="Turn Rivet ScaledBy feature On/XS/Off",
        type=str,
        default="On",
    )

    args = parser.parse_args()

    return args


class RivetDistributions:
    def __init__(self, rivet_dir, save_dir="./paper_plots/", rescaling="On", dpi=150):

        self.rivet_dir = rivet_dir
        self.save_dir = save_dir
        self.rescaling = rescaling
        self.dpi = dpi

    # Angluar plots

    def plot_phi_jj(self):

        dphi_jj_file = self.rivet_dir + "dphijj.dat"
        dphi_jj_plotter = RivetPlotter(dphi_jj_file)

        xlabel = r"$\Delta\phi_{jj}$ [rad]"
        if self.rescaling == "On":
            ylabel = r"d$\sigma/$d$\Delta\phi_{jj}$ [fb rad$^{-1}$]"
        elif self.rescaling == "XS":
            ylabel = r"$1/\sigma$ d$\sigma/$d$\Delta\phi_{jj}$ [rad$^{-1}$]"

        dphi_jj_fig = dphi_jj_plotter.plot(
            xlabel=xlabel,
            ylabel=ylabel,
            rescaling=self.rescaling,
        )
        dphi_jj_fig.savefig(
            self.save_dir + "dphi_jj.png", dpi=self.dpi, bbox_inches="tight"
        )

    def plot_r_jy(self):

        dr_jy_file = self.rivet_dir + "rsepjy.dat"
        dr_jy_plotter = RivetPlotter(dr_jy_file)

        xlabel = r"$R_{j\gamma}$"
        if self.rescaling == "On":
            ylabel = r"d$\sigma/$d$R_{j\gamma}$ [fb]"
        elif self.rescaling == "XS":
            ylabel = r"$1/\sigma$ d$\sigma/$d$R_{j\gamma}$"

        dr_jy_fig = dr_jy_plotter.plot(
            xlabel=xlabel,
            ylabel=ylabel,
            xlim=(0.2, 5),
            rescaling=self.rescaling,
        )
        dr_jy_fig.savefig(
            self.save_dir + "dr_jy.png", dpi=self.dpi, bbox_inches="tight"
        )

    # Di-photon plots

    def plot_eta_yy(self):

        deta_yy_file = self.rivet_dir + "etayy.dat"
        deta_yy_plotter = RivetPlotter(deta_yy_file)

        xlabel = r"$\eta_{\gamma\gamma}$"
        if self.rescaling == "On":
            ylabel = r"d$\sigma/$d$\eta_{\gamma\gamma}$ [fb]"
        elif self.rescaling == "XS":
            ylabel = r"$1/\sigma$ d$\sigma/$d$\eta_{\gamma\gamma}$"

        deta_yy_fig = deta_yy_plotter.plot(
            xlabel=xlabel,
            ylabel=ylabel,
            rescaling=self.rescaling,
        )
        deta_yy_fig.savefig(
            self.save_dir + "deta_yy.png", dpi=self.dpi, bbox_inches="tight"
        )

    def plot_m_yy(self):

        dm_yy_file = self.rivet_dir + "mass.dat"
        dm_yy_plotter = RivetPlotter(dm_yy_file)

        xlabel = r"$\eta_{\gamma\gamma}$"
        if self.rescaling == "On":
            ylabel = r"d$\sigma/$d$\eta_{\gamma\gamma}$ [fb]"
        elif self.rescaling == "XS":
            ylabel = r"$1/\sigma$ d$\sigma/$d$\eta_{\gamma\gamma}$"

        dm_yy_fig = dm_yy_plotter.plot(
            xlabel=xlabel,
            ylabel=ylabel,
            rescaling=self.rescaling,
        )
        dm_yy_fig.savefig(
            self.save_dir + "dm_yy.png", dpi=self.dpi, bbox_inches="tight"
        )

    # p_T plots

    def plot_pt_j1(self):

        dpt_j1_file = self.rivet_dir + "j1pt.dat"
        dpt_j1_plotter = RivetPlotter(dpt_j1_file)

        xlabel = r"$p_{T}$ [GeV]"
        if self.rescaling == "On":
            ylabel = r"d$\sigma/$d$p_{T}$ [fb GeV$^{-1}$]"
        elif self.rescaling == "XS":
            ylabel = r"$1/\sigma$ d$\sigma/$d$p_{T}$ [GeV$^{-1}$]"

        dpt_j1_fig = dpt_j1_plotter.plot(
            xlabel=xlabel,
            ylabel=ylabel,
            rescaling=self.rescaling,
        )
        dpt_j1_fig.savefig(
            self.save_dir + "dpt_j1.png", dpi=self.dpi, bbox_inches="tight"
        )

    def plot_pt_j2(self):

        dpt_j2_file = self.rivet_dir + "j2pt.dat"
        dpt_j2_plotter = RivetPlotter(dpt_j2_file)

        xlabel = r"$p_{T}$ [GeV]"
        if self.rescaling == "On":
            ylabel = r"d$\sigma/$d$p_{T}$ [fb GeV$^{-1}$]"
        elif self.rescaling == "XS":
            ylabel = r"$1/\sigma$ d$\sigma/$d$p_{T}$ [GeV$^{-1}$]"

        dpt_j2_fig = dpt_j2_plotter.plot(
            xlabel=xlabel,
            ylabel=ylabel,
            rescaling=self.rescaling,
        )
        dpt_j2_fig.savefig(
            self.save_dir + "dpt_j2.png", dpi=self.dpi, bbox_inches="tight"
        )

    def plot_all(self):

        self.plot_phi_jj()
        self.plot_r_jy()
        self.plot_eta_yy()
        self.plot_m_yy()
        self.plot_pt_j1()
        self.plot_pt_j2()


if __name__ == "__main__":

    args = parse()

    rivet_distributions = RivetDistributions(
        rivet_dir=args.rivet_dir,
        save_dir=args.save_dir,
        rescaling=args.rescaling,
    )
    rivet_distributions.plot_all()
