import argparse
import matplotlib

from rivet_plotter import RivetPlotter

matplotlib.use("Agg")


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

    args = parser.parse_args()

    return args


class RivetDistributions:
    def __init__(self, rivet_dir, save_dir="./paper_plots/", dpi=150):
        self.rivet_dir = rivet_dir
        self.save_dir = save_dir
        self.dpi = dpi

    def plot_phi_jj(self):

        dphi_jj_file = self.rivet_dir + "dphijj.dat"
        dphi_jj_plotter = RivetPlotter(dphi_jj_file)
        dphi_jj_fig = dphi_jj_plotter.plot(
            xlabel=r"$\Delta\phi_{jj}$ [rad]",
            ylabel=r"d$\sigma/$d$\Delta\phi_{jj}$ [fb rad$^{-1}$]",
        )
        dphi_jj_fig.savefig(
            self.save_dir + "dphi_jj.png", dpi=self.dpi, bbox_inches="tight"
        )

    def plot_r_jy(self):

        dr_jy_file = self.rivet_dir + "rsepjy.dat"
        dr_jy_plotter = RivetPlotter(dr_jy_file)
        dr_jy_fig = dr_jy_plotter.plot(
            xlabel=r"$R_{j\gamma}$",
            ylabel=r"d$\sigma/$d$R_{j\gamma}$ [fb]",
            xlim=(0.2, 5),
        )
        dr_jy_fig.savefig(
            self.save_dir + "dr_jy.png", dpi=self.dpi, bbox_inches="tight"
        )

    def plot_eta_yy(self):

        deta_yy_file = self.rivet_dir + "etayy.dat"
        deta_yy_plotter = RivetPlotter(deta_yy_file)
        deta_yy_fig = deta_yy_plotter.plot(
            xlabel=r"$\eta_{\gamma\gamma}$",
            ylabel=r"d$\sigma/$d$\eta_{\gamma\gamma}$ [fb]",
        )
        deta_yy_fig.savefig(
            self.save_dir + "deta_yy.png", dpi=self.dpi, bbox_inches="tight"
        )

    def plot_m_yy(self):

        dm_yy_file = self.rivet_dir + "mass.dat"
        dm_yy_plotter = RivetPlotter(dm_yy_file)
        dm_yy_fig = dm_yy_plotter.plot(
            xlabel=r"$m_{\gamma\gamma}$ [GeV]",
            ylabel=r"d$\sigma/$d$m_{\gamma\gamma}$ [fb GeV$^{-1}$]",
        )
        dm_yy_fig.savefig(
            self.save_dir + "dm_yy.png", dpi=self.dpi, bbox_inches="tight"
        )

    def plot_pt_j1(self):

        dpt_j1_file = self.rivet_dir + "j1pt.dat"
        dpt_j1_plotter = RivetPlotter(dpt_j1_file)
        dpt_j1_fig = dpt_j1_plotter.plot(
            xlabel=r"$p_{T}$ [GeV]", ylabel=r"d$\sigma/$d$p_{T}$ [fb GeV$^{-1}$]"
        )
        dpt_j1_fig.savefig(
            self.save_dir + "dpt_j1.png", dpi=self.dpi, bbox_inches="tight"
        )

    def plot_pt_j2(self):

        dpt_j2_file = self.rivet_dir + "j2pt.dat"
        dpt_j2_plotter = RivetPlotter(dpt_j2_file)
        dpt_j2_fig = dpt_j2_plotter.plot(
            xlabel=r"$p_{T}$ [GeV]", ylabel=r"d$\sigma/$d$p_{T}$ [fb GeV$^{-1}$]"
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

    rivet_distributions = RivetDistributions(args.rivet_dir, args.save_dir)
    rivet_distributions.plot_all()
