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
        "--legs",
        dest="legs",
        help="Number of legs",
        type=int,
    )

    parser.add_argument(
        "--training_reruns",
        dest="training_reruns",
        help="Number of training reruns",
        type=int,
        default=20,
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
    def __init__(
        self,
        rivet_dir,
        save_dir="./paper_plots/",
        legs=5,
        training_reruns=20,
        rescaling="On",
    ):

        self.rivet_dir = rivet_dir
        self.save_dir = save_dir
        self.legs = legs
        self.training_reruns = training_reruns
        self.rescaling = rescaling

    # Angluar plots

    def plot_phi_jj(self):

        dphi_jj_file = "/diphoton/dphijj.dat"
        dphi_jj_plotter = RivetPlotter(rivet_path=self.rivet_dir, dat_file=dphi_jj_file)

        xlabel = r"$\Delta\phi_{j_1j_2}$ [rad]"
        if self.rescaling == "On":
            ylabel = r"d$\sigma/$d$\Delta\phi_{j_1j_2}$ [fb rad$^{-1}$]"
        elif self.rescaling == "XS":
            ylabel = r"$1/\sigma$ d$\sigma/$d$\Delta\phi_{j_1j_2}$ [rad$^{-1}$]"

        dphi_jj_fig = dphi_jj_plotter.plot_errors(
            xlabel=xlabel,
            ylabel=ylabel,
            training_reruns=self.training_reruns,
            rescaling=self.rescaling,
        )
        dphi_jj_fig.savefig(self.save_dir + "dphi_jj.pdf", bbox_inches="tight")

    def plot_r_jy(self):

        dr_jy_file = "/diphoton/rsepjy.dat"
        dr_jy_plotter = RivetPlotter(
            rivet_path=self.rivet_dir,
            dat_file=dr_jy_file,
        )

        xlabel = r"$R_{j_1\gamma_1}$"
        if self.rescaling == "On":
            ylabel = r"d$\sigma/$d$R_{j_1\gamma_1}$ [fb]"
        elif self.rescaling == "XS":
            ylabel = r"$1/\sigma$ d$\sigma/$d$R_{j_1\gamma_1}$"

        dr_jy_fig = dr_jy_plotter.plot_errors(
            xlabel=xlabel,
            ylabel=ylabel,
            xlim=(0.2, 5),
            training_reruns=self.training_reruns,
            rescaling=self.rescaling,
        )
        dr_jy_fig.savefig(self.save_dir + "dr_jy.pdf", bbox_inches="tight")

    # Di-photon plots

    def plot_eta_yy(self):

        deta_yy_file = "/diphoton/etayy.dat"
        deta_yy_plotter = RivetPlotter(
            rivet_path=self.rivet_dir,
            dat_file=deta_yy_file,
        )

        xlabel = r"$\Delta\eta_{\gamma_1\gamma_2}$"
        if self.rescaling == "On":
            ylabel = r"d$\sigma/$d$\Delta\eta_{\gamma_1\gamma_2}$ [fb]"
        elif self.rescaling == "XS":
            ylabel = r"$1/\sigma$ d$\sigma/$d$\Delta\eta_{\gamma_1\gamma_2}$"

        deta_yy_fig = deta_yy_plotter.plot_errors(
            xlabel=xlabel,
            ylabel=ylabel,
            training_reruns=self.training_reruns,
            rescaling=self.rescaling,
        )
        deta_yy_fig.savefig(self.save_dir + "deta_yy.pdf", bbox_inches="tight")

    def plot_m_yy(self):

        dm_yy_file = "/diphoton/mass.dat"
        dm_yy_plotter = RivetPlotter(
            rivet_path=self.rivet_dir,
            dat_file=dm_yy_file,
        )

        xlabel = r"$m_{\gamma_1\gamma_2}$"
        if self.rescaling == "On":
            ylabel = r"d$\sigma/$d$m_{\gamma_1\gamma_2}$ [fb]"
        elif self.rescaling == "XS":
            ylabel = r"$1/\sigma$ d$\sigma/$d$m_{\gamma_1\gamma_2}$"

        if self.legs == 5:
            ylim = (11e-7, 3)
        else:
            ylim = None

        dm_yy_fig = dm_yy_plotter.plot_errors(
            xlabel=xlabel,
            ylabel=ylabel,
            ylim=ylim,
            training_reruns=self.training_reruns,
            rescaling=self.rescaling,
        )
        dm_yy_fig.savefig(self.save_dir + "dm_yy.pdf", bbox_inches="tight")

    # p_T plots

    def plot_pt_j1(self):

        dpt_j1_file = "/diphoton/j1pt.dat"
        dpt_j1_plotter = RivetPlotter(
            rivet_path=self.rivet_dir,
            dat_file=dpt_j1_file,
        )

        xlabel = r"$p_{T,j_1}$ [GeV]"
        if self.rescaling == "On":
            ylabel = r"d$\sigma/$d$p_{T,j_1}$ [fb GeV$^{-1}$]"
        elif self.rescaling == "XS":
            ylabel = r"$1/\sigma$ d$\sigma/$d$p_{T,j_1}$ [GeV$^{-1}$]"

        if self.legs == 6:
            ylim = (10e-7,10e-1)
        else:
            ylim = None

        dpt_j1_fig = dpt_j1_plotter.plot_errors(
            xlabel=xlabel,
            ylabel=ylabel,
            training_reruns=self.training_reruns,
            rescaling=self.rescaling,
            xlim=(0,500),
            ylim=ylim
        )
        dpt_j1_fig.savefig(self.save_dir + "dpt_j1.pdf", bbox_inches="tight")

    def plot_pt_j2(self):

        dpt_j2_file = "/diphoton/j2pt.dat"
        dpt_j2_plotter = RivetPlotter(
            rivet_path=self.rivet_dir,
            dat_file=dpt_j2_file,
        )

        xlabel = r"$p_{T,j_2}$ [GeV]"
        if self.rescaling == "On":
            ylabel = r"d$\sigma/$d$p_{T,j_2}$ [fb GeV$^{-1}$]"
        elif self.rescaling == "XS":
            ylabel = r"$1/\sigma$ d$\sigma/$d$p_{T,j_2}$ [GeV$^{-1}$]"

        if self.legs == 6:
            ylim = (10e-7,10e-1)
        else:
            ylim = None

        dpt_j2_fig = dpt_j2_plotter.plot_errors(
            xlabel=xlabel,
            ylabel=ylabel,
            rescaling=self.rescaling,
            training_reruns=self.training_reruns,
            xlim=(0,500),
            ylim=ylim
        )
        dpt_j2_fig.savefig(self.save_dir + "dpt_j2.pdf", bbox_inches="tight")

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
        legs=args.legs,
        training_reruns=args.training_reruns,
        rescaling=args.rescaling,
    )
    rivet_distributions.plot_all()
