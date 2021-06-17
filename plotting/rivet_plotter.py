import numpy as np

import matplotlib.pyplot as plt


class RivetPlotter:
    """
    Extract and plot data from Rivet analysis .dat file
    """

    def __init__(self, rivet_path, dat_file=False):
        """
        Parameters
        ----------
        rivet_path: path to rivet folders
        dat_file: path from rivet_path to specific data file
        """

        self.rivet_path = rivet_path
        self.dat_file = dat_file

    def extract_data(self, return_scales=True, **kwargs):
        """
        Extract data from a data file

        Assumptions
        -----------
        Two graphs per data file

        Parameters
        ----------
        return_scales: True for histograms and False for XS

        Returns
        -------
        ScaledBy value for first graph
        ScaledBy value for second graph
        data for first graph as a 2D list
        data for second graph as a 2D list
        """

        dat_file = kwargs.get("dat_file", self.dat_file)
        if dat_file:
            dat_file = self.rivet_path + dat_file
        else:
            raise ValueError("dat_file is not defined")

        scales = []
        njet_data = []
        nn_data = []

        with open(dat_file, "r") as f:
            line = f.readline()
            extracting_data = False
            data_extracting = 0
            while line:
                line = f.readline()
                if line.startswith("ScaledBy"):
                    scales.append(float(line.split("=")[-1]))
                if line.startswith("# xlow"):
                    extracting_data = True
                if line.startswith("# END HIST"):
                    extracting_data = False
                    data_extracting += 1
                if extracting_data:
                    if data_extracting == 0:
                        njet_data.append(line.split("\t"))
                    elif data_extracting == 1:
                        nn_data.append(line.split("\t"))
                    else:
                        raise ValueError(
                            (
                                "data_extracting is now at {} but should only be 0 or 1"
                            ).format(data_extracting)
                        )

        if return_scales:
            return scales[0], scales[1], njet_data, nn_data
        else:
            return njet_data, nn_data

    def parse_data_step(self, data):
        """
        Parse data extracted from a data file

        Parameters
        ----------
        data: 2D list of histogram data with first element being ignored (titles)
              columns:
                  0: bin lower bounds
                  1: bin upper bounds
                  2: bin values
                  3: +/- bin errors (assume symmetric)

        Returns
        -------
        bins: array of bin limits in np.histogram format (i.e. including lower and upper
              bound)
        vals: bin values with 0'th element duplicated
        errs: bin +/- errors with 0'th element duplicated
        """

        data = np.array(data)
        xlow = data[:, 0][1:].astype("float")
        xhigh = data[:, 1][1:].astype("float")
        vals = data[:, 2][1:].astype("float")
        errs = data[:, 3][1:].astype("float")

        bins = np.append(xlow, xhigh[-1])
        vals = np.append(vals[0], vals)
        errs = np.append(errs[0], errs)
        return bins, vals, errs

    def plot_distribution(
        self,
        njet_bins,
        nn_bins,
        njet_vals,
        nn_vals,
        njet_errs,
        nn_errs,
        ylabel,
        xlabel,
        xlim=None,
        ylim=None,
    ):
        """
        Plot two histograms with ratio plot below

        Returns
        -------
        matplotlib figure environment
        """

        plt.clf()

        fig = plt.figure(1, figsize=(5, 7))

        plt.rc("text", usetex=True)
        plt.rc("font", family="serif")

        ax1 = fig.add_axes((0.1, 0.4, 0.8, 0.8))

        ax1.step(njet_bins, njet_vals, color="red", label="NJet")
        ax1.step(nn_bins, nn_vals, color="blue", label="NN")
        ax1.fill_between(
            njet_bins[:-1],
            (njet_vals - njet_errs)[1:],
            (njet_vals + njet_errs)[1:],
            step="post",
            color="red",
            alpha=0.5,
        )
        ax1.fill_between(
            nn_bins[:-1],
            (nn_vals - nn_errs)[1:],
            (nn_vals + nn_errs)[1:],
            step="post",
            color="blue",
            alpha=0.5,
        )
        ax1.set_yscale("log")
        ax1.set_xticklabels([])

        ax1.tick_params(axis="x", labelsize=15, direction="in", top=True)
        ax1.tick_params(
            axis="y", labelsize=15, direction="in", right=True, which="both"
        )

        if xlim is not None:
            ax1.set_xlim(xlim)
        else:
            ax1.set_xlim((0, njet_bins[-1]))

        if ylim is not None:
            ax1.set_ylim(ylim)

        ax1.set_ylabel(r"{}".format(ylabel), fontsize=17, labelpad=10)
        ax1.legend(prop={"size": 17}, frameon=False)

        ax2 = fig.add_axes((0.1, 0.1, 0.8, 0.3))

        ax2.step(njet_bins, np.ones(len(njet_vals)), color="red")
        ax2.step(njet_bins, nn_vals / njet_vals, color="blue")
        ax2.fill_between(
            njet_bins[:-1],
            (np.ones(len(njet_vals)) - (njet_errs / njet_vals))[1:],
            (np.ones(len(njet_vals)) + (njet_errs / njet_vals))[1:],
            step="post",
            color="red",
            alpha=0.5,
        )
        ax2.fill_between(
            nn_bins[:-1],
            ((nn_vals / njet_vals) - (nn_errs / njet_vals))[1:],
            ((nn_vals / njet_vals) + (nn_errs / njet_vals))[1:],
            step="post",
            color="blue",
            alpha=0.5,
        )

        if xlim is not None:
            ax2.set_xlim(xlim)
        else:
            ax2.set_xlim((0, njet_bins[-1]))
        ax2.set_ylim((0.4, 1.6))

        ax2.set_ylabel(r"Ratio", fontsize=17, labelpad=10)
        ax2.set_xlabel(r"{}".format(xlabel), fontsize=17, labelpad=10)
        ax2.tick_params(axis="x", labelsize=15, direction="in", top=True)
        ax2.tick_params(
            axis="y", labelsize=15, direction="in", right=True, which="both"
        )

        return fig

    def rescale(
        self,
        rescaling,
        njet_vals,
        nn_vals,
        njet_errs,
        nn_errs,
        njet_scale=1,
        nn_scale=1,
    ):

        if rescaling == "On":
            njet_vals_pass = njet_vals
            nn_vals_pass = nn_vals
            njet_errs_pass = njet_errs
            nn_errs_pass = nn_errs

        elif rescaling == "Off":
            njet_vals_pass = njet_vals / njet_scale
            nn_vals_pass = nn_vals / nn_scale
            njet_errs_pass = njet_errs / njet_scale
            nn_errs_pass = nn_errs / nn_scale

        elif rescaling == "XS":
            njet_vals_pass = njet_vals / np.sum(njet_vals)
            nn_vals_pass = nn_vals / np.sum(nn_vals)
            njet_errs_pass = njet_errs / np.sum(njet_vals)
            nn_errs_pass = nn_errs / np.sum(nn_vals)

        else:
            raise ValueError(
                "rescaling takes values: On/XS/Off but you have passed {}".format(
                    rescaling
                )
            )

        return njet_vals_pass, nn_vals_pass, njet_errs_pass, nn_errs_pass

    def plot(self, xlabel, ylabel, xlim=None, ylim=None, rescaling="On"):

        njet_scale, nn_scale, njet_data, nn_data = self.extract_data()
        assert len(njet_data) == len(nn_data)

        njet_bins, njet_vals, njet_errs = self.parse_data_step(njet_data)
        nn_bins, nn_vals, nn_errs = self.parse_data_step(nn_data)

        njet_vals_pass, nn_vals_pass, njet_errs_pass, nn_errs_pass = self.rescale(
            rescaling=rescaling,
            njet_vals=njet_vals,
            nn_vals=nn_vals,
            njet_errs=njet_errs,
            nn_errs=nn_errs,
            njet_scale=njet_scale,
            nn_scale=nn_scale,
        )

        fig = self.plot_distribution(
            njet_bins=njet_bins,
            nn_bins=nn_bins,
            njet_vals=njet_vals_pass,
            nn_vals=nn_vals_pass,
            njet_errs=njet_errs_pass,
            nn_errs=nn_errs_pass,
            xlabel=xlabel,
            ylabel=ylabel,
            xlim=xlim,
            ylim=ylim,
        )

        return fig

    def plot_errors(
        self,
        xlabel,
        ylabel,
        xlim=None,
        ylim=None,
        training_reruns=20,
        rescaling="On",
    ):

        njet_bins_all = []
        njet_vals_all = []
        njet_errs_all = []
        nn_bins_all = []
        nn_vals_all = []
        nn_vals_dataset_all = []

        for i in range(training_reruns):
            dfile = "rivet-plots-{}/".format(i) + self.dat_file

            njet_scale, nn_scale, njet_data, nn_data = self.extract_data(
                dat_file=dfile,
                return_scales=True,
            )
            njet_bins, njet_vals, njet_errs = self.parse_data_step(njet_data)
            njet_bins_all.append(njet_bins)
            njet_vals_all.append(njet_vals)
            njet_errs_all.append(njet_errs)

            nn_bins, nn_vals, _ = self.parse_data_step(nn_data)
            nn_bins_all.append(nn_bins)
            nn_vals_all.append(nn_vals)

            dfile = "rivet-plots-dataset-{}/".format(i) + self.dat_file

            njet_scale, nn_scale, njet_data, nn_data = self.extract_data(
                dat_file=dfile,
                return_scales=True,
            )
            nn_bins, nn_vals, _ = self.parse_data_step(nn_data)
            nn_vals_dataset_all.append(nn_vals)

        njet_bins = njet_bins_all[0]
        njet_vals = njet_vals_all[0]
        njet_errs = njet_errs_all[0]

        nn_bins = nn_bins_all[0]
        nn_vals = np.mean(nn_vals_all, axis=0)
        nn_errs = (
            np.sqrt(
                np.std(nn_vals_all, axis=0) ** 2
                + np.std(nn_vals_dataset_all, axis=0) ** 2
            )
        ) / np.sqrt(20)

        njet_vals_pass, nn_vals_pass, njet_errs_pass, nn_errs_pass = self.rescale(
            rescaling=rescaling,
            njet_vals=njet_vals,
            nn_vals=nn_vals,
            njet_errs=njet_errs,
            nn_errs=nn_errs,
        )

        fig = self.plot_distribution(
            njet_bins=njet_bins,
            nn_bins=nn_bins,
            njet_vals=njet_vals_pass,
            nn_vals=nn_vals_pass,
            njet_errs=njet_errs_pass,
            nn_errs=nn_errs_pass,
            xlabel=xlabel,
            ylabel=ylabel,
            xlim=xlim,
            ylim=ylim,
        )

        return fig
