#!/usr/bin/env python3

import argparse
import pathlib
import pickle

import numpy
import matplotlib.pyplot
import matplotlib.cm

# IO


def _read(filename):
    if not pathlib.Path(filename).is_file():
        print(f"File {filename} not found")
        exit()
    with open(filename, "r") as filecont:
        data = numpy.genfromtxt(filecont, delimiter=" ")
    return data[data[:, 0].argsort()]


def write(filename, data):
    picklefile = filename + ".pickle"
    if pathlib.Path(picklefile).is_file():
        print(f"File {picklefile} already exists.")
        while True:
            decision = input("Overwrite? [y/n]")
            if decision == "n":
                return None
            elif decision == "y":
                break
    with open(picklefile, "wb") as f:
        pickle.dump(data, f)


def read_cache(filename):
    picklefile = filename + ".pickle"
    if pathlib.Path(picklefile).is_file():
        with open(picklefile, "rb") as f:
            return pickle.load(f)


def read(filename):
    data = read_cache(filename)
    if data is None:
        data = _read(filename)
        write(filename, data)
    return data


def _save(fig, filename):
    if pathlib.Path(filename).is_file():
        print(f"Overwriting existing file {filename}")
    else:
        print(f"Writing new file {filename}")

    fig.savefig(
        filename,
        bbox_inches="tight",
        dpi=300,
    )


def save(fig, name):
    for suffix in ("png", "pdf"):
        # for suffix in ("png",):
        _save(fig, name + "." + suffix)


# vis


def init_plots():
    matplotlib.pyplot.rc("text", usetex=True)
    matplotlib.pyplot.rc("font", family="serif")


def new_plot(xlabel=None, ylabel=None):
    fig, ax = matplotlib.pyplot.subplots(figsize=(6.4, 4.8))

    ax.set_prop_cycle("color", matplotlib.cm.tab10(range(10)))

    ax.tick_params(axis="x", labelsize=15, direction="in", top=True, which="both")
    ax.tick_params(axis="y", labelsize=15, direction="in", right=True, which="both")

    if xlabel:
        ax.set_xlabel(xlabel, fontsize=17, labelpad=10)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=17, labelpad=10)

    return fig, ax


def add_legend(ax, loc="best"):
    ax.legend(loc=loc, prop={"size": 15}, frameon=False)


def hist(cols, titles, name, x_pow_max):
    fig, ax = new_plot(
        xlabel="Frequency",
        ylabel="Evaluation time (ms)",
    )

    matplotlib.pyplot.xscale("log")
    lbins = numpy.logspace(-1, x_pow_max, num=1000, base=10)

    for title, datum in zip(titles, cols):
        ax.hist(
            datum,
            bins=lbins,
            histtype="step",
            label=title,
        )

    add_legend(ax, "upper left")

    save(fig, "hist_" + name)


def vio(rows, titles, name):
    fig, ax = new_plot(
        ylabel="Evaluation time (ms)",
    )

    matplotlib.pyplot.yscale("log")

    pos_x = (1, 2, 3)

    parts = ax.violinplot(
        rows,
        positions=pos_x,
        vert=True,
        showextrema=False,
        showmeans=False,
        showmedians=False,
        widths=0.5,
        points=1000,
    )

    ax.set_xticks(pos_x)
    ax.set_xticklabels(titles)

    cmap = matplotlib.cm.Paired

    for pc in parts["bodies"]:
        pc.set_edgecolor(cmap(1))
        pc.set_facecolor(cmap(0))
        pc.set_alpha(1)

    save(fig, "vio_" + name)


def lin(means, stds, line_labels, x_labels):
    fig, ax = new_plot(
        xlabel="Multiplicity",
        ylabel="Evaluation time (ms)",
    )

    matplotlib.pyplot.yscale("log")

    for label, mean, std in zip(line_labels, means, stds):
        ax.fill_between(
            x=x_labels,
            y1=mean - std,
            y2=mean + std,
            alpha=0.2,
        )
        ax.plot(
            x_labels,
            mean,
            label=label,
        )

    add_legend(ax, "upper left")

    save(fig, "lin")


def sca(means, line_labels, x_labels, name="sca"):
    fig, ax = new_plot(
        xlabel="Multiplicity",
        ylabel="Evaluation time (ms)",
    )

    matplotlib.pyplot.yscale("log")

    fmts = ("x", "^", "o")

    for label, mean, fmt in zip(line_labels, means, fmts):
        ax.scatter(x_labels, mean, marker=fmt, label=label)

    add_legend(ax, "upper left")

    save(fig, name)


def err(means, stds, line_labels, x_labels):
    fig, ax = new_plot(
        xlabel="Multiplicity",
        ylabel="Evaluation time (ms)",
    )

    matplotlib.pyplot.yscale("log")

    for label, mean, std in zip(line_labels, means, stds):
        ax.errorbar(x_labels, mean, yerr=std, label=label, fmt=".", capsize=2)

    add_legend(ax, "upper left")

    save(fig, "err")


# main

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "analyse timing of evaluation for available methods over various multiplicities"
    )
    parser.add_argument(
        "-p",
        "--plot",
        action="store_true",
        help="produce plots",
    )
    args = parser.parse_args()

    titles = ("Numerical", "Analytical", "NN ensemble")

    data_4pt = read("2g2a/result.2g2a.csv")

    times_rows_4pt = data_4pt[:, [3, 6, 9]] / 1e6  # ms
    times_4pt = numpy.transpose(times_rows_4pt)

    data_5pt = read("3g2a/result.3g2a.csv")

    times_rows_5pt = data_5pt[:, [3, 6, 9]] / 1e6  # ms
    times_5pt = numpy.transpose(times_rows_5pt)

    titles_6pt = ("Numerical", "NN ensemble f32", "NN ensemble f64")

    data_6pt = read("4g2a/result.4g2a.csv")

    times_rows_6pt = data_6pt[:, [3, 6, 9]] / 1e6  # ms
    times_6pt = numpy.transpose(times_rows_6pt)

    muls = ("4", "5", "6")

    means = numpy.ma.empty((3, 3))
    stds = numpy.ma.empty((3, 3))

    print("Duration values (ms +- %)")

    for j, (m, times_mul) in enumerate(zip(muls, (times_4pt, times_5pt))):
        print(m)

        for i, (impl, time) in enumerate(zip(titles, times_mul)):
            mean = numpy.mean(time)
            abs_std = numpy.std(time)

            rel_std = abs_std / mean
            print(impl, mean, round(100 * rel_std, 1))

            means[i, j] = mean
            stds[i, j] = abs_std

    print(6)
    for i, impl, time in zip(
        (0, 2, 3),
        titles_6pt,
        times_6pt,
    ):
        mean = numpy.mean(time)
        abs_std = numpy.std(time)

        rel_std = abs_std / mean
        print(impl, mean, round(100 * rel_std, 1))

        if i < 3:
            means[i, 2] = mean
            stds[i, 2] = abs_std

    means[1, 2] = numpy.ma.masked
    stds[1, 2] = numpy.ma.masked

    print()

    min_dur = numpy.min(means)

    print("Duration ratios")

    for impl, means_impl in zip(titles, means):
        print(impl)
        for m, mean in zip(muls, means_impl):
            if mean is not numpy.ma.masked:
                print(m, round(mean / min_dur, 1))

    print()

    if args.plot:
        init_plots()

        sca(means, titles, muls, "timing-ensemble")

        titles_single = ("Numerical", "Analytical", "NN")

        means[[0, 1], :] = means[[0, 1], :] / 2
        means[2, :] = means[2, :] / 20

        sca(means, titles_single, muls, "timing-single")

        # hist(times_5pt, titles, "5", 3)
        # vio(times_rows_5pt, titles, "5")

        # hist(times_6pt, titles_6pt, "6", 5)
        # vio(times_rows_6pt, titles_6pt, "6")
