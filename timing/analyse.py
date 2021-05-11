#!/usr/bin/env python3

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
        _save(fig, name + "." + suffix)


# vis


def hist(data, titles):
    fig, ax = matplotlib.pyplot.subplots()

    matplotlib.pyplot.xscale("log")
    bins = numpy.logspace(-1, 3, num=1000, base=10)

    for title, i in zip(titles, range(1, 4)):
        ax.hist(
            data[:, 3 * i] / 1e6,
            bins=bins,
            histtype="step",
            label=title,
        )

    ax.set_xlabel("Evaluation time (ms)")
    ax.set_ylabel("Frequency")
    ax.legend(loc="upper left")

    save(fig, "hist")


def vio(data, titles):
    fig, ax = matplotlib.pyplot.subplots()

    matplotlib.pyplot.yscale("log")

    pos = (1, 2, 3)

    parts = ax.violinplot(
        data[:, [3, 6, 9]] / 1e6,
        positions=pos,
        vert=True,
        showextrema=False,
        showmeans=False,
        showmedians=False,
        widths=0.5,
        points=1000,
    )

    ax.set_xticks(pos)
    ax.set_xticklabels(titles)

    ax.set_ylabel("Evaluation time (ms)")

    cmap = matplotlib.cm.get_cmap("Paired")

    for pc in parts["bodies"]:
        pc.set_edgecolor(cmap(1))
        pc.set_facecolor(cmap(0))
        pc.set_alpha(1)

    save(fig, "vio")


# main

if __name__ == "__main__":
    data = read("3g2a/result.5pt.csv")

    titles = ("Numerical", "Analytical", "Neural net ensemble")

    hist(data, titles)

    vio(data, titles)
