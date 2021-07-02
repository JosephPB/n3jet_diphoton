#!/usr/bin/env python3

import sys
import os

import numpy

dirname = os.path.dirname(__file__)
path1 = os.path.join(dirname, "..")
sys.path.insert(0, path1)

from analyse import read, write


if __name__ == "__main__":
    data = read("result.old.csv")
    fix = read("result.fix.csv")

    data[:, 4:7] = fix[:, 4:7]

    new_file = "result.4g2a.csv"
    numpy.savetxt(new_file, data, delimiter=" ")
    write(new_file, data)  # pickle cache
