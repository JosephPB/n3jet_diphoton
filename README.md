# [n3jet_diphoton](https://gitlab.com/JosephPB/n3jet_diphoton)

[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![build status](https://gitlab.com/JosephPB/n3jet_diphoton/badges/main/pipeline.svg?ignore_skipped=true)](https://gitlab.com/JosephPB/n3jet_diphoton/-/pipelines)

[Mirrored on GitHub](https://github.com/JosephPB/n3jet_diphoton)

This project builds on the package [`n3jet`](https://github.com/JosephPB/n3jet) to use neural networks to evaluate gluon-initiated diphoton scattering amplitudes and integrate these calls into the Monte Carlo event generator Sherpa.
This demonstrates a high precision and performant simulation pipeline for high multiplicity processes at hadron colliders.
We present this work on [arXiv](https://arxiv.org/abs/2106.09474v1).

## Dependencies

This project uses `Python` 3.6+ and `C++14`.
It also uses the following packages, with tested versions quoted:

| Package                                   | Version |
| ----------------------------------------- | ------- |
| [`NJet`](https://bitbucket.org/njet/njet) | 3.0.0   |
| [`Eigen`](https://eigen.tuxfamily.org)    | 3.3.9   |
| [`Sherpa`](https://sherpa-team.gitlab.io) | 2.2.8   |
| [`Rivet`](https://rivet.hepforge.org)     | 2.7.2   |
| [`YODA`](https://yoda.hepforge.org/)      | 1.7.7   |
| [`FastJet`](http://fastjet.fr/)           | 3.3.3   |

Note that `NJet` must have desired analytic amplitudes enabled at the `configure` stage.

This repo uses [Git LFS](https://git-lfs.github.com/) for binary files.

To run Python scripts and [Jupyter notebooks](https://jupyter.org/) which use `n3jet`, we recommend installing `n3jet` in a Python 2 virtual environment.
This requires `python2` and [`virtualenv`](https://virtualenv.pypa.io), which can be installed via the system package manager or via `pip`
```shell
pip install --user virtualenv
```
A Python 2 virtual environment for `n3jet` can be made and activated with
```shell
mkdir ~/venvs
virtualenv ~/venvs/py2-n3jet -p /usr/bin/python2
source ~/venvs/py2-n3jet/bin/activate
```
Then `n3jet` can be installed with
```shell
git clone https://github.com/JosephPB/n3jet.git
cd n3jet
(py2-n3jet) pip install -e .
```
where I indicate the active virtual environment in parentheses.
Within the `py2-n3jet` virtual environment, Python scripts can now be run that depend on `n3jet`.
To use `py2-n3jet` with a Jupyter notebook, first install `jupyter` either through the system package manager or by `pip`
```shell
pip install --user jupyter
```
then add the `py2-n3jet` virtual environment as a kernel
```shell
pip install --user ipykernel
python -m ipykernel install --user --name=py2-n3jet
```
Then `py2-n3jet` can be selected as the notebook kernel under `Kernel > Change kernel > py2-n3jet` from the notebook interface after starting the notebook with
```shell
jupyter notebook
```

## Usage

This project has been tested on Linux and MacOS.

Each directory using `C++` code contains a `Makefile` to compile it.
It is sufficient to run `make` in each such directory to build all targets.
Executable targets can then be run as `./NAME`.
Error messages will display if command line arguments are required.

`Python` code is presented as either:

-   a plain Python script `NAME.py`, which can be run from the terminal as `./NAME.py`.
-   a Jupyter notebook `NAME.ipynb`. After installing `jupyter`, a notebook server can be started in the directory with `jupyter notebook` to access the files.

The project is laid out as follows:

| Directory   | Description                                                            |
| ----------- | ---------------------------------------------------------------------- |
| `analysis`  | Rivet analysis source code                                             |
| `configs`   | Configuration files for model training and testing                     |
| `diagrams`  | Scripts to generate diagrams for the paper                             |
| `interface` | Source code for the C++ model inference and Sherpa-[NN/NJet] interface |
| `models`    | Model files for the pretrained models                                  |
| `plotting`  | Scripts to generate plots for the paper                                |
| `run`       | Scripts to run Sherpa with NJet and trained models                     |
| `timing`    | Analysis of inference timings for paper                                |

The `interface` directory contains:

| Subirectory | Description                                                    |
| ----------- | -------------------------------------------------------------- |
| `generic`   | Source for the C++ NN inference and Sherpa-[NN/NJet] interface |
| `Ng2A-njet` | The Sherpa-NJet interface libraries                            |
| `Ng2A-nn`   | The Sherpa-NN interface libraries                              |
| `ELSE`      | Some scripts to test the C++ NN inference code                 |

The `generic` interface source is configured by setting macros in the various other `interface` subdirectories to produce the interface libraries for different multiplicity and amplitude provider.
The C++ inference code is templated and so can be used with any precision of floating-point number.

The `run` directory contains:

| Subirectory          | Description                                                                            |
| -------------------- | -------------------------------------------------------------------------------------- |
| `Ng2a-njet-fks`      | Scripts to run Sherpa with NJet to generate training datasets                          |
| `Ng2a-nn-fks`        | Scripts to run Sherpa with the pretrained models in `models` to perform the simulation |
| `Ng2a-nn-fks-errors` | As `Ng2a-nn-fks` but keeping track of NN precision/optimality uncertainties            |

The interface libraries in `interface` must be compiled first as the `run` subdirectories contain symlinks to these compiled libraries.
The Rivet analysis in `analysis` is also linked to and must be compiled first.

All dataset generation is performed with `f64`s.
Model training, files, and inference use `f32`s.

## Continuous integration

[`pre-commit`](https://pre-commit.com/) (the program) is used to manage local continuous integration (CI), namely the `git` hook pre-commit.
It is integrated with:

-   Python formatter [`black`](https://github.com/psf/black) ([docs](https://black.readthedocs.io/en/stable/integrations/source_version_control.html))
-   Python linter [`flake8`](https://github.com/pycqa/flake8) ([docs](https://flake8.pycqa.org/en/latest/user/using-hooks.html))
-   YAML formatter [`yamlfmt`](https://github.com/mmlb/yamlfmt) ([hook](https://github.com/jumanjihouse/pre-commit-hook-yamlfmt))
-   YAML linter [`yamllint`](https://github.com/adrienverge/yamllint) ([docs](https://yamllint.readthedocs.io/en/stable/integration.html))

These hooks can be updated with:

```shell
pre-commit autoupdate
```

Checks and formatters can be manually invoked with:

```shell
pre-commit run --all-files
```

This can all be initialised by running `./init.sh`.
You only need to do this once (after you clone the repo).
If the script has a problem with installing `pre-commit`, just [install it manually](https://pre-commit.com/index.html#installation).

On the GitLab remote, which provides remote CI, there is also linting.
This is configured in [.gitlab-ci.yml](.gitlab-ci.yml).
