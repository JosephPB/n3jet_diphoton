# [n3jet_diphoton](https://gitlab.com/JosephPB/n3jet_diphoton)

[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![build status](https://gitlab.com/JosephPB/n3jet_diphoton/badges/main/pipeline.svg?ignore_skipped=true)](https://gitlab.com/JosephPB/n3jet_diphoton/-/pipelines)

[Mirrored on GitHub](https://github.com/JosephPB/n3jet_diphoton)

## Dependencies

This project uses `Python` 3.6+ and `C++14`.
It also uses the following packages, with tested versions quoted.

| Package                                 | Version |
| --------------------------------------- | ------- |
| [NJet](https://bitbucket.org/njet/njet) | 3.0.0   |
| [Eigen](https://eigen.tuxfamily.org)    | 3.3.9   |
| [Sherpa](https://sherpa-team.gitlab.io) | 2.2.8   |
| [Rivet](https://rivet.hepforge.org)     | 2.7.2   |
| [YODA](https://yoda.hepforge.org/)      | 1.7.7   |
| [FastJet](http://fastjet.fr/)           | 3.3.3   |

## Usage

This project has been tested on Linux and MacOS.

Each directory using `C++` code contains a `Makefile` to compile it.
It is sufficient to run `make` in each such directory to build all targets.
Targets can then be run as `./NAME`.
Error messages will display if command line arguments are required.

`Python` code is presented as either:

* a plain Python script `NAME.py`, which can be run from the terminal as `./NAME.py`.
* a [Jupyter notebook](https://jupyter.org/) `NAME.ipynb`. After installing `jupyter`, a notebook server can be started in the directory with `jupyter notebook` to access the files.

The project is laid out as follows

|Directory|Description|
|---|---|
|analysis||
|configs||
|diagrams||
|interface||
|models||
|plotting||
|run||
|timing||
 
## Continuous integration

[`pre-commit`](https://pre-commit.com/) (the program) is used to manage local continuous integration (CI), namely the `git` hook pre-commit.
It is integrated with

-   Python formatter [`black`](https://github.com/psf/black) ([docs](https://black.readthedocs.io/en/stable/version_control_integration.html))
-   Python linter [`flake8`](https://github.com/pycqa/flake8) ([docs](https://flake8.pycqa.org/en/latest/user/using-hooks.html))
-   YAML formatter [`yamlfmt`](https://github.com/mmlb/yamlfmt) ([hook](https://github.com/jumanjihouse/pre-commit-hook-yamlfmt))
-   YAML linter [`yamllint`](https://github.com/adrienverge/yamllint) ([docs](https://yamllint.readthedocs.io/en/stable/integration.html))

These hooks can be updated with

```shell
pre-commit autoupdate
```

Checks and formatters can be manually invoked with

```shell
pre-commit run --all-files
```

This can all be initialised by running `./init.sh`.
You only need to do this once (after you clone the repo).
If the script has a problem with installing `pre-commit`, just [install it manually](https://pre-commit.com/index.html#installation).

On the GitLab remote, which provides remote CI, there is also linting.
This is configured in [.gitlab-ci.yml](.gitlab-ci.yml).
