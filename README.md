# [n3jet_diphoton](https://gitlab.com/JosephPB/n3jet_diphoton)

[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![build status](https://gitlab.com/JosephPB/n3jet_diphoton/badges/main/pipeline.svg?ignore_skipped=true)](https://gitlab.com/JosephPB/n3jet_diphoton/-/pipelines)

[Mirrored on GitHub](https://github.com/JosephPB/n3jet_diphoton)

## Continuous Integration
[`pre-commit`](https://pre-commit.com/) (the program) is used to manage local continuous integration, namely the `git` hook pre-commit.
It is integrated with
* Python formatter [`black`](https://github.com/psf/black) ([doc](https://black.readthedocs.io/en/stable/version_control_integration.html))
* Python linter [`flake8`](https://github.com/pycqa/flake8) ([doc](https://flake8.pycqa.org/en/latest/user/using-hooks.html))
* YAML formatter [`yamlfmt`](https://github.com/mmlb/yamlfmt) ([hook](https://github.com/jumanjihouse/pre-commit-hook-yamlfmt))
* YAML linter [`yamllint`](https://github.com/adrienverge/yamllint) ([doc](https://yamllint.readthedocs.io/en/stable/integration.html))

These hooks can be updated with `pre-commit autoupdate`.

Checks and formatters can be manually invoked with `pre-commit run --all-files`.

This can all be initialised by running `./init.sh`.
You only need to do this once (after you clone the repo).
If the script has a problem with installing `pre-commit`, just [install it manually](https://pre-commit.com/index.html#installation).

On the GitLab remote, there is also linting.
This is configured in [.gitlab-ci.yml](.gitlab-ci.yml).
