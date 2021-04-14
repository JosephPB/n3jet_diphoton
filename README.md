# n3jet_diphoton

[![build status](https://gitlab.example.com/JosephPB/n3jet_diphoton/badges/master/pipeline.svg?ignore_skipped=true)](https://gitlab.com/JosephPB/n3jet_diphoton/-/pipelines)

## Continuous Integration
[`pre-commit`](https://pre-commit.com/) (the program) is used to manage local continuous integration, namely the `git` hook pre-commit.
It is integrated with
* Python formatter `black` ([doc](https://black.readthedocs.io/en/stable/version_control_integration.html))
* Python linter `flake8` ([doc](https://flake8.pycqa.org/en/latest/user/using-hooks.html))
* YAML formatter `yamlfmt` ([hook](https://github.com/jumanjihouse/pre-commit-hook-yamlfmt))
* YAML linter `yamllint` ([doc](https://yamllint.readthedocs.io/en/stable/integration.html))

These hooks can be updated with `pre-commit autoupdate`.

Checks and formatters can be manually invoked with `pre-commit run --all-files`.

This can all be initialised by running `./init.sh`.
You only need to do this once (after you clone the repo).

On the GitLab remote, there is also linting.
This is configured in [.gitlab-ci.yml](.gitlab-ci.yml).
