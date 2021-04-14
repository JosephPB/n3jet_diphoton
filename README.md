# n3jet_diphoton

Test

## Continuous Integration
[`pre-commit`](https://pre-commit.com/) (the program) is used to manage local continuous integration, namely the `git` hook pre-commit.
It is integrated with
* Python autoformatter `black` ([doc](https://black.readthedocs.io/en/stable/version_control_integration.html))
* Python linter `flake8` ([doc](https://flake8.pycqa.org/en/latest/user/using-hooks.html))

These hooks can be updated with `pre-commit autoupdate`.

This can all be initialised by running `./init.sh`.
You only need to do this once (after you clone the repo).

On the GitLab remote, there is also linting.
This is configured in [.gitlab-ci.yml](.gitlab-ci.yml).
