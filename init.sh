#!/usr/bin/env bash

echo "Initialising local repo git hooks..."

echo "Checking that pre-commit is installed..."
if ! type pre-commit; then
    echo "Installing pre-commit..."
    python3 -m pip install --user pre-commit
fi

if ! type pre-commit >/dev/null; then
    echo "I tried to install pre-commit but now I can't find it."
    echo "It's up to you now to install it (try `pip install pre-commit`) or add it to your path!"
    echo "Remember to initialise pre-commit for this repo after with `pre-commit install`"
else
    pre-commit install
    echo "Successfully completed initialisation!"
fi

# echo "Checking that clang-format is installed..."
# if ! type clang-format; then
#     echo "Installing clang-format..."
#     python3 -m pip install --user clang-format
# fi
