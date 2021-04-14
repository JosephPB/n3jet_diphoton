#!/usr/bin/sh

echo "Initialising local repo git hooks..."

echo "Setting git hooks path to .githooks"
git config core.hooksPath .githooks

echo "Checking that pre-commit is installed..."
if ! type pre-commit; then
    echo "Installing pre-commit..."
    python3 -m pip install --user pre-commit
fi

if ! type pre-commit > /dev/null; then
    echo "I tried to install pre-commit but I can't find it. It's up to you now to install it or add it to your path!"
fi

# echo "Checking that clang-format is installed..."
# if ! type clang-format; then
#     echo "Installing clang-format..."
#     python3 -m pip install --user clang-format
# fi

echo "Successfully completed initialisation!"
