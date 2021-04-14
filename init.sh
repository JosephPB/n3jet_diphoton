#!/usr/bin/sh

echo "Initialising local repo git hooks..."

echo "Setting git hooks path to .githooks"
git config core.hooksPath .githooks

echo "Checking that black is installed..."
if ! type black; then
    echo "Installing black..."
    python3 -m pip install --user black
fi

echo "Checking that clang-format is installed..."
if ! type clang-format; then
    echo "Installing clang-format..."
    python3 -m pip install --user clang-format
fi

echo "Successfully completed initialisation!"
