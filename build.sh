#!/bin/bash

if [ -d "submodlib" ]; then
    echo "submodlib already exists. Skipping clone."
else
    echo "Cloning submodlib repository..."
    git clone https://github.com/decile-team/submodlib.git
    cd submodlib
    pip install -e .
    cd ..
fi

