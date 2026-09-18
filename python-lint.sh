#!/bin/bash -e

ROOT=$(realpath $(dirname $0))
cd "$ROOT"

source venv.sh
pip install pylint
pip install flake8

echo "Linting s3cutorchconnector"
cd $ROOT/s3cutorchconnector
pylint .
flake8 .
