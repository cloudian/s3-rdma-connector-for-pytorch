#!/bin/bash -e

if [ "$1" = "--version" ]; then
    version=$2
    sed -i "s/0\\.0\\.1/$version/g" pycuobjclient/pyproject.toml
    sed -i "s/0\\.0\\.1/$version/g" s3cutorchconnector/pyproject.toml
fi

source "$(dirname "$0")/venv.sh"

python3 -m pip install --upgrade build
for pkg in pycuobjclient s3cutorchconnector; do
    pushd $pkg
    python -m build
    popd
done
