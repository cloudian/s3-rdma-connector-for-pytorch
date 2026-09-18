#!/usr/bin/bash -e
#  Copyright Cloudian, Inc. All Rights Reserved.
#  // SPDX-License-Identifier: BSD

# Clone the aws torch connector
git clone --depth 1 -b v1.3.2 https://github.com/awslabs/s3-connector-for-pytorch.git

cd s3-connector-for-pytorch
patch -p1 < ../aws-connector-rdma-benchmarks.patch

cd ..

pip install -e s3-connector-for-pytorch/s3torchbenchmarking/
ln -sf s3-connector-for-pytorch/s3torchbenchmarking/src src
ln -sf s3-connector-for-pytorch/s3torchbenchmarking/conf conf
ln -sf s3-connector-for-pytorch/s3torchbenchmarking/utils utils
