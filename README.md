# Cloudian S3 Connector for PyTorch with RDMA transport

This repository provides an S3 connector, based on the [Amazon S3 Connector for PyTorch](https://github.com/awslabs/s3-connector-for-pytorch), but using RDMA for data transfers instead of TCP.

## S3CuTorchConnector

This package provides pytorch `DataSet` implementations that iterate over objects in a Cloudian S3 bucket, using an `S3RdmaClient` to read objects.
See the [example notebook](examples/demo.ipynb) for an example of how to create data sets.
It also support checkpoints (via `s3cutorchconnector.Checkpoint`, `s3cutorchconnector.lightning.S3LightningCheckpoint` and `dcp` module)

The `s3cutorchconnector.client.S3RdmaClient` depends on our [botocore fork with RDMA extensions](https://github.com/cloudian/botocore), which in turn depends on the Nvidia cuobjclient library, distributed in the CUDA Toolkit (minimum version 13.1.1).

To install the torch connector:

First ensure you have the [CUDA toolkit](https://developer.nvidia.com/cuda-downloads) installed, then run:

```
pip install ./s3cutorchconnector
```

## Testing on non-RDMA capable systems

Set `S3RDMA_CLIENT_ALWAYS_USE_TCP =true` to switch the S3 RDMA client to use TCP instead of RDMA. This will involve an extra copy and should only be used for testing - use the Amazon S3 Connector for PyTorch for non RDMA production systems.
