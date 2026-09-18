# Benchmarking the S3 Connector for PyTorch

This directory holds patches that can be applied to the AWS S3 connector for PyTorch repository to add support to the benchmark suite to use the Cloudian connector for RDMA data transfers.

# Installing

If running  RHEL (or Rocky Linux) 8 openssl3 needs to be installed. This is part of EPEL 8 so that needs to be installed first - see [instructions](https://docs.fedoraproject.org/en-US/epel/getting-started/).
Once EPEL 8 is installed you can install openssl3:

```
sudo dnf install openssl3-libs
```

AWS recommends using `conda` to create a virtual environment

```
conda create -n pytorch-benchmarks python=3.12
conda activate pytorch-benchmarks
```

Before running the install script, ensure you python environment has the Cloudian RDMA connector for PyTorch installed.

Run the `./install.sh` script. This will:

1. Clone the aws s3 connector
2. Apply the patches to enable using the Cloudian RDMA capable RDMA connector
3. Pip install the aws s3 connector benchmark package

Ensure AWS environment variables are set up to point at your RDMA enabled Cloudian HyperStore cluster.

```
export AWS_REGION=<your region>
export AWS_ACCESS_KEY_ID=<access key>
export AWS_SECRET_ACCESS_KEY=<secret key>
export AWS_ENDPOINT_URL=<s3 endpoint url>
```

# Using RDMA in benchmarks

## Dataset benchmarks

Edit `./conf/dataset.yaml` and set the bucket name and region. A new `s3iterabledataset-rdma` dataloader is defined that you can use in the `hydra.sweeper.params.+dataset` to test against the Cloudian RDMA connector for PyTorch. 

```
vim ./conf/dataset.yaml           # 1. edit config
./utils/run_dataset_benchmarks.sh # 2. run scenario
```

## PyTorch Checkpointing benchmarks

Edit `./conf/pytorch_checkpointing.yaml` and set the s3uri (`s3://<bucket>/<prefix>`) and region. A new `s3rdma` storage location can be used in `hydra.swpper.params.+checkpoint.storage` to store checkpoints using the Cloudian RDMA connector for PyTorch.

```
vim ./conf/pytorch_checkpointing.yaml # 1. edit config
./utils/run_checkpoints_benchmarks.sh # 2. run scenario
```

## PyTorch Lightning Checkpointing benchmarks

Edit `./conf/lightning_checkpointing.yaml` and set the s3uri (`s3://<bucket>/<prefix>`) and region. A new `s3rdma` storage location can be used in `hydra.swpper.params.+checkpoint.storage` to store checkpoints using the Cloudian RDMA connector for PyTorch.

```
vim ./conf/lightning_checkpointing.yaml # 1. edit config
./utils/run_lighning_benchmarks.sh      # 2. run scenario
```


# PyTorch’s Distributed Checkpointing (DCP) benchmarks

Edit `./conf/dcp.yaml` and set the s3uri (`s3://<bucket>/<prefix>`) and region. A new `s3rdma` storage location can be used in `hydra.swpper.params.+checkpoint.storage` to store checkpoints using the Cloudian RDMA connector for PyTorch.

```
vim ./conf/dcp.yaml           # 1. edit config
./utils/run_dcp_benchmarks.sh # 2. run scenario
```