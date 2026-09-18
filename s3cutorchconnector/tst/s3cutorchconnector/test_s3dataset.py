#  Copyright Cloudian, Inc. All Rights Reserved.
#  // SPDX-License-Identifier: BSD

import uuid

from s3cutorchconnector.s3client import S3RdmaClient
from s3cutorchconnector import S3MapDataset, S3IterableDataset


def test_map():
    client = S3RdmaClient()
    bucket_name = f"test-map-dataset-{uuid.uuid4()}"
    client.s3_client.create_bucket(Bucket=bucket_name)

    try:
        contents = [b'data1', b'data contents 2']
        for idx, content in enumerate(contents):
            client.s3_client.put_object(Bucket=bucket_name, Key=f"test/file-{idx}", Body=content)

        dataset = S3MapDataset.from_prefix(f"s3://{bucket_name}/test")
        for i, obj in enumerate(dataset):
            assert obj.read() == contents[i]

    finally:
        for idx in range(len(contents)):
            client.s3_client.delete_object(Bucket=bucket_name, Key=f"test/file-{idx}")
        client.s3_client.delete_bucket(Bucket=bucket_name)


def test_iterable():
    client = S3RdmaClient()
    bucket_name = f"test-iter-dataset-{uuid.uuid4()}"
    client.s3_client.create_bucket(Bucket=bucket_name)

    try:
        contents = [b'data1', b'data contents 2']
        for idx, content in enumerate(contents):
            client.s3_client.put_object(Bucket=bucket_name, Key=f"test/file-{idx}", Body=content)

        dataset = S3IterableDataset.from_prefix(f"s3://{bucket_name}/test")
        for i, obj in enumerate(dataset):
            assert obj.read() == contents[i]

    finally:
        for idx in range(len(contents)):
            client.s3_client.delete_object(Bucket=bucket_name, Key=f"test/file-{idx}")
        client.s3_client.delete_bucket(Bucket=bucket_name)
