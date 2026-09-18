#  Copyright Cloudian, Inc. All Rights Reserved.
#  // SPDX-License-Identifier: BSD

import uuid

import pytest
from botocore.exceptions import ProfileNotFound

from s3cutorchconnector.s3client import S3ClientConfig, S3RdmaClient
from s3cutorchconnector import S3Checkpoint

CHUNK_SIZE = 10 * 1024 * 1024


# Regression test: S3Checkpoint.__init__ previously called S3RdmaClient(region, endpoint,
# s3client_config) positionally, which binds s3client_config to aws_access_key_id instead -
# silently dropping the config. A bogus profile name only surfaces ProfileNotFound if
# s3client_config was actually forwarded and applied.
def test_checkpoint_forwards_s3client_config():
    with pytest.raises(ProfileNotFound):
        S3Checkpoint(s3client_config=S3ClientConfig(profile="s3cutorchconnector-nonexistent-profile"))


def test_checkpoint():
    client = S3RdmaClient()
    bucket_name = f"test-checkpoint-{uuid.uuid4()}"
    key = "epoch0.ckpt"
    s3_uri = "s3://" + bucket_name + "/" + key
    client.s3_client.create_bucket(Bucket=bucket_name)
    checkpoint = S3Checkpoint()

    try:
        # Below chunk size, test PutObject
        contents = b'data contents 2'

        with checkpoint.writer(s3_uri) as writer:
            writer.write(contents)

        b = bytearray(len(contents))
        with checkpoint.reader(s3_uri) as reader:
            reader.seek(0)
            reader.readinto(b)

        assert b == contents

        # Above chunk size, test multipart upload
        dupe = CHUNK_SIZE // len(contents) + 1

        with checkpoint.writer(s3_uri) as writer:
            for i in range(0, dupe):
                writer.write(contents)

        b = bytearray(len(contents) * dupe)
        with checkpoint.reader(s3_uri) as reader:
            reader.seek(0)
            reader.readinto(b)

        for i in range(0, dupe * len(contents), len(contents)):
            assert b[i:i + len(contents)] == contents

    finally:
        client.s3_client.delete_object(Bucket=bucket_name, Key=key)
        client.s3_client.delete_bucket(Bucket=bucket_name)
