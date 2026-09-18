#  Copyright Cloudian, Inc. All Rights Reserved.
#  // SPDX-License-Identifier: BSD

import uuid

import pytest
from botocore.exceptions import ProfileNotFound

from s3cutorchconnector.s3client import S3ClientConfig, S3RdmaClient
from s3cutorchconnector.s3reader import S3Reader
from s3cutorchconnector.s3writer import S3Writer
from s3cutorchconnector._types import S3BucketKey


def test_s3client_config_defaults():
    config = S3ClientConfig()
    assert config.max_attempts == 10
    assert config.profile is None
    assert config.requester_pays is False


def test_s3client_config_rejects_requester_pays():
    with pytest.raises(ValueError):
        S3ClientConfig(requester_pays=True)


# botocore only validates a profile lazily, when a client resolves credentials/config from it -
# so a bogus profile name reliably surfaces ProfileNotFound only if it was actually forwarded to
# botocore.session.Session, rather than silently ignored.
def test_profile_is_forwarded_to_session():
    with pytest.raises(ProfileNotFound):
        S3RdmaClient(s3client_config=S3ClientConfig(profile="s3cutorchconnector-nonexistent-profile"))


def test_max_attempts_configures_client_retries():
    client = S3RdmaClient(s3client_config=S3ClientConfig(max_attempts=3))
    # botocore's "legacy" retry mode (the default when no mode is specified) reports
    # total_max_attempts as max_attempts + 1 (the configured value plus the initial attempt).
    assert client.s3_client.meta.config.retries['total_max_attempts'] == 4


# Live-server round trip via our own S3Reader/S3Writer, confirming a client configured with a
# custom max_attempts still works normally.
def test_client_with_custom_max_attempts_round_trips():
    client = S3RdmaClient(s3client_config=S3ClientConfig(max_attempts=3))
    bucket_name = f"test-s3client-config-{uuid.uuid4()}"
    client.s3_client.create_bucket(Bucket=bucket_name)
    s3_uri = f"s3://{bucket_name}/obj"

    try:
        contents = b"hello custom max_attempts"
        with S3Writer(client, s3_uri) as writer:
            writer.write(contents)

        reader = S3Reader(client, S3BucketKey(bucket_name, "obj"))
        try:
            assert reader.read() == contents
        finally:
            reader.close()
    finally:
        client.s3_client.delete_object(Bucket=bucket_name, Key="obj")
        client.s3_client.delete_bucket(Bucket=bucket_name)
