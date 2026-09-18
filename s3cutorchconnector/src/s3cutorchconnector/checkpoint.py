#  Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#  Copyright Cloudian, Inc. All Rights Reserved.
#  // SPDX-License-Identifier: BSD

from typing import Optional


from ._common import parse_s3_uri
from .s3client import S3ClientConfig, S3RdmaClient
from .s3reader import S3Reader
from .s3writer import S3Writer
from ._types import S3BucketKey
from .reader_constructor import S3ReaderConstructor, S3ReaderConstructorProtocol


class S3Checkpoint:
    # pylint: disable=too-many-function-args
    def __init__(
            self,
            region: Optional[str] = None,
            endpoint: Optional[str] = None,
            s3client_config: Optional[S3ClientConfig] = None,
            reader_constructor: Optional[S3ReaderConstructorProtocol] = None):
        self._client = S3RdmaClient(region, endpoint, s3client_config=s3client_config)
        self._reader_constructor = reader_constructor or S3ReaderConstructor.default()

    @property
    def rdma_client(self):
        return self._client

    @property
    def s3_client(self):
        return self._client.s3_client

    def reader(self, s3_uri: str) -> S3Reader:
        """Creates an S3ObjectReader from a given s3_uri."""
        bucket, key = parse_s3_uri(s3_uri)
        bucket_key = S3BucketKey(bucket, key)
        return self._reader_constructor(self._client, bucket_key)

    def writer(self, s3_uri: str) -> S3Writer:
        """Creates an S3ObjectWriter from a given s3_uri."""
        return S3Writer(self._client, s3_uri)
