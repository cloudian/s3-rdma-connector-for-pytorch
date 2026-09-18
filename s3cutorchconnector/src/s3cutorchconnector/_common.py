#  Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#  Copyright Cloudian, Inc. All Rights Reserved.
#  // SPDX-License-Identifier: BSD

from typing import Iterable, Union, Tuple

from .s3client import S3RdmaClient
from ._types import S3BucketKey
from .bucket_lister import BucketLister


def identity(obj: bytes) -> bytes:
    return obj


def parse_s3_uri(uri: str) -> Tuple[str, str]:
    if not uri or not uri.startswith("s3://"):
        raise ValueError("Only s3:// URIs are supported")
    uri = uri[len("s3://"):]
    if not uri:
        raise ValueError("Bucket name must be non-empty")
    split = uri.split("/", maxsplit=1)
    if len(split) == 1:
        bucket = split[0]
        prefix = ""
    else:
        bucket, prefix = split
    if not bucket:
        raise ValueError("Bucket name must be non-empty")
    return bucket, prefix


def get_objects_from_uris(
    object_uris: Union[str, Iterable[str]],
) -> Iterable[S3BucketKey]:
    if isinstance(object_uris, str):
        object_uris = [object_uris]
    bucket_key_pairs = [parse_s3_uri(uri) for uri in object_uris]

    return (S3BucketKey(bucket, key) for bucket, key in bucket_key_pairs)


def get_objects_from_prefix(
    client: S3RdmaClient,
    prefix: str
) -> Iterable[S3BucketKey]:
    bucket, prefix = parse_s3_uri(prefix)
    return BucketLister(client, bucket, prefix)
