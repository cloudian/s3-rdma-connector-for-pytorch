#  Copyright Cloudian, Inc. All Rights Reserved.
#  // SPDX-License-Identifier: BSD

from .s3client import S3RdmaClient
from ._types import ObjectInfo, S3BucketKey


class BucketLister:
    def __init__(self, client: S3RdmaClient, bucket: str, prefix: str):
        self.client = client
        self.bucket = bucket
        self.prefix = prefix
        self.idx = 0
        self.rsp = None

    def __iter__(self):
        self.rsp = self.client.s3_client.list_objects_v2(Bucket=self.bucket, Prefix=self.prefix, MaxKeys=1000)
        self.idx = 0
        return self

    def __next__(self) -> S3BucketKey:
        if self.idx == self.rsp['KeyCount']:
            if not self.rsp['IsTruncated']:
                raise StopIteration
            self.rsp = self.client.s3_client.list_objects_v2(Bucket=self.bucket, Prefix=self.prefix, MaxKeys=1000,
                                                             ContinuationToken=self.rsp['NextContinuationToken'])
            self.idx = 0
        entry = self.rsp['Contents'][self.idx]
        self.idx += 1
        info = ObjectInfo(entry.get('ETag'), entry['Size'], entry.get('LastModified'), entry.get('StorageClass'), None)
        return S3BucketKey(self.bucket, entry['Key'], info)
