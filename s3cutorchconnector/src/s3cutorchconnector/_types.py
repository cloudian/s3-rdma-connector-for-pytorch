#  Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#  Copyright Cloudian, Inc. All Rights Reserved.
#  // SPDX-License-Identifier: BSD

from dataclasses import dataclass
from typing import Optional

from .s3client import S3RdmaClient


@dataclass
class RestoreStatus:
    in_progress: bool
    expiry: Optional[int]


@dataclass
class ObjectInfo:
    etag: str
    size: int
    last_modified: int
    storage_class: Optional[str]
    restore_status: Optional[RestoreStatus]


@dataclass
class S3BucketKey:
    bucket: str
    key: str
    object_info: Optional[ObjectInfo] = None

    def ensure_object_info(self, client: S3RdmaClient):
        if self.object_info is None:
            rsp = client.s3_client.head_object(Bucket=self.bucket, Key=self.key)
            self.object_info = ObjectInfo(rsp['ETag'], rsp['ContentLength'], rsp['LastModified'], None, None)
