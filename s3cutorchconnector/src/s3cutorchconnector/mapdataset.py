#  Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#  Copyright Cloudian, Inc. All Rights Reserved.
#  // SPDX-License-Identifier: BSD

from typing import List, Any, Callable, Iterable, Optional, Union
import logging

import torch.utils.data

from ._common import get_objects_from_uris, get_objects_from_prefix, identity
from ._types import S3BucketKey
from .s3client import S3ClientConfig, S3RdmaClient
from .s3reader import S3Reader
from .reader_constructor import S3ReaderConstructor, S3ReaderConstructorProtocol


log = logging.getLogger(__name__)


# pylint: disable=too-many-arguments
class S3MapDataset(torch.utils.data.Dataset):
    """A Map-Style dataset created from S3 objects.

    To create an instance of S3MapDataset, you need to use
    `from_prefix` or `from_objects` methods.
    """

    def __init__(
        self,
        client: S3RdmaClient,
        transform: Callable[[S3Reader], Any],
        dataset_iter: Iterable[S3BucketKey],
        reader_constructor: Optional[S3ReaderConstructorProtocol] = None,
    ):
        self._client = client
        self._transform = transform
        self._dataset_iter = dataset_iter
        self._reader_constructor = reader_constructor or S3ReaderConstructor.default()
        self.dataset = None

    @property
    def rdma_client(self):
        return self._client

    @property
    def s3_client(self):
        return self._client.s3_client

    @property
    def _dataset_bucket_key_pairs(self) -> List[S3BucketKey]:
        if not self.dataset:
            self.dataset = list(self._dataset_iter)
        assert self.dataset is not None
        return self.dataset

    @classmethod
    def from_objects(
            cls,
            object_uris: Union[str, Iterable[str]],
            *,
            region: Optional[str] = None,
            endpoint: Optional[str] = None,
            transform: Callable[[S3Reader], Any] = identity,
            s3client_config: Optional[S3ClientConfig] = None,
            reader_constructor: Optional[S3ReaderConstructorProtocol] = None):
        log.info("Building RDMA %s from_objects", cls.__name__)
        client = S3RdmaClient(region=region, endpoint=endpoint, s3client_config=s3client_config)
        return cls(client, transform, get_objects_from_uris(object_uris), reader_constructor)

    @classmethod
    def from_prefix(
            cls,
            s3_uri: str,
            *,
            region: Optional[str] = None,
            endpoint: Optional[str] = None,
            transform: Callable[[S3Reader], Any] = identity,
            s3client_config: Optional[S3ClientConfig] = None,
            reader_constructor: Optional[S3ReaderConstructorProtocol] = None):
        log.info("Building RDMA %s from_prefix %s", cls.__name__, s3_uri)
        client = S3RdmaClient(region=region, endpoint=endpoint, s3client_config=s3client_config)
        return cls(
            client,
            transform,
            get_objects_from_prefix(client, s3_uri),
            reader_constructor,
        )

    def entry(self, i: int) -> S3Reader:
        return self._dataset_bucket_key_pairs[i]

    def __getitem__(self, i: int) -> Any:
        meta = self.entry(i)
        obj = self._reader_constructor(self._client, meta)
        return self._transform(obj)

    def __len__(self):
        return len(self._dataset_bucket_key_pairs)
