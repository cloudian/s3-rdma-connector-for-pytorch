#  Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#  Copyright Cloudian, Inc. All Rights Reserved.
#  // SPDX-License-Identifier: BSD

from typing import Any, Callable, Iterable, Optional, Union
import logging

import torch.utils.data
import torch


from ._common import get_objects_from_uris, get_objects_from_prefix, identity
from ._types import S3BucketKey
from .s3client import S3ClientConfig, S3RdmaClient
from .s3reader import S3Reader
from .reader_constructor import S3ReaderConstructor, S3ReaderConstructorProtocol


log = logging.getLogger(__name__)


# pylint: disable=too-many-arguments
class S3IterableDataset(torch.utils.data.Dataset):
    """A Iterator-Style dataset created from S3 objects.

    To create an instance of S3IterableDataset, you need to use
    `from_prefix` or `from_objects` methods.
    """

    def __init__(
        self,
        client: S3RdmaClient,
        transform: Callable[[S3Reader], Any],
        dataset_iter: Iterable[S3BucketKey],
        enable_sharding: bool,
        reader_constructor: Optional[S3ReaderConstructorProtocol] = None,
    ):
        self._client = client
        self._transform = transform
        self._dataset_iter = dataset_iter
        self._enable_sharding = enable_sharding
        self._reader_constructor = reader_constructor or S3ReaderConstructor.default()

        self._rank = 0
        self._world_size = 1
        if torch.distributed.is_initialized():
            self._rank = torch.distributed.get_rank()
            self._world_size = torch.distributed.get_world_size()

    @property
    def rdma_client(self):
        return self._client

    @property
    def s3_client(self):
        return self._client.s3_client

    @classmethod
    def from_objects(
            cls,
            object_uris: Union[str, Iterable[str]],
            *,
            region: Optional[str] = None,
            endpoint: Optional[str] = None,
            transform: Callable[[S3Reader], Any] = identity,
            s3client_config: Optional[S3ClientConfig] = None,
            enable_sharding: bool = False,
            reader_constructor: Optional[S3ReaderConstructorProtocol] = None,
    ):
        log.info("Building RDMA %s from_objects", cls.__name__)
        client = S3RdmaClient(region=region, endpoint=endpoint, s3client_config=s3client_config)
        return cls(client, transform, get_objects_from_uris(object_uris), enable_sharding, reader_constructor)

    @classmethod
    def from_prefix(
            cls,
            s3_uri: str,
            *,
            region: Optional[str] = None,
            endpoint: Optional[str] = None,
            transform: Callable[[S3Reader], Any] = identity,
            s3client_config: Optional[S3ClientConfig] = None,
            enable_sharding: bool = False,
            reader_constructor: Optional[S3ReaderConstructorProtocol] = None,
    ):
        log.info("Building RDMA %s from_prefix %s", cls.__name__, s3_uri)
        client = S3RdmaClient(region=region, endpoint=endpoint, s3client_config=s3client_config)
        return cls(
            client,
            transform,
            get_objects_from_prefix(client, s3_uri),
            enable_sharding,
            reader_constructor,
        )

    def _get_transformed_object(self, bucket_key: S3BucketKey) -> Any:
        return self._transform(self._reader_constructor(self._client, bucket_key))

    def __iter__(self):
        worker_id = 0
        num_workers = 1
        if self._enable_sharding:
            worker_info = torch.utils.data.get_worker_info()
            if worker_info is not None:
                worker_id = worker_info.id
                num_workers = worker_info.num_workers

        if not self._enable_sharding or (self._world_size == 1 and num_workers == 1):
            # sharding disabled or only one shard is available, so return the entire dataset
            return map(self._get_transformed_object, self._dataset_iter)

        # In a multi-process setting (e.g., distributed training), the dataset needs to be
        # sharded across multiple processes. The following variables control this sharding:
        #
        # _rank: The rank (index) of the current process within the world (group of processes).
        # _world_size: The total number of processes in the world (group).
        #
        # In addition, within each process, the dataset may be further sharded across multiple
        # worker threads or processes (e.g., for data loading). The following variables control
        # this intra-process sharding:
        #
        # worker_id: The ID of the current worker thread/process within the process.
        # num_workers: The total number of worker threads/processes within the process.

        # First, distribute objects across ranks
        rank_sharded_objects = (
            obj
            for idx, obj in enumerate(self._dataset_iter)
            if idx % self._world_size == self._rank
        )

        # Then, distribute objects within each rank across workers
        worker_sharded_objects = (
            obj
            for idx, obj in enumerate(rank_sharded_objects)
            if idx % num_workers == worker_id
        )

        return map(self._get_transformed_object, worker_sharded_objects)

    def __getitem__(self, _i: int):
        raise NotImplementedError
